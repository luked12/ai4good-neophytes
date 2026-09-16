import os
import re
import cv2
import torch
import rasterio
import numpy as np
import pandas as pd
import warnings

from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from omegaconf import ListConfig
from rasterio.errors import NotGeoreferencedWarning
from albumentations import Compose, Normalize
from typing import Tuple, Optional
from tqdm import tqdm

from utils.utils import extract_dataset_name

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

DEFAULT_TILE_GSD_MM = 2.34  # median over the drone tiles; fallback for non-georeferenced input


def _tile_month(path):
    """Acquisition month from the tile filename (YYYY_MM_DD_...), else None."""
    m = re.search(r'(?:^|[/_])(\d{4})_(\d{2})_\d{2}_', os.path.basename(str(path)))
    return int(m.group(2)) if m else None


def _tile_gsd_mm(src):
    """Ground sampling distance of an open rasterio dataset, in mm/px."""
    try:
        gsd = abs(src.transform.a) * 1000.0
        return gsd if gsd > 0 else DEFAULT_TILE_GSD_MM
    except Exception:
        return DEFAULT_TILE_GSD_MM


class NeophyteDataset(Dataset):
    def __init__(self, image_dirs, classes, transform=None, dsm_transform=None, ndsm_transform=None, dataset_stats=None, geo=False, weighted_sampling=None, cap_quantile=None, lift_quantile=None, bg_fraction=None, min_strat_images=10, value_mapping=None, ignore_index=-1, load_auxiliary=False, load_dsm=False, load_dtm=False, load_ndsm=False, few_shot_k=None, few_shot_mode='max_px_site_pheno', few_shot_bg_k=100, max_nodata_frac=None, load_into_ram=False, preload_workers=None, return_dsm_dtm=False):
        self.image_dirs = image_dirs if isinstance(image_dirs, ListConfig) or isinstance(image_dirs, list) else [image_dirs]
        self.classes = classes
        self.num_classes = len(classes)
        self.transform = transform
        self.dsm_transform = dsm_transform # one (identical) transform for DSM and DTM
        self.ndsm_transform = ndsm_transform # separate transform for nDSM (= DSM - DTM)
        self.dataset_stats = dataset_stats
        self.geo = geo
        self.weighted_sampling = weighted_sampling
        self.value_mapping = value_mapping
        self.ignore_index = ignore_index
        self.load_auxiliary = load_auxiliary
        self.load_dsm = load_dsm
        self.load_dtm = load_dtm
        self.load_ndsm = load_ndsm
        if load_ndsm:
            self.load_dsm = True
            self.load_dtm = True
        self.return_dsm_dtm = return_dsm_dtm and load_ndsm  # only meaningful when ndsm is loaded
        self.few_shot_k = few_shot_k # any number or None (default: None -> full set) (only neophyte classes)
        self.few_shot_mode = few_shot_mode
        self.few_shot_bg_k = few_shot_bg_k # number of fewshot background samples
        self.load_into_ram = load_into_ram
        self.preload_workers = preload_workers
        
        # # Collect all image files and their corresponding mask files
        self.image_files = []
        self.mask_files = []
        self.dsm_files = []
        self.dtm_files = []
        for image_dir in self.image_dirs:
            images_subdir = os.path.join(image_dir, "images")
            for root, _, files in sorted(os.walk(images_subdir)):
                self.image_files.extend(sorted([os.path.join(root, f) for f in files if f.lower().endswith(('.tif', '.png', '.jpg'))]))
            masks_subdir = os.path.join(image_dir, "masks_prep")
            if not os.path.isdir(masks_subdir):
                masks_subdir = os.path.join(image_dir, "labels")
            for root, _, files in sorted(os.walk(masks_subdir)):
                self.mask_files.extend(sorted([os.path.join(root, f) for f in files if f.lower().endswith(('.tif', '.png'))]))
            if self.load_dsm:
                dsm_subdir = os.path.join(image_dir, "dsm")
                for root, _, files in sorted(os.walk(dsm_subdir)):
                    self.dsm_files.extend(sorted([os.path.join(root, f) for f in files if f.lower().endswith('.tif')]))
            if self.load_dtm:
                dtm_subdir = os.path.join(image_dir, "dtm")
                for root, _, files in sorted(os.walk(dtm_subdir)):
                    self.dtm_files.extend(sorted([os.path.join(root, f) for f in files if f.lower().endswith('.tif')]))
        
        # # Ensure the number of images and masks are the same
        assert len(self.image_files) == len(self.mask_files), "Number of images and masks must be the same."
        
        # stats_names in class index order
        self.stats_names = [classes[i]["stats_name"] for i in sorted(classes.keys())]
        
        # background stats name
        self.background_stats_name = classes[0]["stats_name"]
        
        # neophytes only
        self.neophytes_stats_names = [s for s in self.stats_names if s != self.background_stats_name]
        
        # pixel columns in stats CSV
        self.class_px_cols = [class_px_col(stats_name) for stats_name in self.stats_names]
        
        # # Load dataset stats and filter it by images to be used (image_dirs)
        self.stats_df = None
        if self.dataset_stats is not None:
            self.stats_df = pd.read_csv(self.dataset_stats)
        
            # Normalize paths to avoid mismatches
            self.stats_df["img_path"] = self.stats_df["img_path"].apply(os.path.normpath)
            image_files_norm = [os.path.normpath(p) for p in self.image_files]
        
            # Filter CSV to current split
            self.stats_df = self.stats_df[
                self.stats_df["img_path"].isin(image_files_norm)
            ].copy()
        
            # Ensure same ordering as image_files
            self.stats_df = self.stats_df.set_index("img_path").loc[image_files_norm]
            
            # --- remove images w/o a single pixel in any valid class ---
            total_px = self.stats_df[self.class_px_cols].sum(axis=1)
            valid_mask = total_px > 0

            if not valid_mask.all():
                removed_paths = self.stats_df.index[~valid_mask].tolist()

                print(f"\nFiltering {len(removed_paths)} images w/o a single pixel in any valid class:")
                for path in removed_paths:
                    print(f"  removed: {path}")

                self._filter_by(valid_mask)

            # --- drop mostly-empty orthophoto-edge tiles, but NEVER a labelled one ---
            # A tile can be 40% nodata and still carry a hand-labelled neophyte.
            # Real labels are the scarce resource here, so they always win; the
            # filter only thins out the background pool.
            if max_nodata_frac is not None:
                labelled = self.stats_df[[class_px_col(s) for s in
                                          self.neophytes_stats_names]].sum(axis=1) > 0
                edge = self._nodata_frac() > max_nodata_frac
                keep = labelled | ~edge
                print(f"Dropping {int((~keep).sum())} of {len(keep)} tiles with "
                      f"nodata_frac > {max_nodata_frac} (orthophoto edges); "
                      f"kept {int((edge & labelled).sum())} edge tiles that carry real labels")
                self._filter_by(keep)

            assert len(self.stats_df) == len(self.image_files), (
                "Filtered dataset_stats does not match dataset length."
            )
        
        # # optional few-shot subset of the split
        if few_shot_k is not None:
            self.apply_few_shot(few_shot_k, few_shot_mode, few_shot_bg_k=few_shot_bg_k)
        
        # # Calculate sample weights
        self.sample_weights = None
        if weighted_sampling:
            self.sample_weights = self.calculate_sample_weights(weighted_sampling=weighted_sampling, cap_quantile=cap_quantile, lift_quantile=lift_quantile, bg_fraction=bg_fraction, min_strat_images=min_strat_images)

        # # set dsm noDataValue and normalization mean
        self.dsm_noDataValue = -32767
        if dsm_transform:
            self.dsm_mean, _ = get_normalize_params(dsm_transform)
            
        # # preload into RAM
        self.images_ram = None
        self.masks_ram = None
        self.meta_ram = None
        self.canopy_ram = None
        self.phenology_ram = None
        self.dsm_ram = None
        self.dtm_ram = None
        self.ndsm_ram = None
        
        if self.load_into_ram:
            self._preload_data()
    
    
    def _nodata_frac(self):
        """Per-image fraction of empty (orthophoto-edge) pixels, from the stats CSV."""
        if "nodata_px" not in self.stats_df.columns:
            raise KeyError(
                "max_nodata_frac needs the 'nodata_px' and 'total_px' columns in "
                f"{self.dataset_stats}; regenerate it with dataset_stats.py."
            )
        return self.stats_df["nodata_px"] / self.stats_df["total_px"].clip(lower=1)


    def _filter_by(self, keep):
        """Apply a boolean row mask to stats_df and every parallel file list."""
        idx = np.where(np.asarray(keep))[0]
        self.stats_df = self.stats_df.loc[keep]
        self.image_files = [self.image_files[i] for i in idx]
        self.mask_files  = [self.mask_files[i]  for i in idx]
        if self.load_dsm:
            self.dsm_files = [self.dsm_files[i] for i in idx]
        if self.load_dtm:
            self.dtm_files = [self.dtm_files[i] for i in idx]


    def apply_few_shot(
        self,
        few_shot_k,
        few_shot_mode,
        few_shot_bg_k=50,
    ):
        selected = set()
    
        # --- per-class few-shot selection ---
        for class_name in self.neophytes_stats_names:
            
            idxs = select_few_shot_for_class(
                df=self.stats_df,
                class_name=class_name,
                k=few_shot_k,
                mode=few_shot_mode,
            )
            selected.update(idxs)
    
        # --- optional background-only sampling ---
        if few_shot_bg_k > 0:
            bg_col = f"only_{self.background_stats_name}"
            bg_candidates = self.stats_df[
                (self.stats_df[bg_col] == 1) &
                (~self.stats_df.index.isin(selected))
            ]
    
            if not bg_candidates.empty:
                bg_selected = bg_candidates.sample(
                    n=min(few_shot_bg_k, len(bg_candidates)),
                    replace=False,
                ).index.tolist()
    
                selected.update(bg_selected)
    
        # --- filter dataset (file lists and stats_df MUST stay in the same order,
        #     otherwise sample_weights get mapped to the wrong images) ---
        selected = sorted(selected)
        pos = [self.stats_df.index.get_loc(idx) for idx in selected]

        self.stats_df = self.stats_df.loc[selected]

        self.image_files = [self.image_files[i] for i in pos]
        self.mask_files  = [self.mask_files[i]  for i in pos]
        if self.load_dsm:
            self.dsm_files = [self.dsm_files[i] for i in pos]
        if self.load_dtm:
            self.dtm_files = [self.dtm_files[i] for i in pos]


    def calculate_sample_weights(
        self,
        weighted_sampling="inverse",
        beta=0.5,
        cap_quantile=None,
        lift_quantile=None,
        bg_fraction=None,
        min_strat_images=10,
    ):
        if weighted_sampling == "uniform":
            return [1.0] * len(self.stats_df)

        if weighted_sampling == "flat":
            # Equal weight for all neophyte-containing images; background capped via bg_fraction
            group_cols = self.class_px_cols
            presence = self.stats_df[group_cols].to_numpy() > 0
            neo_presence = presence[:, 1:]
            sample_weights = np.zeros(len(presence))
            sample_weights[neo_presence.any(axis=1)] = 1.0
            is_bg = sample_weights == 0
            n_bg = int(is_bg.sum())
            W_neo = float(sample_weights.sum())
            if bg_fraction is not None and n_bg > 0 and W_neo > 0:
                w_bg = bg_fraction * W_neo / ((1.0 - bg_fraction) * n_bg)
            else:
                w_bg = 1.0
            sample_weights[is_bg] = w_bg
            return sample_weights.tolist()

        if weighted_sampling is True:
            weighted_sampling = "inverse"

        _PHENO_MODES = {
            "inverse_pheno", "log_inverse_pheno", "power_inverse_pheno",
            "inverse_image_pheno", "log_inverse_image_pheno", "power_inverse_image_pheno",
        }
        is_pheno = weighted_sampling in _PHENO_MODES

        # --- build group columns and presence matrix ---
        if is_pheno:
            known_phenos = ["nonflower", "flower", "fruiting"]
            bg_col = class_px_col(self.background_stats_name)
            all_pheno_groups = [
                (class_name, pheno, f"{pheno}_{class_name}_px")
                for class_name in self.neophytes_stats_names
                for pheno in known_phenos
                if f"{pheno}_{class_name}_px" in self.stats_df.columns
            ]
            # split into valid (>= min_strat_images) and small (excluded from stratification)
            group_counts = {col: int((self.stats_df[col] > 0).sum()) for _, _, col in all_pheno_groups}
            known_pheno_groups = [(cn, ph, col) for cn, ph, col in all_pheno_groups if group_counts[col] >= min_strat_images]
            small_pheno_groups  = [(cn, ph, col) for cn, ph, col in all_pheno_groups if group_counts[col] <  min_strat_images]
            group_cols = [bg_col] + [col for _, _, col in known_pheno_groups]
        else:
            group_cols = self.class_px_cols

        presence = self.stats_df[group_cols].to_numpy() > 0

        # --- base counts (images or pixels) ---
        if "image" in weighted_sampling:
            base = presence.sum(axis=0).astype(np.float64)
        else:
            base = self.stats_df[group_cols].sum().to_numpy(dtype=np.float64)
        base[base == 0] = 1.0

        # --- raw group weights ---
        if weighted_sampling in ("inverse", "inverse_image", "inverse_pheno", "inverse_image_pheno"):
            group_weights = 1.0 / base
        elif weighted_sampling in ("log_inverse", "log_inverse_image", "log_inverse_pheno", "log_inverse_image_pheno"):
            group_weights = 1.0 / np.log1p(base)
        elif weighted_sampling in ("power_inverse", "power_inverse_image", "power_inverse_pheno", "power_inverse_image_pheno"):
            group_weights = (1.0 / base) ** beta
        else:
            raise ValueError(f"Unknown weighted_sampling mode: {weighted_sampling}")

        # --- cap and lift neophyte weights (quantiles computed on raw neo_w) ---
        neo_w = group_weights[1:].copy()
        q_cap  = np.quantile(neo_w, cap_quantile)  if cap_quantile  is not None else np.inf
        q_lift = np.quantile(neo_w, lift_quantile) if lift_quantile is not None else -np.inf
        neo_w  = np.clip(neo_w, q_lift, q_cap)

        # --- per-sample weights for neophyte-containing images (known phenology) ---
        neo_presence = presence[:, 1:]
        sample_weights = np.zeros(len(presence))
        for i in np.where(neo_presence.any(axis=1))[0]:
            sample_weights[i] = float(neo_w[neo_presence[i]].max())

        # --- pheno mode: handle unknown and small groups ---
        if is_pheno:
            # per-class min/max of valid group weights
            class_min_weight = {}
            class_max_weight = {}
            for class_name in self.neophytes_stats_names:
                indices = [i for i, (cn, _, _) in enumerate(known_pheno_groups) if cn == class_name]
                if indices:
                    class_min_weight[class_name] = float(neo_w[indices].min())
                    class_max_weight[class_name] = float(neo_w[indices].max())

            # unknown pixels: min weight of class (unreliable labels, don't oversample)
            unknown_cols = [
                (class_name, f"unknown_{class_name}_px")
                for class_name in self.neophytes_stats_names
                if f"unknown_{class_name}_px" in self.stats_df.columns
            ]
            if unknown_cols:
                presence_unknown = self.stats_df[[col for _, col in unknown_cols]].to_numpy() > 0
                for i in range(len(sample_weights)):
                    for j, (class_name, _) in enumerate(unknown_cols):
                        if presence_unknown[i, j] and class_name in class_min_weight:
                            sample_weights[i] = max(sample_weights[i], class_min_weight[class_name])

            # small groups (< min_strat_images): max weight of class (treat as rare)
            if small_pheno_groups:
                presence_small = self.stats_df[[col for _, _, col in small_pheno_groups]].to_numpy() > 0
                for i in range(len(sample_weights)):
                    for j, (class_name, _, _) in enumerate(small_pheno_groups):
                        if presence_small[i, j] and class_name in class_max_weight:
                            sample_weights[i] = max(sample_weights[i], class_max_weight[class_name])

        # --- background weight derived from target fraction ---
        # pure background = no neophyte pixels of any phenology (weight still 0 after above)
        is_bg = sample_weights == 0
        n_bg = int(is_bg.sum())
        W_neo = float(sample_weights.sum())
        if bg_fraction is not None and n_bg > 0 and W_neo > 0:
            w_bg = bg_fraction * W_neo / ((1.0 - bg_fraction) * n_bg)
        else:
            if bg_fraction is not None and W_neo == 0:
                print("Warning: no neophyte images found -- bg_fraction ignored, using uniform weights.")
            w_bg = float(group_weights[0])  # raw 1/count fallback
        sample_weights[is_bg] = w_bg

        return sample_weights.tolist()


    def _load_sample(self, i):
        """Load one sample (index ``i``) from disk into a dict of arrays.

        Pure reads plus deterministic (non-random) dsm/ndsm transforms, so it is
        safe to run from several threads at once; rasterio releases the GIL during
        I/O and decode, which is what makes the thread pool in ``_preload_data``
        scale. Returned dicts are collected in input order by ``_preload_data``.
        """
        out = {}

        # image + meta
        with rasterio.open(self.image_files[i]) as src:
            out['image'] = np.stack(src.read([1, 2, 3]), axis=-1)
            out['tile_gsd_mm'] = _tile_gsd_mm(src)
            if self.geo:
                out['meta'] = src.meta

        # mask (+ auxiliary)
        with rasterio.open(self.mask_files[i]) as src:
            mask = src.read(1).astype(np.int64)
            if self.load_auxiliary:
                out['canopy'] = src.read(2).astype(np.uint8)
                out['phenology'] = src.read(3).astype(np.uint8)

        # Modify mask according to value_mapping
        if self.value_mapping:
            dataset_name = extract_dataset_name(self.mask_files[i], self.value_mapping.keys())
            mapping_dict = self.value_mapping.get(dataset_name)
            if mapping_dict:
                for old_value, new_value in mapping_dict.items():
                    mask[mask == int(old_value)] = new_value
        out['mask'] = mask.astype(np.int8)

        # DSM + DTM
        if self.load_dsm:
            with rasterio.open(self.dsm_files[i]) as src:
                dsm_raw = src.read(1).astype(np.float32)
                dsm_nodata = src.nodata if src.nodata is not None else self.dsm_noDataValue
        if self.load_dtm:
            with rasterio.open(self.dtm_files[i]) as src:
                dtm_raw = src.read(1).astype(np.float32)
                dtm_nodata = src.nodata if src.nodata is not None else self.dsm_noDataValue

        # nDSM = DSM - DTM (computed from raw values before normalization)
        if self.load_ndsm:
            # Resize DTM to match DSM resolution if they differ (DTM is often coarser)
            if dtm_raw.shape != dsm_raw.shape:
                dtm_raw = cv2.resize(dtm_raw, (dsm_raw.shape[1], dsm_raw.shape[0]), interpolation=cv2.INTER_LINEAR)
            nodata_mask = (
                (dsm_raw == dsm_nodata) | np.isnan(dsm_raw) |
                (dtm_raw == dtm_nodata) | np.isnan(dtm_raw)
            )
            ndsm = dsm_raw - dtm_raw
            ndsm[nodata_mask] = 0.0  # no height info where either is nodata
            if self.ndsm_transform:
                ndsm = self.ndsm_transform(image=ndsm)['image']
            out['ndsm'] = ndsm
        else:
            if self.load_dsm:
                dsm = dsm_raw.copy()
                if self.dsm_transform:
                    dsm[dsm == dsm_nodata] = self.dsm_mean
                    dsm = self.dsm_transform(image=dsm)['image']
                out['dsm'] = dsm
            if self.load_dtm:
                dtm = dtm_raw.copy()
                if self.dsm_transform:
                    dtm[dtm == dtm_nodata] = self.dsm_mean
                    dtm = self.dsm_transform(image=dtm)['image']
                out['dtm'] = dtm

        return out

    def _preload_data(self):

        self.images_ram = []
        self.masks_ram = []
        self.tile_gsd_ram = []
        self.meta_ram = [] if self.geo else None
        self.canopy_ram = [] if self.load_auxiliary else None
        self.phenology_ram = [] if self.load_auxiliary else None
        self.dsm_ram = [] if (self.load_dsm and not self.load_ndsm) else None
        self.dtm_ram = [] if (self.load_dtm and not self.load_ndsm) else None
        self.ndsm_ram = [] if self.load_ndsm else None

        n = len(self.image_files)
        workers = self.preload_workers or min(32, (os.cpu_count() or 8))
        workers = max(1, min(int(workers), n))
        print(f"Preloading dataset into RAM ({n} samples, {workers} threads)...")

        # ThreadPoolExecutor.map yields results in input order, so the *_ram lists
        # stay aligned with image_files even though loading finishes out of order.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for out in tqdm(ex.map(self._load_sample, range(n)), total=n,
                            desc="Loading samples", unit="img"):
                self.images_ram.append(out['image'])
                self.masks_ram.append(out['mask'])
                self.tile_gsd_ram.append(out['tile_gsd_mm'])
                if self.geo:
                    self.meta_ram.append(out['meta'])
                if self.load_auxiliary:
                    self.canopy_ram.append(out['canopy'])
                    self.phenology_ram.append(out['phenology'])
                if self.load_ndsm:
                    self.ndsm_ram.append(out['ndsm'])
                else:
                    if self.load_dsm:
                        self.dsm_ram.append(out['dsm'])
                    if self.load_dtm:
                        self.dtm_ram.append(out['dtm'])

        print("Finished preloading dataset.")


    def __len__(self):
        return len(self.image_files)


    def __getitem__(self, idx):
        image = None
        mask = None
        canopy = None
        phenology = None
        dsm = None
        dtm = None
        ndsm = None

        # # Load from RAM =====================================================
        if self.load_into_ram:

            image = self.images_ram[idx].copy()
            mask = self.masks_ram[idx].astype(np.int64)
            tile_gsd = self.tile_gsd_ram[idx]

            meta = self.meta_ram[idx] if self.geo else {}

            if self.load_auxiliary:
                canopy = self.canopy_ram[idx].copy()
                phenology = self.phenology_ram[idx].copy()

            if self.load_ndsm:
                ndsm = self.ndsm_ram[idx].copy()
            else:
                if self.load_dsm:
                    dsm = self.dsm_ram[idx].copy()
                if self.load_dtm:
                    dtm = self.dtm_ram[idx].copy()
            
        # # Load from disk ====================================================       
        else:
            
            try:
                with rasterio.open(self.image_files[idx]) as src:
                    image = src.read([1, 2, 3])
                    image = np.stack(image, axis=-1)
                    tile_gsd = _tile_gsd_mm(src)
                    if self.geo:
                        meta = src.meta
                    else:
                        meta = {}
            except rasterio.errors.RasterioIOError as e:
                print(f"Failed to read image at index {idx}: {self.image_files[idx]}")
                raise e  # R
            
            if image.dtype !='uint8':
                print(self.image_files[idx], image.dtype)
    
            # Load mask and additional infos (e.g. canopy / phenology) based on file type 
            try:
                with rasterio.open(self.mask_files[idx]) as src:
                    mask = src.read(1).astype(np.int64)
                    if self.load_auxiliary:
                        canopy = src.read(2).astype(np.int64)
                        phenology = src.read(3).astype(np.int64)
            except rasterio.errors.RasterioIOError as e:
                print(f"Failed to read mask at index {idx}: {self.mask_files[idx]}")
                raise e  # R
                
            # Modify mask according to value_mapping
            if self.value_mapping:
                dataset_name = extract_dataset_name(self.mask_files[idx], self.value_mapping.keys())
                mapping_dict = self.value_mapping.get(dataset_name)
                if mapping_dict:
                    for old_value, new_value in mapping_dict.items():
                        mask[mask == int(old_value)] = new_value
            
            # DSM / DTM / nDSM
            if self.load_dsm:
                with rasterio.open(self.dsm_files[idx]) as src:
                    dsm_raw = src.read(1).astype(np.float32)
                    dsm_nodata = src.nodata if src.nodata is not None else self.dsm_noDataValue
            if self.load_dtm:
                with rasterio.open(self.dtm_files[idx]) as src:
                    dtm_raw = src.read(1).astype(np.float32)
                    dtm_nodata = src.nodata if src.nodata is not None else self.dsm_noDataValue

            if self.load_ndsm:
                if dtm_raw.shape != dsm_raw.shape:
                    dtm_raw = cv2.resize(dtm_raw, (dsm_raw.shape[1], dsm_raw.shape[0]), interpolation=cv2.INTER_LINEAR)
                nodata_mask = (
                    (dsm_raw == dsm_nodata) | np.isnan(dsm_raw) |
                    (dtm_raw == dtm_nodata) | np.isnan(dtm_raw)
                )
                ndsm = dsm_raw - dtm_raw
                ndsm[nodata_mask] = 0.0
                if self.ndsm_transform:
                    augmented = self.ndsm_transform(image=ndsm)
                    ndsm = augmented['image']
                if self.return_dsm_dtm:
                    dsm = dsm_raw.copy()
                    if self.dsm_transform:
                        dsm[dsm == dsm_nodata] = self.dsm_mean
                        augmented = self.dsm_transform(image=dsm)
                        dsm = augmented['image']
                    dtm = dtm_raw.copy()
                    if self.dsm_transform:
                        dtm[dtm == dtm_nodata] = self.dsm_mean
                        augmented = self.dsm_transform(image=dtm)
                        dtm = augmented['image']
            else:
                if self.load_dsm:
                    dsm = dsm_raw.copy()
                    if self.dsm_transform:
                        dsm[dsm == dsm_nodata] = self.dsm_mean
                        augmented = self.dsm_transform(image=dsm)
                        dsm = augmented['image']
                if self.load_dtm:
                    dtm = dtm_raw.copy()
                    if self.dsm_transform:
                        dtm[dtm == dtm_nodata] = self.dsm_mean
                        augmented = self.dsm_transform(image=dtm)
                        dtm = augmented['image']
                    
        
        # # Apply transformations =============================================
        if self.transform:
            # Per-tile side information for augmentations that need it: albumentations
            # has no hook for it, so the dataset pushes it in before the call. The drone
            # GSD ranges 1.5-3.0 mm/px, so anything scale-aware (e.g. pasting objects at
            # a realistic size) has to read the tile's own resolution rather than an average.
            from utils.transform_utils import set_tile_context
            set_tile_context(self.transform, gsd=tile_gsd,
                             month=_tile_month(self.image_files[idx]))
            augmented = self.transform(image=image, mask=mask, canopy=canopy, phenology=phenology, dsm=dsm, dtm=dtm, ndsm=ndsm)
            image = augmented['image']
            mask = augmented['mask']

            if self.load_auxiliary:
                canopy = augmented['canopy']
                phenology = augmented['phenology']

            if self.load_ndsm:
                ndsm = augmented['ndsm']
            if self.return_dsm_dtm or not self.load_ndsm:
                if self.load_dsm:
                    dsm = augmented['dsm']
                if self.load_dtm:
                    dtm = augmented['dtm']

        # # Create sample =====================================================
        sample = {'image': image,
                  'mask': mask.long(),
                  'meta': meta,
                  'name': self.image_files[idx],
                  }
        if self.load_auxiliary:
            sample['phenology'] = phenology
            sample['canopy'] = canopy.float()

        if self.load_ndsm:
            sample['ndsm'] = ndsm.unsqueeze(0)
        if self.return_dsm_dtm or not self.load_ndsm:
            if self.load_dsm:
                sample['dsm'] = dsm.unsqueeze(0)
            if self.load_dtm:
                sample['dtm'] = dtm.unsqueeze(0)
            
        return sample
    
    

def get_normalize_params(compose: Compose) -> Tuple[Optional[float], Optional[float]]:
    """
    Extracts the mean and std from the first Normalize transform in an Albumentations Compose.

    Args:
        compose (Compose): Albumentations Compose object.

    Returns:
        Tuple[Optional[float], Optional[float]]: (mean, std) if Normalize is found, else (None, None)
    """
    for t in compose.transforms:
        if isinstance(t, Normalize):
            return t.mean, t.std
    # If no Normalize found
    return None, None



def class_px_col(class_name):
    return f"{class_name}_px"

def only_col(class_name):
    return f"only_{class_name}"

def pheno_px_cols(class_name):
    return {
        "nonflower": f"nonflower_{class_name}_px",
        "flower": f"flower_{class_name}_px",
        "fruiting": f"fruiting_{class_name}_px",
    }

def select_few_shot_for_class(
    df,
    class_name,
    k,
    mode,
    site_col="site",
):
    px_col = class_px_col(class_name)
    only_flag = only_col(class_name)
    pheno_cols = pheno_px_cols(class_name)

    # --- strict filtering ---
    candidates = df[
        (df[only_flag] == 1) &
        (df[px_col] > 0)
    ].copy()

    if candidates.empty:
        return []

    # --- ranking score ---
    if mode == "random":
        candidates = candidates.sample(frac=1.0)
    elif mode == "all":
        candidates = df[df[px_col] > 0].copy() # w/o only_col restriction
    else:
        candidates = candidates.sort_values(px_col, ascending=False)
        

    # --- no constraint ---
    if mode in {"all", "random", "max_px"}:
        return candidates.head(k).index.tolist()

    # --- define grouping key ---
    if mode == "max_px_site":
        def key_fn(row):
            return (row[site_col],)

    elif mode == "max_px_pheno":
        def key_fn(row):
            for p, col in pheno_cols.items():
                if row[col] > 0:
                    return (p,)
            return None  # unknown → ignored

    elif mode == "max_px_site_pheno":
        def key_fn(row):
            for p, col in pheno_cols.items():
                if row[col] > 0:
                    return (row[site_col], p)
            return None

    else:
        raise ValueError(f"Unknown few-shot mode: {mode}")

    # --- build groups ---
    groups = {}

    for idx, row in candidates.iterrows():
        key = key_fn(row)
        if key is None:
            continue
        groups.setdefault(key, []).append(idx)

    if not groups:
        # No grouping key available (e.g. phenology unknown for every candidate, as
        # for datasets without phenology labels). Fall back to top-k by pixel count
        # so the selection is non-empty instead of silently returning nothing.
        return candidates.head(k).index.tolist()

    # --- fair round-robin sampling ---
    return fair_sample(groups, k)


def fair_sample(groups, k):
    """
    groups: dict[key -> list[idx]] (lists already sorted by priority)
    """
    selected = []
    i = 0

    while len(selected) < k:
        added = False
        for items in groups.values():
            if i < len(items):
                selected.append(items[i])
                added = True
                if len(selected) == k:
                    return selected
        if not added:
            break
        i += 1

    return selected
