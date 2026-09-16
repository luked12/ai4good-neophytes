import os
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from datasets.neophyte_dataset import NeophyteDataset
from utils.utils import ensure_list_values


def _has_elevation(image_dirs):
    """True if any of the given image dirs has a non-empty 'dsm' subfolder."""
    dirs = [image_dirs] if isinstance(image_dirs, str) else list(image_dirs)
    for d in dirs:
        dsm_dir = os.path.join(d, "dsm")
        if os.path.isdir(dsm_dir) and any(
            f.lower().endswith(".tif") for f in os.listdir(dsm_dir)
        ):
            return True
    return False

class NeophyteDataModule(pl.LightningDataModule):
    def __init__(self, image_dirs, classes, transform_train, transform_val, transform_test, dsm_transform=False, ndsm_transform=False, dataset_stats=None, weighted_sampling=False, cap_quantile=None, lift_quantile=None, bg_fraction=None, min_strat_images=10, value_mapping=None, ignore_index=-1, load_auxiliary=False, load_dsm=False, load_dtm=False, load_ndsm=False, few_shot_k=None, few_shot_mode='max_px_site_pheno', few_shot_bg_k=100, max_nodata_frac=None, val_few_shot_k=None, val_few_shot_mode='all', val_few_shot_bg_k=0, pick_few_shot_k=10, pick_few_shot_mode='max_px_site_pheno', pick_few_shot_bg_k=0, load_into_ram=False, batch_size_train=16, batch_size_val=16, batch_size_test=16, num_workers=4):
        super().__init__()
        self.image_dirs = ensure_list_values(image_dirs)
        self.classes = classes
        self.transform_train = transform_train
        self.transform_val = transform_val
        self.transform_test = transform_test
        self.dsm_transform = dsm_transform
        self.ndsm_transform = ndsm_transform
        self.dataset_stats = dataset_stats
        self.weighted_sampling = weighted_sampling
        self.cap_quantile = cap_quantile
        self.lift_quantile = lift_quantile
        self.bg_fraction = bg_fraction
        self.min_strat_images = min_strat_images
        self.value_mapping = value_mapping
        self.ignore_index = ignore_index
        self.load_auxiliary = load_auxiliary
        self.load_dsm = load_dsm
        self.load_dtm = load_dtm
        self.load_ndsm = load_ndsm
        self.few_shot_k = few_shot_k
        self.few_shot_mode = few_shot_mode
        self.few_shot_bg_k = few_shot_bg_k
        self.max_nodata_frac = max_nodata_frac
        self.val_few_shot_k = val_few_shot_k
        self.val_few_shot_mode = val_few_shot_mode
        self.val_few_shot_bg_k = val_few_shot_bg_k
        self.pick_few_shot_k = pick_few_shot_k
        self.pick_few_shot_mode = pick_few_shot_mode
        self.pick_few_shot_bg_k = pick_few_shot_bg_k
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.load_into_ram = load_into_ram

    def setup(self, stage=None):
        # Initialize Datasets
        if stage == "fit" and not hasattr(self, "train_dataset") or stage is None:
            # # Train dataset
            self.train_dataset = NeophyteDataset(self.image_dirs.train, self.classes, transform=self.transform_train, dsm_transform=self.dsm_transform, ndsm_transform=self.ndsm_transform, dataset_stats=self.dataset_stats, weighted_sampling=self.weighted_sampling, cap_quantile=self.cap_quantile, lift_quantile=self.lift_quantile, bg_fraction=self.bg_fraction, min_strat_images=self.min_strat_images, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_auxiliary=self.load_auxiliary, load_dsm=self.load_dsm, load_dtm=self.load_dtm, load_ndsm=self.load_ndsm, few_shot_k=self.few_shot_k, few_shot_mode=self.few_shot_mode, few_shot_bg_k=self.few_shot_bg_k, max_nodata_frac=self.max_nodata_frac, load_into_ram=self.load_into_ram, preload_workers=self.num_workers)
            # # Val dataset. val_few_shot_k restricts it to labelled images, which makes
            # # validation much faster but also removes every background-only tile -- see
            # # the note in configs/data/data.yaml before switching it on.
            self.val_dataset = NeophyteDataset(self.image_dirs.val, self.classes, transform=self.transform_val, dsm_transform=self.dsm_transform, ndsm_transform=self.ndsm_transform, dataset_stats=self.dataset_stats, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_auxiliary=self.load_auxiliary, load_dsm=self.load_dsm, load_dtm=self.load_dtm, load_ndsm=self.load_ndsm, few_shot_k=self.val_few_shot_k, few_shot_mode=self.val_few_shot_mode, few_shot_bg_k=self.val_few_shot_bg_k, load_into_ram=self.load_into_ram, preload_workers=self.num_workers)
            self.data_dims = self.train_dataset[0]['image'].shape
            # Initialize WeightedRandomSampler (only for train dataset)
            if self.weighted_sampling:
                self.sampler = WeightedRandomSampler(self.train_dataset.sample_weights, num_samples=len(self.train_dataset), replacement=True)
        if stage == "test" or stage is None:
            # Initialize Datasets
            self.test_dataset = NeophyteDataset(self.image_dirs.test, self.classes, transform=self.transform_test, dsm_transform=self.dsm_transform, ndsm_transform=self.ndsm_transform, dataset_stats=self.dataset_stats, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_auxiliary=self.load_auxiliary, load_dsm=self.load_dsm, load_dtm=self.load_dtm, load_ndsm=self.load_ndsm, return_dsm_dtm=True)
            # Load ndsm for qualitative plots when elevation data is available; for
            # RGB-only datasets (no dsm/ folder) fall back to RGB+mask only so the
            # pick still works (save_model_predictions drops the ndsm column itself).
            # return_dsm_dtm keeps dsm/dtm in the batch alongside ndsm so models that
            # need those channels still work.
            pick_load_ndsm = _has_elevation(self.image_dirs.test)
            self.pick_dataset = NeophyteDataset(self.image_dirs.test, self.classes, transform=self.transform_test, dsm_transform=self.dsm_transform, ndsm_transform=self.ndsm_transform, dataset_stats=self.dataset_stats, value_mapping=self.value_mapping, ignore_index=self.ignore_index, load_auxiliary=self.load_auxiliary, load_dsm=self.load_dsm, load_dtm=self.load_dtm, load_ndsm=pick_load_ndsm, few_shot_k=self.pick_few_shot_k, few_shot_mode=self.pick_few_shot_mode, few_shot_bg_k=self.pick_few_shot_bg_k, return_dsm_dtm=True)
            self.data_dims = self.test_dataset[0]['image'].shape

    # you could each add: multiprocessing_context='fork', persistent_workers=True, timeout=120
    def _loader_kwargs(self):
        """DataLoader options affecting throughput only, never the data itself.

        Each worker builds a whole batch and batches are yielded in worker order,
        so one expensive batch stalls the pipeline while other workers idle. A
        deeper prefetch queue absorbs that, which matters whenever the cost of an
        augmentation varies from sample to sample.
        """
        if self.num_workers == 0:                      # both need worker processes
            return {}
        return dict(persistent_workers=True, prefetch_factor=4, pin_memory=True)

    def train_dataloader(self):
        if self.weighted_sampling:
            return DataLoader(self.train_dataset, sampler=self.sampler, batch_size=self.batch_size_train, num_workers=self.num_workers, **self._loader_kwargs()) # shuffling automatically in sampler
        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size_train, shuffle=True, num_workers=self.num_workers, **self._loader_kwargs())

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size_val, shuffle=False, num_workers=self.num_workers, **self._loader_kwargs())

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers)
    
    def pick_dataloader(self):
        return DataLoader(self.pick_dataset, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers)
