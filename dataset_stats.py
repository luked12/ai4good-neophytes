#!/usr/bin/env python3
"""Build the per-image statistics table of the neophyte dataset.

The dataset is heavily imbalanced: more than 99% of all pixels are background,
and the six species differ by two orders of magnitude in labelled area. Sampling
and subset selection therefore need to know what is inside a tile *before*
opening it, which is what this table provides -- one row per tile with the pixel
count and area per class, split by phenological phase, plus how much of the tile
is empty orthophoto margin.

``NeophyteDataset`` reads the file for weighted sampling, few-shot selection and
the ``max_nodata_frac`` filter (configured as ``dataset_stats`` in
configs/data/data.yaml), so it has to be regenerated whenever tiles are added.

    python dataset_stats.py                                  # ../data/Neophytes
    python dataset_stats.py --data-root ../data/Neophytes --sanity-run

Expected layout::

    <data-root>/<year>/<site>/<split>/images/*.tif
    <data-root>/<year>/<site>/<split>/masks_prep/*.tif

Each mask carries three bands: class, canopy density, phenology.
"""

import argparse
import os

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm

# Class index -> short name used in the column names of the table
CLASS_SHORT = {
    1: "B_davidii",
    2: "R_typhina",
    3: "R_japonica",
    4: "B_orientalis",
    5: "S_inaequidens",
    6: "A_altissima",
}

# Phenology band values -> bucket name
PHENO_MAP = {0: "unknown", 1: "nonflower", 2: "flower", 3: "fruiting"}
PHENO_ORDER = ["nonflower", "flower", "fruiting", "unknown"]


def load_mask_layers(path):
    """Class, canopy and phenology band of a mask tile, plus the pixel area in m²."""
    with rasterio.open(path) as src:
        res = abs(src.transform[0])
        return src.read(1), src.read(2), src.read(3), res * res


def count_nodata(img_path):
    """(empty pixels, total pixels) of an orthophoto tile.

    Tiles at the edge of an orthophoto are partly empty, and nothing in the mask
    reveals that: the label calls those pixels background. They do carry the
    tile's alpha band, which rasterio exposes through ``dataset_mask()``.
    """
    with rasterio.open(img_path) as src:
        dm = src.dataset_mask()
    return int((dm == 0).sum()), int(dm.size)


def collect_stats(base_dir, out_file, sanity_run=False, max_images_per_split=10):
    """Walk <base_dir>/<year>/<site>/<split>/ and write one CSV row per tile.

    Columns:
        year, site, split, img_path, mask_path
        nodata_px, total_px                  empty orthophoto margin
        background_px, <class>_px            pixels per class
        <phenology>_<class>_px               pixels per class and phase
        background_m2, <class>_m2, ...       the same in m²
        only_background, only_<class>        tile contains just this one class
        exactly_<n>_neophytes                tile contains exactly n species

    ``sanity_run`` limits the walk to a few tiles per split, to check the output
    format without reading the whole dataset.
    """
    if os.path.isdir(out_file):
        out_file = os.path.join(out_file, "stats_imagewise.csv")

    # ---- collect the tile list ------------------------------------------------
    all_rows = []
    for year in sorted(os.listdir(base_dir)):
        ypath = os.path.join(base_dir, year)
        if not os.path.isdir(ypath) or not year.isdigit():
            continue

        for site in sorted(os.listdir(ypath)):
            spath = os.path.join(ypath, site)
            for split in ["train", "val", "test"]:
                masks_path = os.path.join(spath, split, "masks_prep")
                images_path = os.path.join(spath, split, "images")
                if not os.path.isdir(masks_path) or not os.path.isdir(images_path):
                    continue

                mask_files = sorted(f for f in os.listdir(masks_path) if f.lower().endswith(".tif"))
                image_files = sorted(f for f in os.listdir(images_path) if f.lower().endswith(".tif"))
                if len(mask_files) != len(image_files):
                    print(f"Warning: {site}/{split} mask/image count mismatch, skipped!")
                    continue

                if sanity_run:
                    mask_files = mask_files[:max_images_per_split]
                    image_files = image_files[:max_images_per_split]

                for mask_f, img_f in zip(mask_files, image_files):
                    all_rows.append((year, site, split,
                                     os.path.join(masks_path, mask_f),
                                     os.path.join(images_path, img_f)))

    if not all_rows:
        raise SystemExit(f"No <year>/<site>/<split>/images tiles found under {base_dir}")

    # ---- column layout --------------------------------------------------------
    area_cols = [f"{ph}_{cname}_m2" for ph in PHENO_ORDER for cname in CLASS_SHORT.values()]
    pixel_cols = [f"{ph}_{cname}_px" for ph in PHENO_ORDER for cname in CLASS_SHORT.values()]
    class_area_cols = [f"{cname}_m2" for cname in CLASS_SHORT.values()]
    class_px_cols = [f"{cname}_px" for cname in CLASS_SHORT.values()]
    single_class_cols = ["only_background"] + [f"only_{cname}" for cname in CLASS_SHORT.values()]
    exactly_n_cols = [f"exactly_{n}_neophytes" for n in range(2, 7)]
    bool_cols = single_class_cols + exactly_n_cols

    # ---- one row per tile -----------------------------------------------------
    final_rows = []
    for year, site, split, mask_p, img_p in tqdm(all_rows, desc="Processing masks", unit="file"):
        classes, canopy, phenology, pixel_area = load_mask_layers(mask_p)
        # canopy 1 and phenology 4 mark pixels excluded from the annotation
        valid = (canopy != 1) & (phenology != 4)

        row = {"year": year, "site": site, "split": split,
               "img_path": img_p, "mask_path": mask_p}
        for col in area_cols + class_area_cols + pixel_cols + class_px_cols:
            row[col] = 0.0
        for col in bool_cols:
            row[col] = 0

        bg_mask = (classes == 0) & valid
        row["background_px"] = int(bg_mask.sum())
        row["background_m2"] = bg_mask.sum() * pixel_area
        row["nodata_px"], row["total_px"] = count_nodata(img_p)

        for cid, cname in CLASS_SHORT.items():
            class_mask = (classes == cid) & valid
            if not np.any(class_mask):
                continue
            unique_ph, counts = np.unique(phenology[class_mask], return_counts=True)
            for ph_code, count in zip(unique_ph, counts):
                bucket = PHENO_MAP.get(int(ph_code), "unknown")
                row[f"{bucket}_{cname}_px"] += count
                row[f"{bucket}_{cname}_m2"] += count * pixel_area
            row[f"{cname}_px"] = int(class_mask.sum())
            row[f"{cname}_m2"] = class_mask.sum() * pixel_area

        # ---- boolean shortcuts used by the few-shot selection ----
        present = [cname for cname in CLASS_SHORT.values() if row[f"{cname}_px"] > 0]
        row["only_background"] = int(not present)
        for cname in present:
            row[f"only_{cname}"] = int(len(present) == 1)
        for n in range(2, 7):
            row[f"exactly_{n}_neophytes"] = int(len(present) == n)

        final_rows.append(row)

    df = pd.DataFrame(final_rows)[
        ["year", "site", "split", "img_path", "mask_path", "nodata_px", "total_px",
         "background_px"] + class_px_cols + pixel_cols +
        ["background_m2"] + class_area_cols + area_cols + bool_cols
    ]
    df.to_csv(out_file, index=False)

    print(f"\nSaved image-wise stats for {len(df)} tiles to {out_file}")
    print(f"  tiles without any neophyte label: {int(df['only_background'].sum())}")
    for cname in CLASS_SHORT.values():
        print(f"  {cname:<16} {int((df[f'{cname}_px'] > 0).sum()):>6} tiles"
              f"  {df[f'{cname}_m2'].sum():>12,.1f} m²")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="../data/Neophytes",
                    help="dataset root holding <year>/<site>/<split>/ (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output CSV (default: <data-root>/stats_imagewise.csv). It belongs "
                         "with the dataset, since the data configs read it from there.")
    ap.add_argument("--sanity-run", action="store_true",
                    help="only process a few tiles per split, to check the output format")
    args = ap.parse_args()

    collect_stats(args.data_root,
                  args.out or os.path.join(args.data_root, "stats_imagewise.csv"),
                  sanity_run=args.sanity_run)
