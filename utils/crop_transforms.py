import math

import numpy as np
from albumentations.augmentations.crops import functional as fcrops
from albumentations.augmentations.crops.transforms import RandomResizedCrop


class TargetedRandomResizedCrop(RandomResizedCrop):
    """RandomResizedCrop that anchors the crop window on foreground pixels.

    Standard RandomResizedCrop places the crop window uniformly at random, so
    sparse classes (only a few plants per tile, e.g. Senecio / S. inaequidens)
    are frequently cropped away entirely and disappear from the training crop.

    This variant keeps the *exact* same area (`scale`) and aspect (`ratio`)
    sampling as RandomResizedCrop, but with probability `targeted_prob` -- when a
    foreground pixel exists in the mask -- it anchors the crop window on a
    foreground pixel. "Foreground" = any pixel that is not `background` and not
    `ignore_index`; which class it belongs to does not matter (with one plant per
    tile it is usually unambiguous anyway, and otherwise the choice is random).
    The anchored pixel sits at a random position *inside* the window (not
    centered), so the crop keeps spatial diversity. When no foreground pixel
    exists, or with probability ``1 - targeted_prob``, it falls back to standard
    random placement.

    ``anchor_mode`` controls *which* foreground pixel is chosen:

    - ``"spatial"`` (default): draw a uniform random point over the whole image
      and anchor on the foreground pixel *nearest* to it. Because the point is
      uniform (label-independent), every label region is picked in proportion to
      its spatial Voronoi catchment area rather than its pixel count, so a small
      isolated instance is sampled about as often as a large blob instead of
      being swamped by it.
    - ``"pixel"``: draw a foreground pixel uniformly at random, i.e. anchoring
      is proportional to foreground area, so a dense blob dominates.

    Args:
        size: output (height, width) after resize.
        targeted_prob: probability of doing a targeted crop when foreground is
            present. Otherwise behaves like a normal RandomResizedCrop.
        anchor_mode: ``"spatial"`` (Voronoi-uniform, default) or ``"pixel"``
            (area-weighted).
        anchor_downsample: for ``"spatial"``, factor ``s`` for OR-pooling the
            foreground onto an ``s x s`` grid before the nearest-pixel search.
            Bounds the cost on dense tiles at the price of anchoring within ``s``
            pixels of the true foreground. ``1`` or ``None`` disables pooling.
        background: label value treated as background (excluded from anchoring).
        ignore_index: label value treated as ignore (excluded from anchoring).
        target_classes: optional list of class values to restrict anchoring to.
            ``None`` (default) anchors on any non-background, non-ignore pixel.
        scale: range of area fraction of the source to crop.
        ratio: range of aspect ratios.
        mask_key: key in the data dict holding the integer label mask.
        interpolation / mask_interpolation / p: as in RandomResizedCrop.
    """

    def __init__(self, size, targeted_prob=0.7, anchor_mode="spatial",
                 anchor_downsample=4, background=0, ignore_index=-1,
                 target_classes=None, scale=(0.08, 1.0),
                 ratio=(0.75, 1.3333333333333333), interpolation=1,
                 mask_interpolation=0, mask_key="mask", p=1.0):
        super().__init__(size=size, scale=scale, ratio=ratio,
                         interpolation=interpolation,
                         mask_interpolation=mask_interpolation, p=p)
        if anchor_mode not in ("spatial", "pixel"):
            raise ValueError(
                f"anchor_mode must be 'spatial' or 'pixel', got {anchor_mode!r}")
        self.targeted_prob = targeted_prob
        self.anchor_mode = anchor_mode
        self.anchor_downsample = anchor_downsample
        self.background = background
        self.ignore_index = ignore_index
        self.target_classes = tuple(target_classes) if target_classes else None
        self.mask_key = mask_key

    def _sample_crop_size(self, image_shape):
        """Sample (height, width) from scale/ratio, same logic as the parent."""
        image_height, image_width = image_shape[:2]
        area = image_height * image_width
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = self.py_random.uniform(*self.scale) * area
            aspect_ratio = math.exp(self.py_random.uniform(*log_ratio))
            width = int(round(math.sqrt(target_area * aspect_ratio)))
            height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < width <= image_width and 0 < height <= image_height:
                return height, width
        return None

    def _foreground_mask(self, mask):
        """Boolean mask of anchorable pixels."""
        if self.target_classes is not None:
            return np.isin(mask, self.target_classes)
        # common case: anything that is not background and not ignore_index.
        # for the typical (background=0, ignore_index<0) this is a single `> 0`.
        if self.background == 0 and self.ignore_index is not None and self.ignore_index < 0:
            return mask > 0
        keep = mask != self.background
        if self.ignore_index is not None:
            keep &= mask != self.ignore_index
        return keep

    def _spatial_candidates(self, fgm):
        """Candidate anchor pixels (y, x) for the nearest-point search.

        Without downsampling these are the foreground pixels themselves. With
        ``anchor_downsample = s > 1`` the mask is OR-pooled onto an s x s grid --
        a cell is kept when any pixel inside it is foreground -- and each kept
        cell contributes its centre. This runs in a single vectorised pass over
        the mask regardless of how dense it is, so a fully-foreground tile costs
        the same as a sparse one; dense blobs collapse to few cells while
        isolated pixels each survive as one cell, so no label region is dropped.
        """
        image_height, image_width = fgm.shape
        s = self.anchor_downsample
        if not s or s <= 1:
            return np.nonzero(fgm)
        grid_h, grid_w = -(-image_height // s), -(-image_width // s)
        pad_h, pad_w = grid_h * s - image_height, grid_w * s - image_width
        if pad_h or pad_w:
            fgm = np.pad(fgm, ((0, pad_h), (0, pad_w)))
        coarse = fgm.reshape(grid_h, s, grid_w, s).any(axis=(1, 3))
        cy, cx = np.nonzero(coarse)
        ay = np.minimum(cy * s + s // 2, image_height - 1)
        ax = np.minimum(cx * s + s // 2, image_width - 1)
        return ay, ax

    def get_params_dependent_on_data(self, params, data):
        image_shape = params["shape"][:2]
        image_height, image_width = image_shape
        mask = data.get(self.mask_key)

        # roll for targeted vs. standard placement
        if mask is None or self.py_random.uniform(0.0, 1.0) > self.targeted_prob:
            return super().get_params_dependent_on_data(params, data)

        crop_size = self._sample_crop_size(image_shape)
        if crop_size is None:
            return super().get_params_dependent_on_data(params, data)
        height, width = crop_size

        fgm = self._foreground_mask(mask)

        if self.anchor_mode == "spatial":
            # draw a uniform random point and anchor on the foreground pixel
            # nearest to it, so each region is picked in proportion to its
            # spatial catchment rather than its pixel count.
            ay, ax = self._spatial_candidates(fgm)
            if ay.size == 0:
                return super().get_params_dependent_on_data(params, data)
            ry = self.py_random.uniform(0.0, image_height)
            rx = self.py_random.uniform(0.0, image_width)
            k = int(np.argmin((ay - ry) ** 2 + (ax - rx) ** 2))
            py, px = int(ay[k]), int(ax[k])
        else:  # "pixel": anchor on a uniformly drawn foreground pixel
            fg = np.flatnonzero(fgm.ravel())
            if fg.size == 0:
                return super().get_params_dependent_on_data(params, data)
            pick = int(fg[self.py_random.randint(0, fg.size - 1)])
            py, px = divmod(pick, image_width)

        # window top-left must keep (py, px) inside; random offset within window
        i_lo, i_hi = max(0, py - height + 1), min(py, image_height - height)
        j_lo, j_hi = max(0, px - width + 1), min(px, image_width - width)
        i = self.py_random.randint(i_lo, i_hi) if i_hi > i_lo else max(0, i_lo)
        j = self.py_random.randint(j_lo, j_hi) if j_hi > j_lo else max(0, j_lo)

        h_start = i / (image_height - height + 1e-10)
        w_start = j / (image_width - width + 1e-10)
        crop_coords = fcrops.get_crop_coords(image_shape, (height, width), h_start, w_start)
        return {"crop_coords": crop_coords}
