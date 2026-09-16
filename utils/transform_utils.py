import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Spatial transforms: rearrange or resample pixels (need interpolation control)
SPATIAL_TRANSFORM_TYPES = {
    'RandomResizedCrop', 'TargetedRandomResizedCrop', 'Resize', 'RandomScale', 'ShiftScaleRotate',
    'Affine', 'PiecewiseAffine', 'ElasticTransform', 'OpticalDistortion',
    'GridDistortion', 'Perspective',
    'RandomRotate90', 'HorizontalFlip', 'VerticalFlip', 'Flip', 'Transpose',
}


class SplitTransform:
    """Two-stage transform that gives float_mask targets bilinear interpolation
    during spatial transforms while keeping integer masks nearest-neighbor.

    Stage 1 (spatial): float_mask targets registered as 'image' type
        → RandomResizedCrop etc. use the image interpolation mode (bilinear/cubic)
        → integer masks still use nearest-neighbor
    Stage 2 (pixel):  float_mask targets registered as 'mask' type
        → ColorJitter, CLAHE, Normalize skip them (only affect 'image' targets)
        → ToTensorV2 converts everything to tensors
    """

    def __init__(self, spatial_compose, pixel_compose, float_mask_keys):
        self.spatial = spatial_compose
        self.pixel = pixel_compose
        self.float_mask_keys = float_mask_keys

    def set_tile_context(self, **ctx):
        """Forward per-tile side information to transforms that ask for it."""
        set_tile_context(self.spatial, **ctx)
        set_tile_context(self.pixel, **ctx)

    def __call__(self, **kwargs):
        if self.spatial is not None:
            for k in self.float_mask_keys:
                if k in kwargs and kwargs[k] is not None:
                    kwargs[k] = kwargs[k][:, :, np.newaxis]

            kwargs = self.spatial(**kwargs)

            for k in self.float_mask_keys:
                if k in kwargs and kwargs[k] is not None:
                    kwargs[k] = kwargs[k][:, :, 0]

        if self.pixel is not None:
            kwargs = self.pixel(**kwargs)

        return kwargs


def set_tile_context(target, **ctx):
    """Push per-tile side information into every transform under `target`.

    Albumentations has no hook for per-sample side information, so the dataset
    pushes it in before the call instead. Handles a SplitTransform, a plain
    A.Compose and nested containers alike. Safe with num_workers>0: each worker
    holds its own copy of the dataset and its transforms.
    """
    if target is None:
        return
    if isinstance(target, SplitTransform):
        target.set_tile_context(**ctx)
        return
    for t in getattr(target, 'transforms', []):
        if hasattr(t, 'set_tile_context'):
            t.set_tile_context(**ctx)
        else:
            set_tile_context(t, **ctx)  # nested Compose / OneOf / Sequential


def get_transforms(transform_config, additional_targets={}, toTensor=True):
    # Detect float_mask targets
    float_mask_keys = {k for k, v in additional_targets.items() if v == 'float_mask'}

    # Build target dicts for Albumentations (only supports 'image' and 'mask')
    # Spatial step: float_mask → 'image' (bilinear), rest unchanged
    # Pixel step: float_mask → 'mask' (skipped by color/normalize transforms)
    spatial_targets = {k: ('image' if k in float_mask_keys else v)
                       for k, v in additional_targets.items() if v != 'float_mask'}
    spatial_targets.update({k: 'image' for k in float_mask_keys})
    pixel_targets = {k: ('mask' if v == 'float_mask' else v)
                     for k, v in additional_targets.items()}

    # Build transform instances
    all_transforms = []
    for transform in transform_config:
        transform_type = transform['type']
        params = transform.get('params', {})
        try:
            transform_class = getattr(A, transform_type)
        except AttributeError:
            from utils import crop_transforms
            transform_class = getattr(crop_transforms, transform_type, None)
            if transform_class is None:
                raise ValueError(f"Transform '{transform_type}' not found in Albumentations "
                                 f"or utils.crop_transforms. Check if it was removed or "
                                 f"renamed in albumentations v2.0+.")
        try:
            all_transforms.append((transform_type, transform_class(**params)))
        except Exception as e:
            raise ValueError(f"Failed to instantiate '{transform_type}' with params {params}: {e}")

    if toTensor:
        all_transforms.append(('ToTensorV2', ToTensorV2()))

    # If no float_mask targets or no spatial transforms → single Compose (unchanged behavior)
    has_spatial = any(name in SPATIAL_TRANSFORM_TYPES for name, _ in all_transforms)
    if not float_mask_keys or not has_spatial:
        targets = pixel_targets if additional_targets else {}
        return A.Compose([t for _, t in all_transforms], additional_targets=targets)

    # Split into spatial + pixel Compose, wrap in SplitTransform
    spatial_list = [t for name, t in all_transforms if name in SPATIAL_TRANSFORM_TYPES]
    pixel_list = [t for name, t in all_transforms if name not in SPATIAL_TRANSFORM_TYPES]

    spatial_compose = A.Compose(spatial_list, additional_targets=spatial_targets) if spatial_list else None
    pixel_compose = A.Compose(pixel_list, additional_targets=pixel_targets) if pixel_list else None

    return SplitTransform(spatial_compose, pixel_compose, float_mask_keys)
