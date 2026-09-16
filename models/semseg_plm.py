"""PyTorch-Lightning module for semantic segmentation of neophytes.

``SegmentationModel`` wraps an architecture from segmentation-models-pytorch (SMP)
and adds the pieces this project needs:

- optional fusion of an auxiliary elevation channel (DSM / DTM / nDSM), either by
  concatenating it to the RGB input or by injecting it into a pretrained RGB
  encoder at every scale (``ElevationFusion``),
- patch-wise inference for transformer encoders whose positional embedding fixes
  the input size,
- the usual train/val/test metrics (macro accuracy, IoU, F1, AUROC + per-class F1).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassAUROC, MulticlassF1Score, MulticlassJaccardIndex,
)

from utils.scheduler_utils import get_scheduler


class ElevationFusion(nn.Module):
    """Multi-scale fusion of an auxiliary input (DSM/DTM/nDSM) into encoder features.

    For every encoder stage the auxiliary input is resampled to that stage's
    resolution and projected to a per-channel scale and shift, which modulate the
    encoder features (FiLM-style): ``feat * (1 + gamma) + beta``.

    The theoretical advantage over concatenation is that the RGB encoder keeps its
    pretrained weights: the first conv still sees three channels, whereas a
    concatenated fourth channel changes its shape and throws those weights away.
    The last BatchNorm is zero-initialised, so at the start of training the fusion
    contributes exactly nothing and the model behaves like the RGB baseline.
    Whether that translates into better scores is an empirical question -- both
    modes are available (``fusion_mode``) so they can be compared.
    """

    def __init__(self, aux_channels, encoder_out_channels):
        super().__init__()
        self.projections = nn.ModuleList()
        for i, ch in enumerate(encoder_out_channels):
            if i == 0 or ch == 0:
                # Skip stage 0 (raw input) and dummy stages with 0 channels (e.g. MiT encoders)
                self.projections.append(None)
            else:
                hidden = max(aux_channels * 4, 16)
                bn = nn.BatchNorm2d(2 * ch)
                nn.init.zeros_(bn.weight)  # zero-init gamma: fusion adds nothing at start
                self.projections.append(nn.Sequential(
                    nn.Conv2d(aux_channels, hidden, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden, 2 * ch, kernel_size=1, bias=False),
                    bn,
                ))

    def forward(self, aux, encoder_features):
        fused = []
        for feat, proj in zip(encoder_features, self.projections):
            if proj is None:
                fused.append(feat)
            else:
                h, w = feat.shape[2:]
                aux_down = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=False)
                gamma, beta = proj(aux_down).chunk(2, dim=1)
                fused.append(feat * (1 + gamma) + beta)
        return fused

    def get_fusion_scales(self):
        """Mean absolute BN weight per active stage -- a proxy for how much the
        model actually leans on elevation at that scale. Logged during training."""
        return {f"elev_fusion_scale_stage{i}": proj[-1].weight.abs().mean().item()
                for i, proj in enumerate(self.projections) if proj is not None}


class SegmentationModel(pl.LightningModule):
    """Lightning wrapper around an SMP segmentation network.

    Args:
        model: SMP architecture name, e.g. ``Unet``, ``DeepLabV3+``, ``SegFormer``.
        encoder_name: SMP/timm encoder, e.g. ``mit_b2`` or ``tu-resnet50``.
        img_size: training crop size; only used when ``patch_2_img_size`` is set.
        num_classes: number of classes including background.
        learning_rate: initial LR of the optimizer.
        in_channels: dict of input name -> channel count, e.g.
            ``{'image': 3, 'ndsm': 1}``. The keys must exist in the batch.
        fusion_mode: ``concat`` (all channels into the encoder, pretrained stem is
            re-initialised) or ``stem`` (RGB encoder untouched, auxiliary fused via
            :class:`ElevationFusion`).
        patch_2_img_size: if True, inputs larger than ``img_size`` are split into
            overlapping patches, pushed through the model and stitched back
            together. Needed for encoders with a fixed positional embedding.
    """

    def __init__(self, model, encoder_name, img_size, num_classes, learning_rate,
                 in_channels=None, ignore_index=-1, optimizer='AdamW', lr_scheduler=None,
                 loss='CE', weight=None, patch_2_img_size=False, fusion_mode='concat'):
        super().__init__()
        self.save_hyperparameters()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fusion_mode = fusion_mode

        # # Determine encoder input channels based on fusion mode
        if fusion_mode == 'stem':
            # RGB-only encoder (preserves pretrained weights), aux fused via ElevationFusion
            num_channels = in_channels.get('image', 3)
            aux_channels = sum(v for k, v in in_channels.items() if k != 'image')
        else:
            # Default: concatenate all channels as input
            num_channels = sum(in_channels.values())

        # # Load model architecture
        smp_models = {
            'Unet': smp.Unet, 'Unet++': smp.UnetPlusPlus, 'DeepLabV3+': smp.DeepLabV3Plus,
            'SegFormer': smp.Segformer, 'DPT': smp.DPT, 'UPerNet': smp.UPerNet,
            'MAnet': smp.MAnet, 'PAN': smp.PAN,
        }
        if model not in smp_models:
            raise ValueError(f"Unknown model '{model}'. Available: {sorted(smp_models)} "
                             f"(many more exist in segmentation-models-pytorch).")
        self.model = smp_models[model](encoder_name=encoder_name, in_channels=num_channels,
                                       classes=num_classes)

        # # Build elevation fusion module when using stem fusion
        self.elevation_fusion = None
        if fusion_mode == 'stem' and aux_channels > 0:
            self.elevation_fusion = ElevationFusion(aux_channels, self.model.encoder.out_channels)

        # # Loss
        if weight:
            weight = torch.tensor(weight)
        if loss == 'CE':
            self.criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        elif loss == 'Focal':
            # down-weights easy pixels; the background class dominates by orders of magnitude
            self.criterion = smp.losses.FocalLoss(mode='multiclass', ignore_index=ignore_index)
        else:
            raise ValueError(f"Unknown loss '{loss}'. Available: CE, Focal.")

        # # Metrics
        for split in ["train", "val", "test"]:
            setattr(self, f"{split}_acc", MulticlassAccuracy(num_classes=num_classes, average='macro', ignore_index=ignore_index))
            setattr(self, f"{split}_iou", MulticlassJaccardIndex(num_classes=num_classes, average='macro', ignore_index=ignore_index))
            setattr(self, f"{split}_f1", MulticlassF1Score(num_classes=num_classes, average='macro', ignore_index=ignore_index))
            setattr(self, f"{split}_auc", MulticlassAUROC(num_classes=num_classes, average='macro', ignore_index=ignore_index, thresholds=5))
            setattr(self, f"{split}_f1_per_class", MulticlassF1Score(num_classes=num_classes, average=None, ignore_index=ignore_index))

    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------
    def set_patch_2_img_size(self, new_patch_2_img_size):
        self.hparams.patch_2_img_size = new_patch_2_img_size

    def configure_optimizers(self):
        if self.hparams.optimizer == 'Adam':
            optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.hparams.optimizer == 'AdamW':
            optimizer = optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        else:
            raise ValueError(f"Unknown optimizer '{self.hparams.optimizer}'. Available: Adam, AdamW.")
        if self.hparams.lr_scheduler:
            scheduler = get_scheduler(optimizer, self.hparams.lr_scheduler)
            interval = self.hparams.lr_scheduler.get("interval", "epoch")
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": interval}}
        return optimizer

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def prepare_inputs(self, batch):
        """Split a batch into (main input, auxiliary input) according to fusion_mode.

        Returns ``(input_tensor, aux_tensor_or_None)``.
        """
        if self.fusion_mode == 'stem':
            rgb = batch['image']
            aux_parts = [batch[k] for k in self.in_channels if k != 'image']
            aux = torch.cat(aux_parts, dim=1) if aux_parts else None
            return rgb, aux
        if self.in_channels:
            inputs = []
            for key in self.in_channels:
                if key not in batch:
                    raise KeyError(f"Expected key '{key}' in batch but not found.")
                inputs.append(batch[key])
            return torch.cat(inputs, dim=1), None
        return batch['image'], None

    def _forward_stem(self, rgb, aux):
        """RGB through the pretrained encoder, auxiliary fused at every scale."""
        features = self.model.encoder(rgb)
        if self.elevation_fusion is not None and aux is not None:
            features = self.elevation_fusion(aux, features)
        decoder_output = self.model.decoder(features)
        return self.model.segmentation_head(decoder_output)

    def _forward_with_patches(self, x, aux=None):
        batch_size, channels, img_size, _ = x.shape
        patches, num_patches, stride, pad = patch_image_to_batch(x, self.hparams.img_size)
        aux_patches = patch_image_to_batch(aux, self.hparams.img_size)[0] if aux is not None else None

        if self.fusion_mode == 'stem':
            output_patches = self._forward_stem(patches, aux_patches)
        else:
            output_patches = self.model(patches)
        return merge_output_patches_to_image(output_patches, num_patches, self.hparams.img_size,
                                             stride, batch_size, img_size, pad)

    def _forward(self, x, aux=None):
        """Raw logits -- no final activation, the losses apply their own."""
        if x.shape[-1] != self.hparams.img_size and self.hparams.patch_2_img_size:
            return self._forward_with_patches(x, aux)
        if self.fusion_mode == 'stem':
            return self._forward_stem(x, aux)
        return self.model(x)

    def forward(self, x, aux=None):
        """Inference forward: per-pixel class probabilities (B, K, H, W)."""
        return torch.softmax(self._forward(x, aux), dim=1)

    # -------------------------------------------------------------------------
    # train / val / test
    # -------------------------------------------------------------------------
    def _step(self, batch, stage):
        masks = batch['mask']
        inputs, aux = self.prepare_inputs(batch)
        logits = self._forward(inputs, aux)
        loss = self.criterion(logits, masks.long())
        preds = torch.argmax(logits, dim=1)

        self.log(f'{stage}_loss', loss, on_step=stage == 'train', on_epoch=True,
                 prog_bar=True, batch_size=inputs.shape[0])
        getattr(self, f"{stage}_acc").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_iou").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_f1").update(preds.detach(), masks.detach())
        getattr(self, f"{stage}_auc").update(logits.detach(), masks.detach().long())
        getattr(self, f"{stage}_f1_per_class").update(preds.detach(), masks.detach())
        return loss

    def _epoch_end(self, stage, prog_bar=True):
        self.log_dict({
            f"{stage}_acc": getattr(self, f"{stage}_acc").compute(),
            f"{stage}_iou": getattr(self, f"{stage}_iou").compute(),
            f"{stage}_f1": getattr(self, f"{stage}_f1").compute(),
            f"{stage}_auc": getattr(self, f"{stage}_auc").compute(),
        }, prog_bar=prog_bar)

        for i, f1 in enumerate(getattr(self, f"{stage}_f1_per_class").compute()):
            self.log(f"{stage}_f1_class_{i}", f1, prog_bar=False)

        if stage == 'train' and self.elevation_fusion is not None:
            for k, v in self.elevation_fusion.get_fusion_scales().items():
                self.log(k, v, prog_bar=False)

        for metric in ("acc", "iou", "f1", "auc", "f1_per_class"):
            getattr(self, f"{stage}_{metric}").reset()

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, 'test')

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")


# =============================================================================
# Patch-wise inference for encoders with a fixed input size
# =============================================================================

def compute_patch_params(img_size, patch_size, overlap_ratio=0.1):
    """Stride and padding so that patches cover the image plus a border of overlap."""
    desired_overlap = 2 * round((patch_size * overlap_ratio) / 2)  # keep it even
    stride = patch_size - desired_overlap

    min_covered = img_size + desired_overlap
    num_patches = math.ceil((min_covered - patch_size) / stride) + 1
    total_covered = stride * (num_patches - 1) + patch_size
    pad = (total_covered - img_size) // 2

    return stride, pad


def patch_image_to_batch(image, patch_size, overlap_ratio=0.1):
    """(B, C, S, S) -> (B * num_patches, C, patch_size, patch_size)."""
    _, channels, img_size, _ = image.shape
    stride, pad = compute_patch_params(img_size, patch_size, overlap_ratio)

    image = F.pad(image, (pad, pad, pad, pad), mode="reflect")
    patches = F.unfold(image, kernel_size=patch_size, stride=stride)
    num_patches = patches.shape[-1]
    patches = patches.permute(0, 2, 1).reshape(-1, channels, patch_size, patch_size)

    return patches, num_patches, stride, pad


def merge_output_patches_to_image(output_patches, num_patches, patch_size, stride,
                                  batch_size, img_size, pad):
    """Inverse of :func:`patch_image_to_batch`: crop half the overlap off every
    patch (its least reliable part) and fold the rest back into one image."""
    crop = (patch_size - stride) // 2
    border = pad - crop
    output_patches = output_patches[:, :, crop:patch_size - crop, crop:patch_size - crop]
    output_patches = output_patches.reshape(batch_size, num_patches, -1).permute(0, 2, 1)

    outputs = F.fold(output_patches, output_size=(img_size + border * 2, img_size + border * 2),
                     kernel_size=(stride, stride), stride=(stride, stride))

    return outputs[:, :, border:border + img_size, border:border + img_size]
