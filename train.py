"""Train a segmentation model on the neophyte drone tiles.

    python train.py                                   # defaults from configs/train.yaml
    python train.py data=neophytes_split_cv1_train_1024 model=model_neophytes_mit_b2
    python train.py model.lr=0.0002 data.batch_size_train=4

Everything is configured through Hydra (configs/), so any value in the printed
config can be overridden on the command line with ``key=value``.

The run writes to ``lightning_logs/<exp_name>/``; evaluate it afterwards with
``python test.py exp_name=<exp_name>``.
"""

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # # load only the elevation rasters the model actually consumes
    if "dsm" in cfg.model.in_channels: cfg.data.load_dsm = True
    if "dtm" in cfg.model.in_channels: cfg.data.load_dtm = True
    if "ndsm" in cfg.model.in_channels: cfg.data.load_ndsm = True
    print(OmegaConf.to_yaml(cfg))

    # # Imports inside main: hydra resolves the config before torch/cuda start up,
    # # so a config typo fails in seconds instead of after the import overhead.
    import os
    import time

    import matplotlib
    import pytorch_lightning as pl
    import torch
    import torch.multiprocessing
    from pytorch_lightning import seed_everything
    from pytorch_lightning.loggers import CSVLogger, WandbLogger

    from datasets.neophyte_datamodule import NeophyteDataModule
    from models.semseg_plm import SegmentationModel
    from utils.callback_utils import get_callbacks
    from utils.transform_utils import get_transforms

    print('Imports completed.')

    # # avoids shared-memory allocation errors with many dataloader workers
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matplotlib.rcParams.update(cfg.matplotlib_style)

    if cfg.pl_seed:
        seed_everything(cfg.pl_seed, workers=True)

    # %% data
    train_transforms = get_transforms(cfg.data.train_transforms, cfg.data.transforms_additional_targets)
    val_transforms = get_transforms(cfg.data.val_transforms, cfg.data.transforms_additional_targets)
    test_transforms = get_transforms(cfg.data.test_transforms, cfg.data.transforms_additional_targets)
    dsm_transforms = get_transforms(cfg.data.dsm_transforms, toTensor=False)
    ndsm_transforms = get_transforms(cfg.data.ndsm_transforms, toTensor=False)

    dataModule = NeophyteDataModule(cfg.data.image_dir,
                                    cfg.data.classes,
                                    train_transforms,
                                    val_transforms,
                                    test_transforms,
                                    dsm_transform=dsm_transforms,
                                    ndsm_transform=ndsm_transforms,
                                    dataset_stats=cfg.data.dataset_stats,
                                    weighted_sampling=cfg.data.weighted_sampling,
                                    cap_quantile=cfg.data.cap_quantile,
                                    lift_quantile=cfg.data.lift_quantile,
                                    bg_fraction=cfg.data.bg_fraction,
                                    min_strat_images=cfg.data.min_strat_images,
                                    value_mapping=cfg.data.value_mapping,
                                    ignore_index=cfg.data.ignore_index,
                                    load_auxiliary=cfg.data.load_auxiliary,
                                    load_dsm=cfg.data.load_dsm,
                                    load_dtm=cfg.data.load_dtm,
                                    load_ndsm=cfg.data.load_ndsm,
                                    few_shot_k=cfg.data.few_shot_k,
                                    few_shot_mode=cfg.data.few_shot_mode,
                                    few_shot_bg_k=cfg.data.few_shot_bg_k,
                                    max_nodata_frac=cfg.data.max_nodata_frac,
                                    val_few_shot_k=cfg.data.val_few_shot_k,
                                    val_few_shot_mode=cfg.data.val_few_shot_mode,
                                    val_few_shot_bg_k=cfg.data.val_few_shot_bg_k,
                                    load_into_ram=cfg.data.load_into_ram,
                                    batch_size_train=cfg.data.batch_size_train,
                                    batch_size_val=cfg.data.batch_size_val,
                                    batch_size_test=cfg.data.batch_size_test,
                                    num_workers=cfg.data.num_workers)

    dataModule.prepare_data()
    dataModule.setup()
    print('# Train imgs: ', len(dataModule.train_dataset))
    print('# Val imgs:   ', len(dataModule.val_dataset))
    print('# Test imgs:  ', len(dataModule.test_dataset))

    # # sanity check: one batch through the pipeline before the model is built
    batch = next(iter(dataModule.train_dataloader()))
    print('Image batch shape: ', batch['image'].shape, '\nMask batch shape:  ', batch['mask'].shape)

    # %% model
    model = SegmentationModel(cfg.model.model_name,
                              cfg.model.encoder_name,
                              cfg.model.img_size,
                              cfg.model.num_classes,
                              cfg.model.lr,
                              in_channels=cfg.model.in_channels,
                              ignore_index=cfg.data.ignore_index,
                              optimizer=cfg.model.optimizer,
                              lr_scheduler=cfg.model.lr_scheduler,
                              loss=cfg.model.loss,
                              weight=cfg.model.weight,
                              patch_2_img_size=cfg.model.patch_2_img_size,
                              fusion_mode=cfg.model.fusion_mode)

    # %% training
    callbacks = get_callbacks(cfg.model.callbacks)
    if cfg.logger == 'wandb':
        logger = WandbLogger(name=cfg.exp_name, project=cfg.log_name)
    else:
        logger = CSVLogger(cfg.log_dir, name=cfg.exp_name)
    trainer = pl.Trainer(max_epochs=cfg.model.max_epochs,
                         precision=cfg.model.precision,
                         callbacks=callbacks,
                         logger=logger,
                         deterministic=cfg.deterministic_train,
                         check_val_every_n_epoch=cfg.model.check_val_every_n_epoch)

    # # resume an interrupted run: same exp_name -> same folder -> picks up last.ckpt
    last_ckpt_path = os.path.join(cfg.log_path, 'checkpoints/last.ckpt')
    if os.path.exists(last_ckpt_path):
        print(f'Loading weights from: {last_ckpt_path}')
        model = SegmentationModel.load_from_checkpoint(last_ckpt_path, weights_only=False).to(device)

    start_time = time.time()
    trainer.fit(model, dataModule)
    print('Training finished. Elapsed Time:', round((time.time() - start_time) / 60, 2), 'min')

    # %% quick test-set pass with the best checkpoint
    # # (the detailed evaluation lives in test.py)
    ckpt_path = trainer.checkpoint_callback.best_model_path
    print('ckpt_path: ', ckpt_path)
    model = SegmentationModel.load_from_checkpoint(ckpt_path, weights_only=False).to(device)
    model.eval()
    trainer.test(model=model, dataloaders=dataModule)

    if cfg.logger == 'wandb':
        import wandb
        wandb.finish()

    hydra.core.global_hydra.GlobalHydra.instance().clear()


if __name__ == "__main__":
    main()
