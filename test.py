"""Evaluate a trained checkpoint on its test split.

    python test.py exp_name=<run directory in lightning_logs/>
    python test.py exp_name=<run> ckpt_type=last
    python test.py exp_name=<run> eval_site=True eval_month=True

The matching test data config is derived from the ``data_name`` embedded in
``exp_name`` (a cv1 run is tested on ``neophytes_split_cv1_test``); pass
``data=<config>`` explicitly to override that.

Results land in ``lightning_logs/<exp_name>/test_<timestamp>_<data>_<ckpt>/``:
    quantitative/  metrics as JSON + score plots + confusion matrices
    qualitative/   prediction figures for a few picked test tiles
"""

import hydra
from omegaconf import DictConfig, OmegaConf


def _auto_data_config(exp_name: str) -> str | None:
    """Infer the test data config from the data_name embedded in exp_name."""
    import os
    import re

    # exp_name format: YYYYMMDD_HHMMSS.ffffff_<data_name>_<model_...>
    m = re.match(r'^\d{8}_[\d.]+_(.+)$', exp_name)
    if not m:
        return None
    rest = m.group(1)

    # more specific patterns first: a half-resolution run (_ds2) must not fall back
    # to the full-resolution test config, or it would be evaluated at the wrong scale
    patterns = [
        (r'^np_cv(\d+)_ds2', lambda m: f'neophytes_split_cv{m.group(1)}_test_ds2'),
        (r'^np_cv(\d+)', lambda m: f'neophytes_split_cv{m.group(1)}_test'),
        (r'^np_loc', lambda _: 'neophytes_split_local_test'),
        (r'^np_debug', lambda _: 'neophytes_split_debug_test'),
    ]
    for pattern, make_cfg in patterns:
        pm = re.match(pattern, rest)
        if pm:
            test_cfg = make_cfg(pm)
            return test_cfg if os.path.isfile(os.path.join("configs", "data", f"{test_cfg}.yaml")) else None
    return None


def _require_exp_name(argv, script):
    """Fail early with a readable message instead of a Hydra interpolation error."""
    if not any(a.startswith("exp_name=") for a in argv):
        import sys
        sys.exit(f"Missing exp_name. Usage: python {script} exp_name=<run directory in "
                 f"lightning_logs/>\nAvailable runs: ls lightning_logs/")


@hydra.main(version_base=None, config_path="configs", config_name="test")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    import os

    import matplotlib
    import numpy as np
    import torch
    import torch.multiprocessing
    from pytorch_lightning import seed_everything

    from datasets.neophyte_datamodule import NeophyteDataModule
    from models.semseg_plm import SegmentationModel
    from utils.eval_utils import (evaluate_model, save_model_predictions, visualize_f1_per_month,
                                  visualize_save_confusions, visualize_scores_per_auxiliary,
                                  visualize_scores_per_class, visualize_scores_per_class_cv)
    from utils.transform_utils import get_transforms
    from utils.utils import getListOfFiles, make_dir, natural_keys, shrink_dict

    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.set_float32_matmul_precision('medium')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matplotlib.rcParams.update(cfg.matplotlib_style)

    if cfg.pl_seed:
        seed_everything(cfg.pl_seed, workers=True)

    # %% data -- no augmentation, no weighted sampling, no few-shot filtering
    test_transforms = get_transforms(cfg.data.test_transforms, cfg.data.transforms_additional_targets)
    dsm_transforms = get_transforms(cfg.data.dsm_transforms, toTensor=False)
    ndsm_transforms = get_transforms(cfg.data.ndsm_transforms, toTensor=False)

    dataModule = NeophyteDataModule(cfg.data.image_dir, cfg.data.classes,
                                    test_transforms, test_transforms, test_transforms,
                                    dsm_transform=dsm_transforms, ndsm_transform=ndsm_transforms,
                                    dataset_stats=cfg.data.dataset_stats,
                                    value_mapping=cfg.data.value_mapping,
                                    ignore_index=cfg.data.ignore_index,
                                    load_auxiliary=cfg.data.load_auxiliary,
                                    load_dsm=cfg.data.load_dsm,
                                    load_dtm=cfg.data.load_dtm,
                                    load_ndsm=cfg.data.load_ndsm,
                                    batch_size_train=cfg.data.batch_size_test,
                                    batch_size_val=cfg.data.batch_size_test,
                                    batch_size_test=cfg.data.batch_size_test,
                                    num_workers=cfg.data.num_workers)
    dataModule.prepare_data()
    dataModule.setup()

    batch = next(iter(dataModule.test_dataloader()))
    print('Image batch shape: ', batch['image'].shape, '\nMask batch shape:  ', batch['mask'].shape)

    # %% checkpoint
    ckpts_paths = sorted(getListOfFiles(cfg.ckpt_path), key=natural_keys)
    matched = [s for s in ckpts_paths if cfg.ckpt_type in s]
    if not matched:
        matched = [s for s in ckpts_paths if 'last' in s]
        if not matched:
            raise FileNotFoundError(f"No checkpoint matching '{cfg.ckpt_type}' (or 'last') in {cfg.ckpt_path}")
        print(f"[test.py] WARNING: no '{cfg.ckpt_type}' checkpoint found, falling back to 'last'")
    ckpt_path = matched[-1]
    print('ckpt_path: ', ckpt_path)

    # # keep the epoch number in the output folder name (best_epoch=42 rather than best)
    cfg.ckpt_type = os.path.splitext(os.path.basename(ckpt_path))[0]
    OmegaConf.resolve(cfg)  # test_path interpolates ckpt_type, so resolve again

    model = SegmentationModel.load_from_checkpoint(ckpt_path, weights_only=False).to(device)
    # # evaluate on whole tiles: patching was a training-time constraint of the encoder
    model.set_patch_2_img_size(False)
    print('Model loaded successfully.')

    # # class names/colors in class-index order, plus the display order for the plots
    class_names = [cfg.data.classes[i].name for i in sorted(cfg.data.classes.keys(), key=int)]
    class_colors = [cfg.data.classes[i].color for i in sorted(cfg.data.classes.keys(), key=int)]
    display_order = [0] + list(cfg.data.display_order)
    class_names_disp = [class_names[i] for i in display_order]

    def reorder_scores(scores):
        return {k: np.array([v[i] for i in display_order]) for k, v in scores.items()}

    # %% quantitative evaluation over the full test set
    if not cfg.qualitative_only:
        print('Start quantitative evaluation of the full test dataset.')
        metrics_dir = make_dir(cfg.test_path, 'quantitative')
        cv_folds = OmegaConf.to_container(cfg.data.cv_folds, resolve=True) if 'cv_folds' in cfg.data else None
        # canopy/phenology only exist in the batch when the data config loads them
        eval_auxiliary = cfg.eval_auxiliary and cfg.data.load_auxiliary

        metrics = evaluate_model(model, dataModule.test_dataloader(), cfg.model.num_classes, device,
                                 ignore_index=cfg.data.ignore_index,
                                 eval_auxiliary=eval_auxiliary, eval_site=cfg.eval_site,
                                 eval_month=cfg.eval_month, eval_cv=cfg.eval_cv, cv_folds=cv_folds,
                                 compute_auc=cfg.compute_auc, auc_thresholds=cfg.auc_thresholds,
                                 save_dir=metrics_dir, class_names=class_names,
                                 display_order=display_order)

        score_keys = ['F1', 'IoU', 'Precision', 'Recall']
        visualize_scores_per_class(reorder_scores(shrink_dict(metrics['semseg'], score_keys)),
                                   os.path.join(metrics_dir, 'semseg_scores.png'), class_names_disp)
        visualize_save_confusions(metrics['conf'][np.ix_(display_order, display_order)],
                                  metrics_dir, display_labels=class_names_disp)

        print(f"Class-wise F1: {np.array2string(metrics['semseg']['F1'], formatter={'float_kind': lambda x: f'{x:.3f}'})}")
        print(f"mF1 (wo bg): {metrics['semseg']['F1-avg-wo0']:.4f}  |  "
              f"mIoU (wo bg): {metrics['semseg']['IoU-avg-wo0']:.4f}")
        if 'AUC' in metrics['semseg']:
            print(f"mAUC (wo bg): {metrics['semseg']['AUC-avg-wo0']:.4f}  |  "
                  f"binary AUC: {metrics['semseg_binary']['AUC']:.4f}")

        # # the headline metric: mean ± std over the CV folds present in this test split
        if cfg.eval_cv and 'mean' in metrics.get('semseg_cv', {}):
            cv_mean, cv_std = metrics['semseg_cv']['mean'], metrics['semseg_cv']['std']
            print(f"Per-CV mF1 (neophytes): {cv_mean['F1-avg-wo0']:.4f} ± {cv_std['F1-avg-wo0']:.4f}  |  "
                  f"mIoU: {cv_mean['IoU-avg-wo0']:.4f} ± {cv_std['IoU-avg-wo0']:.4f}  |  "
                  f"mRecall: {cv_mean['Recall-avg-wo0']:.4f} ± {cv_std['Recall-avg-wo0']:.4f}")
            visualize_scores_per_class_cv(reorder_scores(shrink_dict(cv_mean, score_keys)),
                                          reorder_scores(shrink_dict(cv_std, score_keys)),
                                          os.path.join(metrics_dir, 'semseg_scores_cv.png'),
                                          class_names_disp)

        if cfg.eval_month and metrics['semseg_month']:
            visualize_f1_per_month(metrics['semseg_month'], os.path.join(metrics_dir, 'f1_per_month.png'),
                                   class_names, class_colors)

        if eval_auxiliary:
            visualize_scores_per_auxiliary(metrics['phenology'],
                                           os.path.join(metrics_dir, 'phenology_scores.png'),
                                           class_names, cfg.data.phenology_names,
                                           x_label="phenological phase",
                                           class_order=list(cfg.data.display_order))
            visualize_scores_per_auxiliary(metrics['canopy'],
                                           os.path.join(metrics_dir, 'canopy_scores.png'),
                                           class_names, cfg.data.canopy_names,
                                           x_label=r"canopy density [\%]",
                                           class_order=list(cfg.data.display_order))

    # %% qualitative results for a few picked test tiles
    print('Start qualitative evaluation of some picked samples from the test set.')
    qual_dir = make_dir(cfg.test_path, 'qualitative')
    ndsm_norm = next((t.params for t in cfg.data.ndsm_transforms if t.type == 'Normalize'), None)
    save_model_predictions(model, dataloader=dataModule.pick_dataloader(), device=device,
                           save_dir=qual_dir, colors=class_colors, class_names=class_names,
                           ignore_index=cfg.data.ignore_index,
                           ndsm_mean=float(ndsm_norm.mean) if ndsm_norm else 0.0,
                           ndsm_std=float(ndsm_norm.std) if ndsm_norm else 1.0,
                           entropy_gamma=cfg.entropy_gamma)

    hydra.core.global_hydra.GlobalHydra.instance().clear()


if __name__ == "__main__":
    import sys

    _require_exp_name(sys.argv[1:], "test.py")

    # # pick the test data config matching exp_name, unless the user named one
    if not any(a.startswith("data=") for a in sys.argv[1:]):
        exp_arg = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("exp_name=")), None)
        auto_cfg = _auto_data_config(exp_arg) if exp_arg else None
        if auto_cfg:
            print(f"[test.py] Auto-selected data config: {auto_cfg}")
            # insert before any flag so hydra reads it as an override, not a stray argument
            first_flag = next((i for i, a in enumerate(sys.argv[1:], 1) if a.startswith("-")), None)
            if first_flag is None:
                sys.argv.append(f"data={auto_cfg}")
            else:
                sys.argv.insert(first_flag, f"data={auto_cfg}")
    main()
