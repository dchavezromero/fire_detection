"""
Base config for Cascade Mask R-CNN on UBC v1 (Urban Building Classification).

Replicates the training recipe of Huang et al. (2022) CVPRW:
  - Cascade Mask R-CNN, ResNet-50-FPN backbone, COCO-pretrained
  - 100 epochs on 560 training tiles (Beijing + Munich, 600x600 each)
  - Learning rate decay at epochs 60 and 90 (factor 0.1)
  - SGD with warmup
  - Custom COCO area thresholds: small <400, medium 400-1600, large >1600 pixels

Task-specific configs (roof_fine, roof_coarse, use_coarse) inherit from this
and override only `num_classes`, the dataset type, and the annotation filenames.
"""

_base_ = [
    '../_base_/models/cascade-mask-rcnn_r50_fpn.py',
    '../_base_/default_runtime.py',
]

# -----------------------------------------------------------------------------
# Training regime — flip MAX_EPOCHS to 10 for sanity run, 100 for full paper run
# -----------------------------------------------------------------------------
MAX_EPOCHS = 100
VAL_INTERVAL = 5
# LR decay schedule scales with MAX_EPOCHS (60% and 90% of training)
LR_STEP_EPOCHS = [int(MAX_EPOCHS * 0.6), int(MAX_EPOCHS * 0.9)]

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
# `data_root` is the only path that should ever need changing per-machine.
data_root = '/home/dennis/UBC_v1.0/'
# `dataset_type` and `ann_file` names are overridden in the task-specific
# configs that inherit from this base.
dataset_type = None      # e.g. 'UBCRoofFineDataset'
train_ann = None         # e.g. 'annotations/roof_fine_train.json'
val_ann = None           # e.g. 'annotations/roof_fine_val.json'

backend_args = None

# -----------------------------------------------------------------------------
# Pipelines
# -----------------------------------------------------------------------------
# Image size: paper uses 600x600 tiles natively. We keep them at native size
# with light multi-scale jitter. No hard resize that would bury small buildings.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        #scale=[(480, 480), (720, 720)],   # +/- 20% around native 600
        scale=[(400, 400), (900, 900)],   # wider multi-scale, ~0.67x-1.5x native 600
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(600, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape',
                   'img_shape', 'scale_factor')),
]

# -----------------------------------------------------------------------------
# Dataloaders
# -----------------------------------------------------------------------------
# Batch size 4 per GPU on RTX 5080 (16 GB) — Cascade Mask R-CNN at 600x600
# fits comfortably. Increase to 6 or 8 if VRAM allows after first run.
train_dataloader = dict(
    batch_size=6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=train_ann,
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=val_ann,
        data_prefix=dict(img='val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

test_dataloader = val_dataloader

# -----------------------------------------------------------------------------
# Evaluator — custom area thresholds per Huang et al. §4.2
# -----------------------------------------------------------------------------
# Paper redefines small/medium/large as area <400 / 400-1600 / >1600 pixels
# (vs COCO default 32^2=1024 / 96^2=9216). Urban tiles have many small buildings.
val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=data_root + '{ANN_PLACEHOLDER}',   # filled in task configs
        metric=['bbox', 'segm'],
        classwise=True,    # per-class AP table in logs (needed for Table 6 replication)
        format_only=False,
        backend_args=backend_args),
]
test_evaluator = val_evaluator

# -----------------------------------------------------------------------------
# Training schedule
# -----------------------------------------------------------------------------
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=MAX_EPOCHS,
    val_interval=VAL_INTERVAL)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# SGD with warmup, LR step decay. LR scaled for batch 4 (1 GPU) from the
# MMDetection default of 0.02 at batch 16 (8 GPUs x 2). 0.02 * 4/16 = 0.005.
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=1000),                         # 1000-iter linear warmup
    dict(
        type='MultiStepLR',
        begin=0,
        end=MAX_EPOCHS,
        by_epoch=True,
        milestones=LR_STEP_EPOCHS,
        gamma=0.1),
]

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

# Linear LR scaling safety net — if you change batch_size later, this
# auto-rescales LR. `base_batch_size=16` encodes the LR=0.02 reference point.
auto_scale_lr = dict(enable=True, base_batch_size=16)

# -----------------------------------------------------------------------------
# Hooks
# -----------------------------------------------------------------------------
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/segm_mAP',        # keep best-by-mask-mAP checkpoint
        max_keep_ckpts=3),                # cap disk usage: keep 3 most recent
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

# -----------------------------------------------------------------------------
# Pretrained weights
# -----------------------------------------------------------------------------
# Start from Cascade Mask R-CNN R-50-FPN COCO-pretrained weights (already
# downloaded during Phase 0 smoke test). Classification layers for box/mask
# heads get reinitialized automatically because num_classes differs from COCO's 80.
load_from = 'checkpoints/cascade_mask_rcnn_r50_fpn_1x_coco.pth'

# Reproducibility
randomness = dict(seed=42, deterministic=False)

# -----------------------------------------------------------------------------
# Visualization backends (TensorBoard)
# -----------------------------------------------------------------------------
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
