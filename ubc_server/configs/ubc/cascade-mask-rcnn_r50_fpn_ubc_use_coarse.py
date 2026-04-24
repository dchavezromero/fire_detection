"""Cascade Mask R-CNN on UBC use_coarse task (5 coarse building-function classes).

Note: Huang et al. report this as the hardest task (AP ~14.4 for Mask R-CNN),
because building function is a latent attribute not fully visible from overhead imagery.
"""
_base_ = './cascade-mask-rcnn_r50_fpn_ubc_base.py'

dataset_type = 'UBCUseCoarseDataset'
train_ann = 'annotations/use_coarse_train.json'
val_ann = 'annotations/use_coarse_val.json'
num_classes = 5

train_dataloader = dict(
    dataset=dict(type=dataset_type, ann_file=train_ann))
val_dataloader = dict(
    dataset=dict(type=dataset_type, ann_file=val_ann))
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type='CocoMetric',
        ann_file=_base_.data_root + val_ann,
        metric=['bbox', 'segm'],
        classwise=True,
        format_only=False)
]
test_evaluator = val_evaluator

model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead', in_channels=256, fc_out_channels=1024,
                roi_feat_size=7, num_classes=num_classes,
                bbox_coder=dict(type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ],
        mask_head=dict(num_classes=num_classes)))
