"""Cascade Mask R-CNN on UBC roof_fine task (11 fine-grained roof classes).

Target: replicate Huang et al. Table 6 result — AP ~15.3, AP50 ~25.2.
"""
_base_ = './cascade-mask-rcnn_r50_fpn_ubc_base.py'

# -----------------------------------------------------------------------------
# Task-specific dataset
# -----------------------------------------------------------------------------
dataset_type = 'UBCRoofFineDataset'
train_ann = 'annotations/roof_fine_train.json'
val_ann = 'annotations/roof_fine_val.json'
num_classes = 11

# Wire dataset_type + ann files into the dataloaders the base config declared
# with placeholders.
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

# -----------------------------------------------------------------------------
# Model — override num_classes in each of Cascade's 3 bbox heads and its mask head
# -----------------------------------------------------------------------------
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ],
        mask_head=dict(num_classes=num_classes)))
