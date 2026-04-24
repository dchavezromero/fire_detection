"""
UBC (Urban Building Classification) v1 dataset classes for MMDetection 3.x.

Three tasks on the same underlying COCO-format annotations:
  * UBCRoofFineDataset   — 11 fine-grained roof classes (Table 6 of Huang et al.)
  * UBCRoofCoarseDataset — 5 coarse roof classes
  * UBCUseCoarseDataset  — 5 coarse building-function classes

Each class declares METAINFO so MMDetection's CocoMetric produces readable
per-class AP tables in the training log.
"""
from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class UBCRoofFineDataset(CocoDataset):
    """UBC fine-grained roof-type classification (11 classes).

    Classes match roof_fine_train.json category ids 1..11:
      flat, flat_roof_complex, shed_roof, gable_roof, gambrel_roof,
      hipped_roof_v1, hipped_roof_v2, mansard_roof, pinnacle_roof,
      arched, other.
    """
    METAINFO = {
        'classes': (
            'flat', 'flat_roof_complex', 'shed_roof', 'gable_roof',
            'gambrel_roof', 'hipped_roof_v1', 'hipped_roof_v2',
            'mansard_roof', 'pinnacle_roof', 'arched', 'other',
        ),
        'palette': [
            (220, 20, 60),   (119, 11, 32),  (0, 0, 142),   (0, 0, 230),
            (106, 0, 228),   (0, 60, 100),   (0, 80, 100),  (0, 0, 70),
            (0, 0, 192),     (250, 170, 30), (100, 170, 30),
        ],
    }


@DATASETS.register_module()
class UBCRoofCoarseDataset(CocoDataset):
    """UBC coarse roof-type classification (5 classes).

    Classes match roof_coarse_train.json: flat, gable, hipped, arched, other.
    """
    METAINFO = {
        'classes': ('flat', 'gable', 'hipped', 'arched', 'other'),
        'palette': [
            (220, 20, 60), (0, 0, 142), (106, 0, 228),
            (250, 170, 30), (100, 170, 30),
        ],
    }


@DATASETS.register_module()
class UBCUseCoarseDataset(CocoDataset):
    """UBC coarse building-function classification (5 classes).

    Classes match use_coarse_train.json:
      residential, commercial, industrial, public, other.
    """
    METAINFO = {
        'classes': ('residential', 'commercial', 'industrial', 'public', 'other'),
        'palette': [
            (220, 20, 60), (0, 0, 142), (106, 0, 228),
            (250, 170, 30), (100, 170, 30),
        ],
    }
