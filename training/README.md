\# Training



Training artifacts and documentation for the UBC Cascade Mask R-CNN models

used by the fire detection pipeline's building classification stage.



\## Shared assets



The configs and dataset class files used for training are the same ones the

inference server loads at runtime, so they live in `ubc\_server/` rather than

being duplicated here:



\- \*\*Configs:\*\* \[`../ubc\_server/configs/ubc/`](../ubc\_server/configs/ubc/)

&#x20; - `cascade-mask-rcnn\_r50\_fpn\_ubc\_base.py` — shared base config

&#x20; - `cascade-mask-rcnn\_r50\_fpn\_ubc\_roof\_coarse.py`

&#x20; - `cascade-mask-rcnn\_r50\_fpn\_ubc\_roof\_fine.py`

&#x20; - `cascade-mask-rcnn\_r50\_fpn\_ubc\_use\_coarse.py`

\- \*\*Base dependencies:\*\* \[`../ubc\_server/configs/\_base\_/`](../ubc\_server/configs/\_base\_/)

&#x20; (MMDetection standard base configs inherited by the UBC configs)

\- \*\*Dataset class:\*\* \[`../ubc\_server/mmdet\_plugins/ubc.py`](../ubc\_server/mmdet\_plugins/ubc.py)

&#x20; (registers three UBC task datasets with MMDetection)



\## Reproducing training



See the "Training" section of the main \[README](../README.md) for full setup

and run instructions.



\## Training dataset



UBC v1 satellite imagery lives at `../training\_datasets/UBC\_v1.0/`. See the

main README's "Dataset" section for the expected layout.



\## Trained weights



Best checkpoints from our training runs land in the server's checkpoints

directory (`../ubc\_server/checkpoints/`) after running:



```bash

cp \~/mmdetection/work\_dirs/cascade-mask-rcnn\_r50\_fpn\_ubc\_<task>/best\_coco\_segm\_mAP\_epoch\_\*.pth \\

&#x20;  ../ubc\_server/checkpoints/<task>.pth

```



Checkpoints are gitignored due to size (\~310 MB each, \~1 GB total).

