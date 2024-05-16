# DWL
This is official Pytorch implementation of "Decouple and weight semi-supervised semantic segmentation of remote sensing images, " ISPRS, 2024."


python3 train.py --gpu 0 --model SegFormer --method DWL --percent 1.0 --dataset DeepGlobe_Landcover --num-classes 7 --ignore-index 6 --num-epochs 20
