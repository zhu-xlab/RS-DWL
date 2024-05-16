# DWL
This is official Pytorch implementation of "Decouple and weight semi-supervised semantic segmentation of remote sensing images, " ISPRS, 2024."


## Datasets
First, please organize your datasets according to the following structure with DeepGlobe_Landcover as an example:
```plaintext
./datasets/DeepGlobe_Landcover/
│
├── images/                  
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
│   │
├── labels/          
│   ├── label_001.png
│   ├── label_002.png
│   └── ...
│
└── lists
│   ├── train_1%_labeled.txt
│   ├── train_1%_unlabeled.txt
│   ├── train_5%_labeled.txt
│   ├── train_5%_unlabeled.txt
│   ├── train_10%_labeled.txt
│   ├── train_10%_unlabeled.txt
│   ├── val.txt
│   ├── test.txt
```
Here, in  'images/' and 'labels/', all the images and labels should be cropped to 512x512. In 'lists', 'train_percent%_labeled.txt' and 'train_percent%_unlabeled.txt' contain the paths of labeled and unlabeled image-label pairs.


# Pretrained Weights
The used SegFormer's codes and pretrained weights are from https://github.com/bubbliiiing/segformer-pytorch. Please download the weights to the directory "./checkpoints/".

## Training
You can train a model (with DeepGlobe_Landcover as an example) as: 
```plaintext
python3 train.py --gpu 0 --model SegFormer --method DWL --percent 1.0 --dataset DeepGlobe_Landcover --num-classes 7 --ignore-index 6 --num-epochs 20
```


## Test
After training, the model can be tested as:
```plaintext
python3 test.py --gpu 0 --model SegFormer --method DWL --percent 1.0 --dataset DeepGlobe_Landcover --num-classes 7 --ignore-index 6
```
