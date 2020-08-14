# LIDC Segmentation of Lung Nodule with U-Net, U-Net++
This repository is the second stage for Lung Cancer project. Please check out my first repository ![LIDC-IDRI-Preprocessing](https://github.com/jaeho3690/LIDC-IDRI-Preprocessing)
This repository would train a segmentation model(U-Net, U-Net++) for Lung Nodules. The whole script is implemented in Pytorch Framework.
The model script for U-Net++ is sourced from ![here](https://github.com/4uiiurz1/pytorch-nested-unet)

```
+-- Unet
|    # This folder contains the model code for U-Net
+-- UnetNested
|    # This folder contains the model code for U-Net++
+-- figures
|    # This folder saves figure images
+-- meta_csv
|    # This folder contains information of each images in a csv format. 
+-- notebook
|    # This folder contains jupyter notebook files for some visuialization
+-- dataset.py
|    # Dataset class for Pytorch, Accepts .npy file format
+-- losses.py
|    # Loss function. Here I use the BCE dice loss. Sourced from 
+-- metrics.py
     # Metric function. It is interesting to note that somehow the dice coefficient doesn't increase as fast as IOU in the early stages of training.
+-- train.py
|    # Training of Segmentation model.
+-- utils.py
|    # Utility file
+-- validate.py
|    # For validation of the model

```

## 1.Check out my LIDC-IDRI-Preprocessing repository
This repository goes through the preprocessing steps of the LIDC-IDRI data. Running the script will return .npy images for each lung cancer slice and mask slice. Also, a meta.csv, clean_meta.csv file will created after running the jupyter file. 
## 2. 
## 5. Contributing and Acknowledgement
I started this Lung cancer detection project a year ago. I was really a newbie to python. I didn't even understand what a directory setting is at the time! However, I had to complete this project
for some personal reasons. I looked through google and other githubs. But most of them were too hard to understand and the code itself lacked information. I hope my codes here could help
other researchers first starting to do lung cancer detection projects. Please give a star if you found this repository useful.

here is the link of github where I learned a lot from. Some of the codes are sourced from below.
1. https://github.com/mikejhuang/LungNoduleDetectionClassification





Evaluatuion Metric
https://www.sciencedirect.com/science/article/pii/S1361841510000587?via%3Dihub
