# LIDC Segmentation of Lung Nodule with U-Net, U-Net++
This repository is the second stage for Lung Cancer project. Please check out my first repository ![LIDC-IDRI-Preprocessing](https://github.com/jaeho3690/LIDC-IDRI-Preprocessing)
Explanation for my first repository is on ![Medium](https://medium.com/@jaeho3690/how-to-start-your-very-first-lung-cancer-detection-project-using-python-part-1-3ab490964aae) as well!
The input for this repository requires the output format from the first stage. 
This repository would train a segmentation model(U-Net, U-Net++) for Lung Nodules. The whole script is implemented in Pytorch Framework.
The model script for U-Net++ and some of the script format is sourced from ![here](https://github.com/4uiiurz1/pytorch-nested-unet)

# Requirements
* pytorch 1.4 
* GPU is needed
## 1. Check out my LIDC-IDRI-Preprocessing repository
This repository goes through the preprocessing steps of the LIDC-IDRI data. Running the script will return .npy images for each lung cancer slice and mask slice. Also, a meta.csv, clean_meta.csv file will be made after running the jupyter file. 
## 2. Fix directory settings
All the scripts were written when I was not so familiar with directory settings. I mostly used absolute directory. Please change each directory setting to fit yours. I apologize for the inconvenience.

# Installation
1. Create a virtual environment
```
conda create -n=<env_name> python=3.6 
conda activate <env_name>
```
2. Install pip packages
```
pip install -r requirements.txt
```

# File Structure
```
+-- Unet
|    # This folder contains the model code for U-Net
+-- UnetNested
|    # This folder contains the model code for U-Net++
+-- figures
|    # This folder saves figure images
+-- meta_csv
|    # This folder contains information of each images in a csv format. 
|    # The cs
+-- notebook
|    # This folder contains jupyter notebook files for some visuialization
+-- dataset.py
|    # Dataset class for Pytorch, Accepts .npy file format
+-- losses.py
|    # Loss function. Here I use the BCE dice loss. Sourced from 
+-- metrics.py
     # Metric function. It is interesting to note that somehow the dice coefficient doesn't increase as fast as IOU in the early stages of training.
+-- train.py
|    # Training of Segmentation model. Adjust hyperparameters
+-- utils.py
|    # Utility file
+-- validate.py
|    # For validation of the model

```

# Training
1. Train the model. There will be total of 4 cases. UNET_base, UNET_with_augmentation, NestedUNET_base, NestedUNET_with_augmentation
```python
# Training U-net
python train.py --name UNET --augmentation True
# Training U-Net++
python train.py --name NestedUNET --augmentation True 
```

2. Validate the model
```python 
# if you want to get the augmented version
python validate.py --name UNET --augmentaton True
```

# Some resources that were useful

Evaluatuion Metric
https://www.sciencedirect.com/science/article/pii/S1361841510000587?via%3Dihub
