import pandas as pd
import argparse
import os
from glob import glob
from collections import OrderedDict
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import MyLidcDataset
from metrics import iou_score,dice_coef
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='folder name of the model')

    args = parser.parse_args()

    return args

def save_output(output,output_directory,test_dataset,counter):
    for i in range(output.shape[0]):
        label = test_dataset[counter]
        print(label)
        if len(label)==61:
            label = label[-25:-4]
        elif len(label)==59:
            label = 
        else:
            label = label[-26:-4]
        np.save(output_directory+label+'_predict.npy',output[i,:,:])
        print(output_directory+label+'_predict.npy')
        counter+=1


    return counter



def main():
    args = parse_args()

    with open('model_outputs/{}/config.yml'.format(args.name), 'r') as f:
        config = yaml.load(f)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model {}".format(args.name))
    if config['name']=='NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load('model_outputs/{}/model.pth'.format(args.name)))
    model = model.cuda()

    # Data loading code
    IMAGE_DIR = '/home/LUNG_DATA/Image'
    MASK_DIR = '/home/LUNG_DATA/Mask'
    CLEAN_DIR_IMG ='/home/LUNG_DATA/Clean/Image'
    CLEAN_DIR_MASK ='/home/LUNG_DATA/Clean/Mask'
    # Directory to save U-Net predict output
    OUTPUT_MASK_DIR = '/home/LUNG_DATA/Unet_output_data/{}'.format(args.name)
    os.makedirs(OUTPUT_MASK_DIR,exist_ok=True)


    test_size = 0.2

    # Get all *npy images into list
    folder_images = list()
    folder_masks = list()
    folder_clean_images = list()
    folder_clean_masks = list()
    for (dirpath, _ , filenames) in os.walk(IMAGE_DIR):
        folder_images += [os.path.join(dirpath, file) for file in filenames]
    for (dirpath, _ , filenames) in os.walk(MASK_DIR):
        folder_masks += [os.path.join(dirpath, file) for file in filenames]
    for (dirpath, _ , filenames) in os.walk(CLEAN_DIR_IMG):
        folder_clean_images += [os.path.join(dirpath, file) for file in filenames]
    for (dirpath, _ , filenames) in os.walk(CLEAN_DIR_MASK):
        folder_clean_masks += [os.path.join(dirpath, file) for file in filenames]

    folder_images.sort()
    folder_masks.sort()
    folder_clean_images.sort()
    # You don't need to sort clean masks because its all empty masks

    _,test_image_paths,_,test_mask_paths = train_test_split(folder_images,folder_masks,test_size=test_size,random_state=1)
    print(len(folder_images))
    clean_proportion = int(len(test_image_paths)*0.2)
    test_image_paths.extend(folder_clean_images[:clean_proportion])
    test_mask_paths.extend(folder_clean_masks[:clean_proportion])
    print("Total length of test images are {}, Among them clean image test sets are {}".format(len(test_image_paths),clean_proportion))
    
    test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=2)
    model.eval()
    print(test_image_paths[:10])


    avg_meters = {'iou': AverageMeter(),
                  'dice': AverageMeter()}

    with torch.no_grad():

        counter = 0
        pbar = tqdm(total=len(test_loader))
        for input, target in test_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])

            output = torch.sigmoid(output).cpu().numpy()
            output = np.squeeze(output,axis=1)
            print(output.shape)

            counter = save_output(output,OUTPUT_MASK_DIR,test_image_paths,counter)
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()



    print('IoU: {:.4f}'.format(avg_meters['iou'].avg))
    print('DICE:{:.4f}'.format(avg_meters['dice'].avg))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
