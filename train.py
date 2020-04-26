import pandas as pd
import argparse
import os
from collections import OrderedDict
from glob import glob
import yaml

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import albumentations as albu
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import losses
from dataset import MyLidcDataset
from metrics import iou_score,dice_coef
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet

def parse_args():
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--name', default="UNET",
                        help='model name: UNET',choices=['UNET', 'NestedUNET'])
    parser.add_argument('--extra','--e', default='base',help="'epoch'_'with_augment'")
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=6, type=int,
                        metavar='N', help='mini-batch size (default: 6)')
    parser.add_argument('--early_stopping', default=30, type=int,
                        metavar='N', help='early stopping (default: 30)')
    parser.add_argument('--num_workers', default=4, type=int)

    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-5, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # data
    parser.add_argument('--augmentation',default=False)



    config = parser.parse_args()

    return config


def train(train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input = input.cuda()
        target = target.cuda()

        output = model(input)
        loss = criterion(output, target)
        iou = iou_score(output, target)
        dice = dice_coef(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice'].update(dice, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('dice',avg_meters['dice'].avg)
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def validate(val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter(),
                  'dice': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))

            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice',avg_meters['dice'].avg)
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice',avg_meters['dice'].avg)])

def main():
    # Get configuration
    config = vars(parse_args())
    # Make Model output directory
    file_name = config['name']+'_'+config['extra']
    os.makedirs('model_outputs/{}'.format(file_name),exist_ok=True)
    print("Creating directory called",file_name)

    print('-' * 20)
    print("Configuration Setting as follow")
    for key in config:
        print('{}: {}'.format(key, config[key]))
    print('-' * 20)

    #save configuration
    with open('model_outputs/{}/config.yml'.format(file_name), 'w') as f:
        yaml.dump(config, f)

    criterion = nn.BCEWithLogitsLoss().cuda()
    cudnn.benchmark = True

    # create model
    print("=> creating model" )
    if config['name']=='NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    params = filter(lambda p: p.requires_grad, model.parameters())


    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError


    #Hyper Parameter setting for Unet Training
    IMAGE_DIR = '/home/LUNG_DATA/Image_1/'
    MASK_DIR = '/home/LUNG_DATA/Mask_1/'
    validation_proportion = 0.25

    #Meta Information
    meta = pd.read_csv('/home/LUNG_DATA/meta_csv/meta.csv')

    # Get train/test label from meta.csv
    meta['original_image']= meta['original_image'].apply(lambda x:IMAGE_DIR+ x +'.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x:MASK_DIR+ x +'.npy')
  
    train_meta = meta[meta['Segmentation_train']==True]
    test_meta = meta[meta['Segmentation_train']==False]

    # Get all *npy images into list
    train_image_paths = list(train_meta['original_image'])
    train_mask_paths = list(train_meta['mask_image'])

    print("*"*50)
    print("The lenght of image, mask folders are {},{}".format(len(train_image_paths),len(train_mask_paths)))

    # Divide list into train, validation, test
    
    train_image_paths,val_image_paths,train_mask_paths,val_mask_paths = train_test_split(train_image_paths,train_mask_paths,test_size=validation_proportion,random_state=1)

    print("*"*50)
    print("Train: {}  Validation: {} ".format(len(train_image_paths),len(val_image_paths)))


    # Create Dataset
    train_dataset = MyLidcDataset(train_image_paths, train_mask_paths,config['augmentation'])
    val_dataset = MyLidcDataset(val_image_paths,val_mask_paths,config['augmentation'])
    #test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
    # Create Dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=2)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=2)

    log= pd.DataFrame(index=[],columns= ['epoch','lr','loss','iou','dice','val_loss','val_iou'])

    best_iou = 0
    trigger = 0



    for epoch in range(config['epochs']):

        # train for one epoch
        train_log = train(train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(val_loader, model, criterion)


        print('Training epoch [{}/{}], Training BCE loss:{:.4f}, Training DICE:{:.4f}, Training IOU:{:.4f}, Validation BCE loss:{:.4f}, Validation Dice:{:.4f}, Validation IOU:{:.4f}'.format(
            epoch + 1, config['epochs'], train_log['loss'], train_log['dice'], train_log['iou'], val_log['loss'], val_log['dice'],val_log['iou']))

        tmp = pd.Series([
            epoch,
            config['lr'],
            train_log['loss'],
            train_log['iou'],
            train_log['dice'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice']
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice', 'val_loss', 'val_iou','val_dice'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('model_outputs/{}/log.csv'.format(file_name), index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'model_outputs/{}/model.pth'.format(file_name))
            best_iou = val_log['iou']
            print("=> saved best model as validation IOU is greater than previous best IOU")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
