import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from net.Network import ResNet18_IEBN as IEBN
import dataset
import Loss
import utils_transform as utils
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--aff_path', type=str, default='datasets/affectnet/', help='AffectNet dataset path.')
parser.add_argument('--raf_path', type=str, default='datasets/RAF-DB/', help='RAF-DB dataset path.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers.')
parser.add_argument("--confusion", action="store_true", default=True, help="show confusion matrix")
args = parser.parse_args()

def test():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device("cpu")
        
    model = IEBN()
    
    """ #model-RAF-DB
    checkpoint = torch.load('./checkpoints/rafdb-85.30-0.0837.pth') """
    
    #model-AffectNet
    checkpoint = torch.load('./checkpoints/affectnet-60.71-0.0876.pth')
    
    model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    model.to(device)

    #RAF-DB
    """ test_data_transforms = utils.val_data_transforms()
    test_dataset = dataset.RafDataSet(args.raf_path, phase = 'test', transform = test_data_transforms)  
    print('The RAF-DB test set size:', test_dataset.__len__()) """
    
    #AffectNet
    test_data_transforms = utils.val_data_transforms_af()
    test_dataset = dataset.AffDataSet(args.aff_path, phase = 'test', transform = test_data_transforms)
    print('The AffectNet test set size:', test_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    
    pre_labels = []
    tar_labels = []
    with torch.no_grad():
        bingo_cnt = 0
        sample_cnt = 0
        model.eval()
        for (imgs, targets) in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)            
            out,feat,heads = model(imgs)   
            _, predicts = torch.max(out, 1)
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)
            pre_labels += predicts.cpu().tolist()
            tar_labels += targets.cpu().tolist()
            
        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        print("The last Validation accuracy:%.4f. " % (acc))

        if args.confusion:
            print('Confusion matrix:')
            print(confusion_matrix(tar_labels,pre_labels))

    """ if args.confusion:
        cm = confusion_matrix(tar_labels, pre_labels)
        cm = np.array(cm)
        labels_name = ['SU', 'FE', 'DI', 'HA', 'SA', 'AN', "NE"]
        #plot_confusion_matrix(cm, labels_name, 'RAF-DB', acc)
        plot_confusion_matrix(cm, labels_name, 'AffectNet', acc) """


if __name__ == "__main__":                    
    test()
