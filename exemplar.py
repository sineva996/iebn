import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from torchvision import datasets
from PIL import Image
import utils_transform as utils
from net.Network import ResNet18_IEBN as IEBN
import random
import time
from contextlib import contextmanager
from typing import Iterable
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import Lambda

class Exemplar:
    def __init__(self, exemplar_num, order,transform = None):     
        self.memory_size = exemplar_num
        self.order = order
        self.novel_cls = 0
        self.train_images = np.zeros(shape=(0, self.memory_size, 3, 224, 224))
        self.train_label =  np.zeros(shape=(0, self.memory_size), dtype=int)
        self.train_images = torch.tensor(self.train_images)
        self.train_label =  torch.tensor(self.train_label)   
        
    
    def update(self, images_train,labels_train,itera):
        self.novel_cls = (len(self.order[itera]))
        # reduce old memory
        if itera == 1:
            self.memory_size = self.memory_size//2
        elif itera == 2:
            self.memory_size = self.memory_size//4

        if itera >0:
            self.train_images = self.train_images[:, :self.memory_size]
            self.train_label  = self.train_label[:, :self.memory_size]
        
        # add new memory
        memory_new_images = np.zeros((self.novel_cls, self.memory_size, 3, 224, 224))
        memory_new_labels = np.zeros((self.novel_cls, self.memory_size), dtype=int)
        memory_new_images = torch.tensor(memory_new_images)
        memory_new_labels = torch.tensor(memory_new_labels)
        
        for k in range(self.novel_cls):
            img = images_train[self.order[itera-1][k], torch.randperm(500)[:self.memory_size]]
            tar = labels_train[self.order[itera-1][k], self.memory_size]
            
            if self.transform is not None:
                img = self.transform(img)

            memory_new_images[k] = img
            #memory_new_labels[k] = torch.tile(self.order[k], (self.memory_size))
            memory_new_labels[k] = tar

        # herding
        if itera > 0:
            self.train_images = torch.cat((self.train_images, memory_new_images),dim=1)
            self.train_label  = torch.cat((self.train_label, memory_new_labels),dim=1)
        else:
            self.train_images = memory_new_images
            self.train_label = memory_new_labels
    
    def get_exemplar_train(self):
        exemplar_train_x = self.train_images
        exemplar_train_y = self.train_label

        return exemplar_train_x, exemplar_train_y

class ExemplarsSelector:
    """具有数据集接口的方法的Exemplar选择"""

    def __init__(self, exemplars_dataset: data.Dataset):
        self.exemplars_dataset = exemplars_dataset

    def __call__(self, model: IEBN, trn_loader: DataLoader, transform):
        clock0 = time.time()
        exemplars_per_class = self._exemplars_per_class_num(model)
        with transform(trn_loader.dataset, transform) as ds_for_selection:
            # change loader and fix to go sequentially (shuffle=False), keeps same order for later, eval transforms
            sel_loader = DataLoader(ds_for_selection, batch_size=trn_loader.batch_size, shuffle=False,
                                    num_workers=trn_loader.num_workers, pin_memory=trn_loader.pin_memory)
            selected_indices = self._select_indices(model, sel_loader, exemplars_per_class, transform)
        with transform(trn_loader.dataset, Lambda(lambda x: np.array(x))) as ds_for_raw:
            x, y = zip(*(ds_for_raw[idx] for idx in selected_indices))
        clock1 = time.time()
        print('| Selected {:d} train exemplars, time={:5.1f}s'.format(len(x), clock1 - clock0))
        return x, y

    def _exemplars_per_class_num(self, model: IEBN):
        if self.exemplars_dataset.max_num_exemplars_per_class:
            return self.exemplars_dataset.max_num_exemplars_per_class

        num_cls = model.task_cls.sum().item()
        num_exemplars = self.exemplars_dataset.max_num_exemplars
        exemplars_per_class = int(np.ceil(num_exemplars / num_cls))
        assert exemplars_per_class > 0, \
            "Not enough exemplars to cover all classes!\n" \
            "Number of classes so far: {}. " \
            "Limit of exemplars: {}".format(num_cls,
                                            num_exemplars)
        return exemplars_per_class

    def _select_indices(self, model: IEBN, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        pass

class HerdingExemplarsSelector(data.Dataset):

    def __init__(self, exemplars_dataset):
        super().__init__(exemplars_dataset)

    def _select_indices(self, model: IEBN, sel_loader: DataLoader, exemplars_per_class: int, transform) -> Iterable:
        model_device = next(model.parameters()).device  # 整个模型都在一个设备上

        # 从模型中提取所有训练样本的输出
        extracted_features = []
        extracted_targets = []
        with torch.no_grad():
            model.eval()
            for images, targets in sel_loader:
                feats = model(images.to(model_device), return_features=True)[1]
                feats = feats / feats.norm(dim=1).view(-1, 1)  # Feature normalization
                extracted_features.append(feats)
                extracted_targets.extend(targets)
        extracted_features = (torch.cat(extracted_features)).cpu()
        extracted_targets = np.array(extracted_targets)
        result = []

        for curr_cls in np.unique(extracted_targets):

            cls_ind = np.where(extracted_targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (exemplars_per_class <= len(cls_ind)), "Not enough samples to store"

            cls_feats = extracted_features[cls_ind]

            cls_mu = cls_feats.mean(0)

            selected = []
            selected_feat = []
            for k in range(exemplars_per_class):

                sum_others = torch.zeros(cls_feats.shape[1])
                for j in selected_feat:
                    sum_others += j / (k + 1)
                dist_min = np.inf

                for item in cls_ind:
                    if item not in selected:
                        feat = extracted_features[item]
                        dist = torch.norm(cls_mu - feat / (k + 1) - sum_others)
                        if dist < dist_min:
                            dist_min = dist
                            newone = item
                            newonefeat = feat
                selected_feat.append(newonefeat)
                selected.append(newone)
            result.extend(selected)
        return result


class Exemplar_rafdb(data.Dataset):
    def __init__(self, raf_path, order, iter, exemplar_num , phase ,transform = None):
        
        self.phase = phase 
        self.transform = transform 
        self.raf_path = raf_path 
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        
        if phase == 'train':
            df = df[df['name'].str.startswith('train')]    
        
        self.df_exemplar = df
        
        train_lab1 = order[0][0]
        train_lab2 = order[0][1]
        train_lab3 = order[1][0]
        train_lab4 = order[1][1]

        df1 = df[(df['label']==(train_lab1+1))]
        df2 = df[(df['label']==(train_lab2+1))]
        df3 = df[(df['label']==(train_lab3+1))]
        df4 = df[(df['label']==(train_lab4+1))]
        if iter == 1:
            df1 = df1.sample(n=int(exemplar_num/2),frac=None,replace=False)
            df2 = df2.sample(n=int(exemplar_num/2),frac=None,replace=False)
            self.df_exemplar = pd.concat([df1,df2])
        elif iter == 2:
            df1 = df1.sample(n=int(exemplar_num/4),frac=None,replace=False)
            df2 = df2.sample(n=int(exemplar_num/4),frac=None,replace=False)
            df3 = df3.sample(n=int(exemplar_num/4),frac=None,replace=False)
            df4 = df4.sample(n=int(exemplar_num/4),frac=None,replace=False)
            self.df_exemplar = pd.concat([df1,df2,df3,df4])
    
        file_names = self.df_exemplar.loc[:, 'name'].values
        self.label = self.df_exemplar.loc[:, 'label'].values - 1
            
        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned/', f)
            self.file_paths.append(path)
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx] 
        image = Image.open(path).convert('RGB')
        label = self.label[idx] 

        if self.transform is not None:
            image = self.transform(image) 
        
        return image, label 

    def get_df(self):
        df_temp = self.df_exemplar
        return df_temp

class Exemplar_affectnet(data.Dataset):
    def __init__(self, aff_path, order, iter, exemplar_num , phase, transform = None):
        
        self.phase = phase 
        self.transform = transform 
        self.aff_path = aff_path
        df = pd.read_csv(os.path.join(self.aff_path, 'EmoLabel/affectnet_new_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        
        if phase == 'train':
            df = df[df['name'].str.startswith('train')]    
        
        self.df_exemplar = df

        train_lab1 = order[0][0]
        train_lab2 = order[0][1]
        train_lab3 = order[1][0]
        train_lab4 = order[1][1]
        df1 = df[(df['label']==(train_lab1+1))]
        df2 = df[(df['label']==(train_lab2+1))]
        df3 = df[(df['label']==(train_lab3+1))]
        df4 = df[(df['label']==(train_lab4+1))]
        if iter == 1:
            df1 = df1.sample(n=int(exemplar_num/2),frac=None,replace=False)
            df2 = df2.sample(n=int(exemplar_num/2),frac=None,replace=False)

            self.df_exemplar = pd.concat([df1,df2])
        elif iter == 2:
            df1 = df1.sample(n=int(exemplar_num/4),frac=None,replace=False)
            df2 = df2.sample(n=int(exemplar_num/4),frac=None,replace=False)
            df3 = df3.sample(n=int(exemplar_num/4),frac=None,replace=False)
            df4 = df4.sample(n=int(exemplar_num/4),frac=None,replace=False)

            self.df_exemplar = pd.concat([df1,df2,df3,df4])
        
        file_names = self.df_exemplar.loc[:, 'name'].values
        self.label = self.df_exemplar.loc[:, 'label'].values - 1
            
        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.aff_path, 'Image/', f)
            self.file_paths.append(path)
    
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx] 
        image = Image.open(path).convert('RGB')
        label = self.label[idx] 

        if self.transform is not None:
            image = self.transform(image) 
        
        return image, label 

    def get_df(self):
        df_temp = self.df_exemplar
        return df_temp
       