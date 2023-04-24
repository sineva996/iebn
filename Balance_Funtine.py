import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from PIL import Image

class Balance_funtine:
    def __init__(self, exemplar_num,total_class,transform = None):
        self.total_class = total_class
        self.memory_size = exemplar_num*(self.total_class)
        self.transform = transform

        self.train_images = np.zeros(shape=(0, self.memory_size, 3, 224, 224))
        self.train_label =  np.zeros(shape=(0, self.memory_size), dtype=int)
        self.train_images = torch.tensor(self.train_images)
        self.train_label =  torch.tensor(self.train_label)      


    def construct_set(self, images_train):
        # add new data
        memory_new_images = np.zeros((self.total_class, self.memory_size, 3, 224, 224))
        memory_new_labels = np.zeros((self.total_class, self.memory_size), dtype=int)
        memory_new_images = torch.tensor(memory_new_images)
        memory_new_labels = torch.tensor(memory_new_labels)
        
        for k in range(self.total_class):
            img = images_train[k, torch.randperm(500)[:self.memory_size]]
            
            if self.transform is not None:
                img = self.transform(img)

            memory_new_images[k] = img
            memory_new_labels[k] = torch.tile(k, (self.memory_size))

        # herding
        self.train_images = torch.cat((self.train_images, memory_new_images),dim=1)
        self.train_label  = torch.cat((self.train_label, memory_new_labels),dim=1)

    def get_Balance_funtine(self):
        train_x = self.train_images
        train_y = self.train_label

        return train_x, train_y


class Balance_rafdb(data.Dataset):
    def __init__(self, raf_path,exemplar_num , phase, transform = None):
        
        self.phase = phase 
        self.transform = transform 
        self.raf_path = raf_path 

        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])

        if phase == 'train':
            df = df[df['name'].str.startswith('train')]    
        
        self.df_balance =  df

        for i in range(7):
            locals()[f'df{i+1}']=df[(df['label']==(i+1))]
        
        df1 = df1.sample(n=int(exemplar_num*1.0),replace=False)
        df2 = df2.sample(n=int(exemplar_num*1.0),replace=True)
        df3 = df3.sample(n=int(exemplar_num*1.5), replace=False)
        df4 = df4.sample(n=int(exemplar_num*1.5),replace=False)
        df5 = df5.sample(n=int(exemplar_num*1.5),replace=False)
        df6 = df6.sample(n=int(exemplar_num*1.0),replace=False)
        df7 = df7.sample(n=int(exemplar_num*1.5),replace=False)

        self.df_balance = pd.concat([df1,df2,df3,df4,df5,df6,df7])

        file_names = self.df_balance.loc[:, 'name'].values
        self.label = self.df_balance.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

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
        df_temp = self.df_balance
        return df_temp


class Balance_affectnet(data.Dataset):
    def __init__(self, aff_path,exemplar_num , phase, transform = None):
        
        self.phase = phase
        self.transform = transform 
        self.aff_path = aff_path

        df = pd.read_csv(os.path.join(self.aff_path, 'EmoLabel/affectnet_new_label.txt'), 
                        sep=' ', header=None,names=['name','label'])

        if phase == 'train':
            df = df[df['name'].str.startswith('train')]    
        
        self.df_balance =  df

        for i in range(7):
            locals()[f'df{i+1}']=df[(df['label']==(i+1))]
        
        df1 = df1.sample(n=int(exemplar_num*1.0),replace=False,weights=None, random_state=None, axis=None)
        df2 = df2.sample(n=int(exemplar_num*1.0),replace=False,weights=None, random_state=None, axis=None)
        df3 = df3.sample(n=int(exemplar_num*1.0),replace=False,weights=None, random_state=None, axis=None)
        df4 = df4.sample(n=int(exemplar_num*1.0),replace=False,weights=None, random_state=None, axis=None)
        df5 = df5.sample(n=int(exemplar_num*1.0),replace=False,weights=None, random_state=None, axis=None)
        df6 = df6.sample(n=int(exemplar_num*1.0),replace=False,weights=None, random_state=None, axis=None)
        df7 = df7.sample(n=int(exemplar_num*1.0),replace=False,weights=None, random_state=None, axis=None) 

        self.df_balance = pd.concat([df1,df2,df3,df4,df5,df6,df7])

        file_names = self.df_balance.loc[:, 'name'].values
        self.label = self.df_balance.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

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
        df_temp = self.df_balance
        return df_temp
