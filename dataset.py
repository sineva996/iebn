import os
import numpy as np
import pandas as pd
import torch.utils.data as data
from PIL import Image

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform = None):
        self.phase = phase #判断训练或测试
        self.transform = transform #数据增强
        self.raf_path = raf_path #读取地址
        """高兴3    惊讶0   中立6   伤心4   生气5   恐惧1   厌恶2   总数
        总  5957    1619	3204	2460	867    355	  877	15339
        训  4772	1290	2524	1982	705	   281	  717	12271
        测  1185	329	    680	    478	    162	   74	  160	3068"""
        #df=
        """	            name	    label
            0	    train_00001.jpg	   5
            ...          ...          ...
            15338	test_3068.jpg	   7"""
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        
        if phase == 'train':
            #df['name'].str.startswith('train')=
            """ 0         True
                ...       ...  
                15338    False"""
            #df[df['name'].str.startswith('train')]=
            """	            name	    label
                0	    train_00001.jpg	   5
                ...          ...          ...
                12270	train_12271.jpg	   7"""
            self.data = df[df['name'].str.startswith('train')]    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        """file_names = ['train_00001.jpg', ... , 'train_12271.jpg']"""       
        """self.label = [4, 4, 3, ..., 6, 6, 6]"""

        """a,b=np.unique(c,return_index=True) ,去除c中重复的元素，并按元素由大到小排列返回给a，b为a的元素在c的位置
            c = [1, 2, 2, 5, 3, 4, 3]
            a = [1 2 3 4 5]
            c = [0 1 4 5 3]"""
        _, self.sample_counts = np.unique(self.label, return_counts=True)
        """_ = [0, 1, 2, 3, 4, 5, 6]"""
        """self.sample_counts = [1290,  281,  717, 4772, 1982,  705, 2524]
            且sum([1290,..., 2524]) = 12271，为训练集总数"""
        
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned/', f)
            self.file_paths.append(path)
        """self.file_paths = ['datasets/RAF-DB/Image/aligned/train_00001_aligned.jpg',...]"""

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签

class AffDataSet(data.Dataset):
    def __init__(self, aff_path, phase, transform = None):
        self.phase = phase #判断训练或测试
        self.transform = transform #数据增强
        self.aff_path = aff_path #读取地址
        """高兴3    惊讶0   中立6   伤心4   生气5   恐惧1   厌恶2   总数
        总  134915  14590	75374	25959	25382  6878	  4303	287401
        训  134415	14090	74874	25459	24882  6378	  3803	283901
        测  500 	500	    500	    500	    500	   500	  500	3500"""
        
        df = pd.read_csv(os.path.join(self.aff_path, 'EmoLabel/affectnet_new_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        
        if phase == 'train':
            #df[df['name'].str.startswith('train')]=
            """	            name	    label
                0	    train_xxx.jpg	   5"""
            self.data = df[df['name'].str.startswith('train')]    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.label = self.data.loc[:, 'label'].values - 1  
        """file_names = ['train_xxx.jpg', ... ,]"""       

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        """sum(self.sample_counts)为训练集总数"""

        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.aff_path, 'Image/', f)
            self.file_paths.append(path)
        """self.file_paths = ['datasets/affectnet/Image/train_xxx.jpg',...]"""

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签

class Raf_increData(data.Dataset):
    #def __init__(self, raf_path, phase, transform = None):
    #def __init__(self, raf_path, itera , phase, transform = None):
    def __init__(self, raf_path, order, iter, exem_df,phase, transform = None):
        
        self.phase = phase #判断训练或测试
        self.transform = transform #数据增强
        self.raf_path = raf_path #读取地址
        
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        """ if lab1 == lab2:#判断读取标签是否相同
            df = df[(df['label']==(lab1+1))]
        else:
            df = df[ (df['label']==(lab1+1)) | (df['label']==(lab2+1)) ] """
        if iter==0:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            df = df[ (df['label']==(train_lab1+1)) | (df['label']==(train_lab2+1)) ]
        elif iter==1:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            df = df[(df['label']==(train_lab1+1)) | (df['label']==(train_lab2+1)) ]
        else:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            train_lab3 = order[iter][2]
            df = df[ (df['label']==(train_lab1+1)) | (df['label']==(train_lab2+1)) |(df['label']==(train_lab3+1))]

        #df = df.reset_index(drop=True)#重新索引 

        """ if itera == 0:
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label_3_0.txt'), 
                        sep=' ', header=None,names=['name','label'])
            elif itera==1:
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label_6_6.txt'), 
                        sep=' ', header=None,names=['name','label'])
            elif itera==2: 
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label_4_2.txt'), 
                        sep=' ', header=None,names=['name','label'])
            elif itera==3: 
                df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label_5_1.txt'), 
                        sep=' ', header=None,names=['name','label']) """
        
        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
            if iter > 0:
                temp_df = exem_df
                self.data = pd.concat([self.data,temp_df])    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        _, self.sample_counts = np.unique(self.label, return_counts=True)
        
        # print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        for f in file_names:
            f = f.split(".")[0]
            f = f +"_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned/', f)
            self.file_paths.append(path)
        """self.file_paths = ['datasets/RAF-DB/Image/aligned/train_00001_aligned.jpg',...]"""

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签

class Aff_increData(data.Dataset):
    def __init__(self, aff_path, order, iter, exem_df,phase, transform = None):
        
        self.phase = phase #判断训练或测试
        self.transform = transform #数据增强
        self.aff_path = aff_path #读取地址
        
        df = pd.read_csv(os.path.join(self.aff_path, 'EmoLabel/affectnet_new_label.txt'), 
                        sep=' ', header=None,names=['name','label'])
        if iter==0:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            df = df[ (df['label']==(train_lab1+1)) | (df['label']==(train_lab2+1)) ]
        elif iter==1:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            df = df[(df['label']==(train_lab1+1)) | (df['label']==(train_lab2+1)) ]
        else:
            train_lab1 = order[iter][0]
            train_lab2 = order[iter][1]
            train_lab3 = order[iter][2]
            df = df[ (df['label']==(train_lab1+1)) | (df['label']==(train_lab2+1)) |(df['label']==(train_lab3+1))]

        #df = df.reset_index(drop=True)#重新索引 
        
        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]
            if iter > 0:
                temp_df = exem_df
                self.data = pd.concat([self.data,temp_df])    
        else:
            self.data = df[df['name'].str.startswith('test')]
   
        file_names = self.data.loc[:, 'name'].values
        self.label = self.data.loc[:, 'label'].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        
        _, self.sample_counts = np.unique(self.label, return_counts=True)
        
        self.file_paths = []
        for f in file_names:
            path = os.path.join(self.aff_path, 'Image/', f)
            self.file_paths.append(path)
        """self.file_paths = ['datasets/affectnet/Image/train_xxx.jpg',...]"""

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签

#----------------------------------------------------------
""" class Raf_incre_wholevalData(data.Dataset):
    def __init__(self, raf_path, order, iter, transform = None):
        self.phase = 'test' #判断训练或测试
        self.transform = transform #数据增强
        self.raf_path = raf_path #读取地址
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), 
                        sep=' ', header=None,names=['name','label'])

    def __len__(self):
        return len(self.file_paths)#返回文件地址个数，即训练或测试集个数

    def __getitem__(self, idx):
        path = self.file_paths[idx] #idx索引的文件地址 
        image = Image.open(path).convert('RGB')#以RGB的方式，读取文件
        label = self.label[idx] #idx索引的表情标签

        if self.transform is not None:
            image = self.transform(image) #转换图片(旋转、平移、、、)
        
        return image, label #返回图像和标签 """