# IEBN
A PyTorch implementation of the [IEBN].

##requirements
certifi==2021.10.8
colorama==0.4.4
joblib==1.1.0
mkl-fft==1.3.1
mkl-random
mkl-service==2.4.0
numpy==1.22.0
olefile 
opencv-python==4.5.5.62
pandas==1.3.5
Pillow==9.0.0
python-dateutil==2.8.2
pytz==2021.3
scikit-learn==1.0.2
scipy==1.7.3
six 
sklearn==0.0
threadpoolctl==3.0.0
torch==1.10.1
torchvision==0.11.2
tqdm==4.62.3
typing_extensions==4.0.1
wincertstore==0.2

## Preparation
- Download pre-trained model of [MSCeleb](https://drive.google.com/file/d/1H421M8mosIVt8KsEWQ1UuYMkQS8X1prf/view?usp=sharing) and move the file to `./pre-models`
- Download [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset and extract the `raf-basic` dir to `./datasets`
- Download [AffectNet](http://mohammadmahoor.com/affectnet/) dataset and extract the `AffectNet` dir  to `./datasets` 
- Convert data of AffectNet to the same format as RAF-DB.
for example:
"0000f8a4575c15055a9ee0a72c9aa5bf9ac00558173565802479a287.jpg"→"train_0000f8a4575c15055a9ee0a72c9aa5bf9ac00558173565802479a287.jpg"
"ff8e6ab180b28a4b07f6f2c071033745aa852519a851c49407eb6bbb.jpg"→"test_ff8e6ab180b28a4b07f6f2c071033745aa852519a851c49407eb6bbb.jpg"

- We provide the new label file of AffectNet, affectnet_new_label.txt.


## Training
We provide the training code for AffectNet and RAF-DB.  

For AffectNet-7 dataset, run:
```
CUDA_VISIBLE_DEVICES=0 python train_affectnet.py
```

For RAF-DB dataset, run:
```
CUDA_VISIBLE_DEVICES=0 python train_rafdb.py
```

## Evaluating
We provide the checkpoint models for AffectNet and RAF-DB. 

- cancel the necessary  annotation in the code to evaluate the model for AffectNet and RAF-DB. 
- change the the name of checkpoint model in the code to evaluate different models.
```
python test_model.py
```
