import warnings
warnings.filterwarnings("ignore")
import os
import argparse
from PIL import Image
import numpy as np
import pandas as pd
import torch
import copy

from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from net.Network import ResNet18_IEBN as IEBN
import dataset
import Balance_Funtine
import exemplar 
import Loss
#import plotCM as pt
import utils_transform as utils
import torch.nn.functional as F

parser = argparse.ArgumentParser()

parser.add_argument('--aff_path', type=str, default='datasets/affectnet/', help='AfectNet dataset path.')#数据集地址
parser.add_argument('--num_class', type=int, default=7, help='number of class.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')#批次大小
parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for adam.')#学习率
parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')#读取数据头数
parser.add_argument('--epochs', type=int, default=15, help='Total training epochs.')#训练批次数量
parser.add_argument('--Temp', type=int, default=2, help='Temperature Coefficient of Knowledge Distillation.')#知识蒸馏的温度系数
parser.add_argument('--alphaKD', type=float, default=0.8, help='loss factor.')#损失系数
parser.add_argument('--db_head', type=int, default=1, help='Number of distributed workers.')#分配工作数
parser.add_argument("--exemplars", default=500, type=int, help="total number of exemplars")#记忆样本数目
parser.add_argument("--select", action="store_true",default=False, help="use herding selection")#羊群算法，挑选记忆样本
parser.add_argument("--proc", default=[2, 2, 3], type=list, help="stage-wise numbers of classes")#不同阶段的类别数目
parser.add_argument("--order", default=[[0,4],[3,6],[1,2,5]], type=list, help="Classes at stage-wise of learning")#不同阶段的学习的类别
parser.add_argument("--confusion", action="store_true", default=True, help="show confusion matrix")#混淆矩阵

args = parser.parse_args()
def training():

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    else:
        device = torch.device("cpu")


    model = IEBN(num_class=args.num_class)
    model.to(device)
  
    train_data_transforms = utils.data_transforms_af()
    test_data_transforms = utils.val_data_transforms_af()

    featpara =Loss.FeatLoss(device).parameters()
    train_dataset = dataset.AffDataSet(args.aff_path, phase = 'train', transform = train_data_transforms)    
    test_dataset = dataset.AffDataSet(args.aff_path, phase = 'test', transform = test_data_transforms)   
    
    print('The AffectNet train set size:', train_dataset.__len__())
    print('The AffectNet test set size:', test_dataset.__len__())
    

    """ train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True) """
    val_loader = torch.utils.data.DataLoader(test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
    
   
    #class_counts = torch.from_numpy( np.array([14090,    6378,   3803,   134415, 25459,  24882,  74874]).astype(np.float32))
    """ class_counts2 = torch.from_numpy( np.array([500,    0,  0,   500,  25459,  0,  74874]).astype(np.float32))  #03-46-125
    class_counts3 = torch.from_numpy( np.array([500,    6378,   3830,   500,  500,  24882,  500]).astype(np.float32)) """

    class_counts2 = torch.from_numpy( np.array([500,    0,  0,   134415, 500,  0,  74874]).astype(np.float32))  #04-36-125
    class_counts3 = torch.from_numpy( np.array([500,    6378,   3830,   500,  500,  24882,  500]).astype(np.float32))

    """ class_counts2 = torch.from_numpy( np.array([14090,    0,  0,   500,  500,  0,  74874]).astype(np.float32))  #34-06-125
    class_counts3 = torch.from_numpy( np.array([500,    6378,   3830,   500,  500,  24882,  500]).astype(np.float32))  """
    class_weight2 = (torch.sum(class_counts2) - class_counts2) / torch.sum(class_counts2) 
    class_weight3 = (torch.sum(class_counts3) - class_counts3) / torch.sum(class_counts3)

    cro_loss = Loss.cross_loss().to(device)
    pt_loss  = Loss.PartitionLoss(device)#device = torch.device("cuda:0") or device = torch.device("cpu")
    kl_loss  = Loss.kl_loss()
    
    for itera in range(len(args.proc)):
        """ if itera == 0:
        else: """
        df=[]
        best_acc = 0.0
        print ("\n")
        print("Stage {} of {} stages".format((itera+1), len(args.proc)))
        if itera == 1:
            cro_loss = torch.nn.CrossEntropyLoss(weight=class_weight2).to(device)
        elif itera ==2:
            cro_loss = torch.nn.CrossEntropyLoss(weight=class_weight3).to(device)
        
        if itera > 0:
            model_old = copy.deepcopy(model)
            exemplar_set = exemplar.Exemplar_affectnet(args.aff_path, args.order, itera, args.exemplars ,phase = 'train',transform = train_data_transforms)
            df = exemplar_set.get_df()
            exemplar_loader =   torch.utils.data.DataLoader(exemplar_set,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)
        
        inc_train_dataset = dataset.Aff_increData(args.aff_path, args.order, itera,df ,phase = 'train', transform = train_data_transforms)
        inc_test_dataset = dataset.Aff_increData(args.aff_path,  args.order, itera,df ,phase = 'test', transform = test_data_transforms)
        print("The AffectNet train set size of {} stage: {}".format((itera+1), inc_train_dataset.__len__())) 
        print("The AffectNet test set size of {} stage: {}".format((itera+1), inc_test_dataset.__len__()))
        #images_train=[]
        #labels_train=[]
        inc_train_loader = torch.utils.data.DataLoader(inc_train_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = True,  
                                               pin_memory = True)                                         
        inc_val_loader = torch.utils.data.DataLoader(inc_test_dataset,
                                               batch_size = args.batch_size,
                                               num_workers = args.workers,
                                               shuffle = False,  
                                               pin_memory = True)
        #set = exemplar.Exemplar(args.exemplar, args.order,transform = train_data_transforms)
        #set.updateimages_train,labels_train,itera)
        params = list(model.parameters()) + list(featpara)
        #params = list(model.parameters())   
        optimizer = torch.optim.Adam(params,lr=args.lr, weight_decay = 0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

        for epoch in range(1, args.epochs + 1): 
            running_loss = 0.0  
            correct_sum = 0     
            iter_cnt = 0        
            acc_break = 0.0     
            

            model.train()
            for (imgs, targets) in inc_train_loader:
                #images_train=imgs
                #labels_train=targets
                iter_cnt += 1 
                optimizer.zero_grad()

                imgs = imgs.to(device)
                targets = targets.to(device)
                out,feat,heads = model(imgs)

                #if itera >0:
                loss = cro_loss(out,targets) + pt_loss.loss(feat,targets,heads)
                    #loss = cro_loss(out,targets) 
                #else:
                    #loss = cro_loss(out,targets)
                
                loss.backward()
                optimizer.step()
            
                running_loss += loss
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num
            

            acc = correct_sum.float() / float(inc_train_dataset.__len__())
            acc_break = acc
            running_loss = running_loss/iter_cnt
            print('Epoch %d : Training accuracy: %.4f. Loss: %.3f. LearningRate %.6f' % (epoch, acc, running_loss,optimizer.param_groups[0]['lr']))


            if itera > 0:
                model.train()
                #for (imgs,targets) in set:             
                for (imgs,targets) in exemplar_loader:
                    
                    optimizer.zero_grad()                    
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    out_old,feat_old,heads_old = model_old(imgs)
                    out,feat,heads = model(imgs)

                    loss_hard = cro_loss(out,targets)
                    outputs_s = F.log_softmax(out/args.Temp,dim=1)
                    outputs_t = F.softmax(out_old/args.Temp,dim=1)
                    loss_soft = kl_loss(outputs_s,outputs_t) 

                    loss = Loss.dist_loss(loss_soft,loss_hard,args.Temp,args.alphaKD)
                    loss.backward()
                    optimizer.step() 

            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                baccs = []

                model.eval()
                for (imgs, targets) in inc_val_loader:
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                
                    out,feat,heads = model(imgs)
                    #loss = loss = cro_loss(out,targets) + pt_loss.loss(feat,targets,heads)
                    loss = cro_loss(out,targets) 

                    running_loss += loss
                    iter_cnt+=1
                    _, predicts = torch.max(out, 1)
                    correct_num  = torch.eq(predicts,targets)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += out.size(0)#torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
                
                running_loss = running_loss/iter_cnt   
                scheduler.step()

                acc = bingo_cnt.float()/float(sample_cnt)
                acc = np.around(acc.numpy(),4)
                best_acc = max(acc,best_acc)

                bacc = np.around(np.mean(baccs),4)
                print("Epoch %d : Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, bacc, running_loss))
                print("best_acc:" + str(best_acc))
            
            if ((epoch > 12) and (acc_break > 0.80)):
                break


    Balance_Exemplar = Balance_Funtine.Balance_affectnet(args.aff_path, args.exemplars ,phase = 'train', transform = train_data_transforms)
    Balance_Exemplar_loader = torch.utils.data.DataLoader(Balance_Exemplar , batch_size = args.batch_size , num_workers=args.workers , shuffle = True, pin_memory = True)
    #Balance_set = Balance_Funtine.Balance_funtine(args.exemplar_num,sum(args.proc),transform = train_data_transforms).construct_set(images_train)
    optimizer_balance = torch.optim.Adam(params,lr=0.4*args.lr, weight_decay = 0)#0.8;  #0.8;   #0.5;   #0.4
    scheduler_blance = torch.optim.lr_scheduler.StepLR(optimizer_balance, step_size=5, gamma=0.5)#5,0.8;    #5,0.7;  #5,0.7;    #5,0.7
    #scheduler_blance = torch.optim.lr_scheduler.StepLR(optimizer_balance, gamma=0.6)
    
    print ("\n")
    print("Balance Training:")

    #class_weight = torch.from_numpy( np.array([1.0,    0.7,   1.1,   0.25, 0.7,  0.7,  0.5]).astype(np.float32)) #best 0.05xx
    #class_weight = torch.from_numpy( np.array([1.0,    1.0,   1.0,   1.0, 1.0,  1.0,  1.0]).astype(np.float32))
    #class_weight = torch.from_numpy( np.array([0.5,    0.6,   0.6,   0.4, 0.6,  0.6,  0.5]).astype(np.float32))
    class_weight = torch.from_numpy( np.array([1.0,    0.8,   0.8,   0.35, 0.6,  0.6,  0.5]).astype(np.float32))

    cro_loss = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    for epoch in range(25): 
        if(epoch % 5 == 0):
            Balance_Exemplar = Balance_Funtine.Balance_affectnet(args.aff_path, args.exemplars ,phase = 'train', transform = train_data_transforms)
            Balance_Exemplar_loader = torch.utils.data.DataLoader(Balance_Exemplar , batch_size = args.batch_size , num_workers=args.workers , shuffle = True, pin_memory = True)
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for (imgs, targets) in Balance_Exemplar_loader:
        #for (imgs, targets) in train_loader:       
            iter_cnt += 1 
            optimizer_balance.zero_grad()

            imgs = imgs.to(device)
            targets = targets.to(device)

            out,feat,heads = model(imgs)
            loss = loss = cro_loss(out,targets) + pt_loss.loss(feat,targets,heads)
            #loss = loss = cro_loss(out,targets)

            loss.backward()
            optimizer_balance.step()

            running_loss += loss
            _, predicts = torch.max(out, 1)
            
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num
            
        acc = correct_sum.float() / float(Balance_Exemplar.__len__())
        running_loss = running_loss/iter_cnt
        print('Epoch %d : Training accuracy: %.4f. Loss: %.3f. LearningRate %.6f' % ((epoch+1), acc, running_loss,optimizer_balance.param_groups[0]['lr']))       
       
        p_label = []
        t_label = []
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            baccs = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                
                out,feat,heads = model(imgs)
                loss = cro_loss(out,targets) + pt_loss.loss(feat,targets,heads)
                #loss = cro_loss(out,targets)

                running_loss += loss
                iter_cnt+=1
                _, predicts = torch.max(out, 1)
                correct_num  = torch.eq(predicts,targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)
                
                p_label += predicts.cpu().tolist()
                t_label += targets.cpu().tolist() 
                baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
            running_loss = running_loss/iter_cnt   
            scheduler_blance.step()

            acc = bingo_cnt.float()/float(sample_cnt)
            acc = np.around(acc.numpy(),4)

            bacc = np.around(np.mean(baccs),4)
                
            print("Epoch %d : Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % ((epoch+1), acc, bacc, running_loss))
            
            """  if (acc > 0.6) & args.confusion:
                print('Confusion matrix:')
                print(metrics.confusion_matrix(t_label,p_label)) """
            
    pre_labels = []
    tar_labels = []
    with torch.no_grad():
        iter_cnt = 0
        bingo_cnt = 0
        sample_cnt = 0
        baccs = []
        model.eval()
        for (imgs, targets) in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)
                
            out,feat,heads = model(imgs)
            loss = cro_loss(out,targets) +  pt_loss.loss(feat,targets,heads)
            #loss = cro_loss(out,targets)

            iter_cnt+=1
            _, predicts = torch.max(out, 1)
            correct_num  = torch.eq(predicts,targets)
            bingo_cnt += correct_num.sum().cpu()
            sample_cnt += out.size(0)

            pre_labels += predicts.cpu().tolist()
            tar_labels += targets.cpu().tolist()    
            baccs.append(balanced_accuracy_score(targets.cpu().numpy(),predicts.cpu().numpy()))
            

        acc = bingo_cnt.float()/float(sample_cnt)
        acc = np.around(acc.numpy(),4)
        bacc = np.around(np.mean(baccs),4)
       
        print("The last Validation accuracy:%.4f. bacc:%.4f." % (acc, bacc))

        if args.confusion:
            print('Confusion matrix:')
            print(metrics.confusion_matrix(tar_labels,pre_labels))
    
    if acc>0.60:
        torch.save({'iter': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),},
                os.path.join('checkpoints', "affectnet"+".pth"))
        print('Model saved.')

if __name__ == "__main__":        
    training()
