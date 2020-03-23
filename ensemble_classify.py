

import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch import autograd, nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
import time
from PIL import Image
from basemod import Basemodule
from torch.utils.data import Dataset, DataLoader,ConcatDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#constansts
num_classes=2300
Batch_size=512
epochs=100
lr=0.01
gamma=0.9
text_file='./test_order_classification.txt'
test_folder='./test_classification/medium/'
id_list=[str(i) for i in range(num_classes)]
id_list=sorted(id_list)
output='./id_list.csv'
saved_model_path='./saved_model1.pt'
model2='./saved_model2.pt'
model3='./saved_model3.pt'
model4='./saved_model4.pt'

#data transform
data_transform=transforms.Compose([transforms.RandomHorizontalFlip(),
  transforms.RandomRotation(30),
  transforms.ToTensor()])

#train and validation dataloader
train_in = datasets.ImageFolder(root='train_data/medium', 
                                transform=data_transform)
#train_in2=datasets.ImageFolder(root='train_data/large',
 # transform=data_transform)

#train_in=ConcatDataset([train_in1, train_in2])
train_dataset = torch.utils.data.DataLoader(train_in,
                                             batch_size=Batch_size,
                                             shuffle=True,
                                             num_workers=1)


valid_in = datasets.ImageFolder(root='validation_classification/medium',
                                           transform=data_transform)

valid_dataset = torch.utils.data.DataLoader(valid_in,
                                             batch_size=Batch_size,
                                             shuffle=True,
                                             num_workers=1)

#class to make test dataset

class Testdataset(Dataset):
  def __init__(self,image_list):
    self.image_list=image_list

  def __len__(self):
    return(len(self.image_list))
  
  def __getitem__(self,index):
    image=Image.open(test_folder+self.image_list[index])
    image=transforms.ToTensor()(image)
    label=0
    return image, label

#making of test dataset
with open(text_file) as f:
  image_list=[num.rstrip('\n') for num in f]
test_in=Testdataset(image_list)
#test dataloader
test_dataset = torch.utils.data.DataLoader(test_in,
                                             batch_size=Batch_size,
                                             shuffle=False,
                                             num_workers=1)

#classification function
def test_inference_classify(model2,model3,model4, dataset,device):
  pred_labels=[]
  model2.eval()
  model3.eval()
  model4.eval()
  with torch.no_grad(): #what is this?
    for (x,y) in dataset: #test_dataset
      x=x.to(device)
      y=y.to(device)
      #testset1=model1(x)
      testset2=model2(x)
      testset3=model3(x)
      testset4=model4(x)
      list_testset=[testset2,testset3,testset4]
   
      mean_testset=torch.mean(torch.stack(list_testset),dim=0)
      

      _,pred=torch.max(F.softmax(mean_testset, dim=1),1) # what is this?
      pred=pred.view(-1)
      pred_labels.extend(pred.cpu().numpy())
 
    return np.array(pred_labels)


#mapping function
def mapping(id_list, pred_labels):
  id_list=[id_list[i] for i in pred_labels]
  return id_list

  
# preparing for training

mymodel=Basemodule()

optimizer=optim.SGD(mymodel.parameters(),lr=lr, momentum=0.9, weight_decay=5e-4,nesterov=True)
lr_scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=1,gamma=gamma)
losstype=nn.CrossEntropyLoss()
mymodel.to(device)
losstype.to(device)
#mymodel.load_state_dict(torch.load(saved_model_path))
#accuracy

def accuracy(dataset,y):
  pred=dataset.argmax(1,keepdim=True)
  accurate=pred.eq(y.view_as(pred)).sum()
  accuracy=accurate.float()/(y.shape[0])
  return(accuracy)

#training function

def train(model,dataset,optimizer,losstype,device):
  epoch_loss=0
  epoch_acc=0
  model.train() 
  for (x,y) in dataset: 
    x=x.to(device)
    y=y.to(device)
    
    optimizer.zero_grad()
    trainset=model(x)
    loss=losstype(trainset,y)
    acc=accuracy(trainset,y)
    loss.backward()
    optimizer.step()
    epoch_loss+=loss.item()
    epoch_acc+=acc.item()
  epoch_loss=epoch_loss/len(dataset)
  epoch_acc=epoch_acc/len(dataset)
  return epoch_loss, epoch_acc

#validation function
def validation(model, dataset, losstype,device):
  epoch_testloss=0
  epoch_testacc=0
  
  model.eval()
  with torch.no_grad():
    for (x,y) in dataset:
      x=x.to(device)
      y=y.to(device)
      testset=model(x)
      loss=losstype(testset,y)
      acc=accuracy(testset,y)
      epoch_testloss+=loss.item()
      epoch_testacc+=acc.item()
 
  return epoch_testloss/len(dataset), epoch_testacc/len(dataset)

#final testing
def testing():
  #learned_model=Basemodule()
  learned_model2=Basemodule()
  learned_model3=Basemodule()
  learned_model4=Basemodule()
  #learned_model=learned_model.to(device)
  learned_model2=learned_model2.to(device)
  learned_model3=learned_model3.to(device)
  learned_model4=learned_model4.to(device)

  #learned_model.load_state_dict(torch.load(saved_model_path))
  learned_model2.load_state_dict(torch.load(model2))
  learned_model3.load_state_dict(torch.load(model3))
  learned_model4.load_state_dict(torch.load(model4))

  pred_labels=test_inference_classify(learned_model2,learned_model3,learned_model4, test_dataset, device)
  correct_id_list=mapping(id_list, pred_labels)

  with open(output, 'w') as out:
    print('Id', 'Category', sep=',', file=out)
    for i in range(len(correct_id_list)):
      print(image_list[i],correct_id_list[i], sep=',', file=out)




#time function
def cal_time(start, end):
  time_elapsed=end-start
  return time_elapsed


#main function
def run_epoch():
  best_v_loss=float('inf')

  for epoch in range(epochs):

    start=time.time()
    

    train_loss, train_acc=train(mymodel, train_dataset, optimizer,losstype,device)
    valid_loss,valid_acc=validation(mymodel,valid_dataset,losstype,device)
    lr_scheduler.step()
    if valid_loss <best_v_loss:
      best_v_loss=valid_loss
      torch.save(mymodel.state_dict(), saved_model_path)
      print(epoch+1)

    end=time.time()
    time_elapsed=cal_time(start,end)

    print(f"time elapsed=" + str(time_elapsed)+ " | "+ str(epoch+1))
    print(f"train loss= {train_loss:.4f} and train_acc = {train_acc*100:.2f}%")
    print(f"valid loss= {valid_loss:.4f} and valid_acc = {valid_acc*100:.2f}%")


run_epoch()
testing()
