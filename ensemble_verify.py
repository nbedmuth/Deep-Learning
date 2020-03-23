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
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#constansts
num_classes=2300
Batch_size=256
epochs=10
lr=0.01
gamma=0.9
test_textfile='./test_trials_verification_student.txt'
test_folder='./test_verification/'
valid_folder='./validation_verification/'
learned_model1=Basemodule()
learned_model1=learned_model1.to(device)

learned_model2=Basemodule()
learned_model2=learned_model2.to(device)

learned_model3=Basemodule()
learned_model3=learned_model3.to(device)

learned_model4=Basemodule()
learned_model4=learned_model4.to(device)

#learned_model1.load_state_dict(torch.load('/home/ubuntu/hw2/11-785hw2p2-s20/clf/saved_model1.pt'))
learned_model2.load_state_dict(torch.load('/home/ubuntu/hw2/11-785hw2p2-s20/clf/saved_model2.pt'))
learned_model3.load_state_dict(torch.load('/home/ubuntu/hw2/11-785hw2p2-s20/clf/saved_model3.pt'))
learned_model4.load_state_dict(torch.load('/home/ubuntu/hw2/11-785hw2p2-s20/clf/saved_model4.pt'))

valid_textfile='./validation_trials_verification.txt'
output='./verify.csv'
# do we need a . before home?


#data transform
data_transform=transforms.Compose([transforms.ToTensor()])



class Makedataset(Dataset): #what is Dataset here?
  def __init__(self,image_list1,image_list2, folder):
    self.image_list1=image_list1
    self.image_list2=image_list2
    self.folder=folder

  def __len__(self):
    return(len(self.image_list1))
  
  def __getitem__(self,index):
    x1=Image.open(self.folder+self.image_list1[index])
    x2=Image.open(self.folder+self.image_list2[index])
    x1=transforms.ToTensor()(x1)
    x2=transforms.ToTensor()(x2)
    return x1, x2

#making of valid dataset
with open(valid_textfile) as f:
  v_image_list=[num.rstrip('\n').split(' ') for num in f]

v_image1_list=[el[0] for el in v_image_list]
v_image2_list=[el[1] for el in v_image_list]
v_truelabel=[el[2] for el in v_image_list]

valid_in=Makedataset(v_image1_list,v_image2_list, valid_folder)


#valid dataloader
valid_dataset = torch.utils.data.DataLoader(valid_in,
                                             batch_size=Batch_size,
                                             shuffle=False,
                                             num_workers=1)

#making of test sets
with open(test_textfile) as f:
  t_image_list=[num.rstrip('\n').split(' ') for num in f]

t_image1_list=[el[0] for el in t_image_list]
t_image2_list=[el[1] for el in t_image_list]

test_in=Makedataset(t_image1_list,t_image2_list,test_folder)

#valid dataloader
test_dataset = torch.utils.data.DataLoader(test_in,
                                             batch_size=Batch_size,
                                             shuffle=False,
                                             num_workers=1)

print("test and valid data loader are ready \n")

def cosine(embed1,embed2):
  cos=nn.CosineSimilarity(dim=1, eps=1e-06)
  
  simi=cos(embed1,embed2)
  return simi

#classification function
def verify(model2,model3,model4, dataset, device):
  
  simi_labels=[]
  model2.eval()
  model3.eval()
  model4.eval()
  with torch.no_grad(): #what is this?
    count=0
    for (x1,x2) in dataset: #test_dataset
      
      x1=x1.to(device)
      x2=x2.to(device)

      embed1_model2=model2(x1)
      embed2_model2=model2(x2)

      embed1_model3=model3(x1)
      embed2_model3=model3(x2)

      embed1_model4=model4(x1)
      embed2_model4=model4(x2)
      embed1_list=[embed1_model2,embed1_model3,embed1_model4]
      embed2_list=[embed2_model2,embed2_model3,embed2_model4]

      embed1=torch.mean(torch.stack(embed1_list),dim=0)
      embed2=torch.mean(torch.stack(embed2_list),dim=0)

      simi=cosine(embed1,embed2)
      #print(simi.shape)
      simi=simi.view(-1)
      #print(simi.shape)
      count+=1
      simi_labels.extend(simi.cpu().numpy())

    print(count)
    
    
    return np.array(simi_labels)

#simi_labels=verify(learned_model, test_dataset, device)


#final testing
def testing():
  simi_labels=verify(learned_model2, learned_model3,learned_model4,test_dataset, device)
  print(len(simi_labels))
  with open(output, 'w') as out:
    print('trial', 'score', sep=',', file=out)
    for i in range(len(t_image_list)):
      print(str(t_image1_list[i]) + " " + str(t_image2_list[i]), simi_labels[i], sep=',', file=out)


testing()
