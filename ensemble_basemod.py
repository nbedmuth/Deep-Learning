import sys
from torch import nn

class Basemodule(nn.Module):
 	def __init__(self):
 		super().__init__()

 		self.feature=nn.Sequential(
 			nn.Conv2d(3, 64, 5, 1, 2),
    		nn.ReLU(inplace=True),
			nn.BatchNorm2d(num_features=64, eps= 1e-05,affine=True, track_running_stats=True),
			nn.MaxPool2d(kernel_size=3,stride=2, padding=1),

			nn.Conv2d(64, 192, 5, 1, 2),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(num_features=192, eps= 1e-05,affine=True, track_running_stats=True),
			nn.MaxPool2d(kernel_size=3,stride=2, padding=1),

			
			nn.Conv2d(192, 384, 3, 1, 1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(num_features=384, eps= 1e-05,affine=True, track_running_stats=True),

			#nn.Dropout(p=0.5),
			nn.Conv2d(384, 256, 3, 1, 1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(num_features=256, eps= 1e-05, affine=True, track_running_stats=True),

			nn.Conv2d(256, 256, 3, 1, 1),
			nn.ReLU(inplace=True),
			nn.BatchNorm2d(num_features=256, eps= 1e-05,affine=True, track_running_stats=True),
			nn.MaxPool2d(kernel_size=3,stride=2, padding=1),
			)
 		self.verification=nn.Sequential(
 			#nn.Dropout(p=0.5),
    		nn.Linear(4096, 4096),
    		nn.ReLU(inplace=True),
    		nn.BatchNorm1d(num_features=4096, eps= 1e-05, affine=True, track_running_stats=True),
    		#nn.Dropout(p=0.5),
    		nn.Linear(4096, 4096),
    		nn.ReLU(inplace=True),
    		nn.BatchNorm1d(num_features=4096, eps= 1e-05, affine=True, track_running_stats=True),
    		)
 		self.classification=nn.Sequential(
	    	nn.Linear(4096,2300),
	    	)
 	def verif(self,x):
 		x=self.feature(x)
 		x=x.view(x.shape[0], -1)
 		x=self.verification(x)
 		return(x)

 	def forward(self, x):
 		x = self.feature(x)
 		x=x.view(x.shape[0], -1)
 		x=self.verification(x)
 		x=self.classification(x)
 		return x