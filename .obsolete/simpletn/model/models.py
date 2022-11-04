import os
os.chdir("..")
from config import *

from model.effnet import LoadEffnetv2
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

import torch


class SimpleTN(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "SimpleTN"
        
        """Utils"""
        self.activate = torch.nn.SiLU
        self.bn = torch.nn.BatchNorm1d

        """image classifier"""
        self.img_main = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)
        
        """text classifier"""
        self.txt_main = torch.nn.Sequential(
            torch.nn.Linear(768, 2048),
            self.activate(),
            torch.nn.Linear(2048, 768),
            self.activate(),
            torch.nn.Linear(768,2048),
            self.activate(),
            torch.nn.Linear(2048,768)
        )
        
        """final classifier"""
        self.cat1cfr = torch.nn.Sequential(
            torch.nn.Linear(1000+768, 512),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(512),
            
            torch.nn.Linear(512, 256),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(256),
            
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(128),
        )
        self.softmax_reg1 = torch.nn.Linear(128, num_classes[0])
        
        
        self.cat2cfr = torch.nn.Sequential(
            torch.nn.Linear(1000+768, 1024),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(1024),
            
            torch.nn.Linear(1024, 512),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(512),
            
            torch.nn.Linear(512, 256),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(256),
        )
        self.softmax_reg2 = torch.nn.Linear(num_classes[0] + 256, num_classes[1])
        
        
        self.cat3cfr = torch.nn.Sequential(
            torch.nn.Linear(1000+768, 1440),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(1440),
            
            torch.nn.Linear(1440, 1024),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(1024),
            
            torch.nn.Linear(1024, 768),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(768),
            
            torch.nn.Linear(768, 512),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(512),
        )
        self.softmax_reg3 = torch.nn.Linear(num_classes[0]+num_classes[1]+512, num_classes[2])
        

    def forward(self, img, txt):
        img = self.img_main(img) 
        txt = self.txt_main(txt)
        concatted = torch.cat((img, txt), dim=1)
    
        cat1 = self.cat1cfr(concatted)
        cat1 = self.softmax_reg1(cat1)        
        
        cat2 = self.cat2cfr(concatted)
        cat2 = torch.cat((cat1, cat2), dim=1)
        cat2 = self.softmax_reg2(cat2)
        
        
        cat3 = self.cat3cfr(concatted)
        cat3 = torch.cat((cat1, cat2, cat3), dim=1)
        cat3 = self.softmax_reg3(cat3)
        
        return cat1, cat2, cat3