import os
os.chdir("..")
from config import *

from model.effnet import LoadEffnetv2
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights


from transformers import AutoModel, MobileViTFeatureExtractor, MobileViTForImageClassification

import torch


class MR918aug(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "MR918aug"
        
        """Utils"""
        self.activate = torch.nn.functional.silu
        self.bn = torch.nn.BatchNorm1d
        
        self.img_dim = 1000
        self.txt_dim = 1024

        """image classifier"""
        # self.img_main = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights)
        # self.feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-small")
        self.img_main = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
        self.img_main_cfr = torch.nn.Linear(self.img_dim, self.img_dim)
        """text classifier"""
        self.txt_main = AutoModel.from_pretrained("klue/roberta-large")
        self.txt_main_cfr = torch.nn.Linear(self.txt_dim, self.txt_dim)
  
        
        """final classifier"""
        self.cat1cfr = torch.nn.Sequential(
            torch.nn.Linear(self.img_dim + self.txt_dim, 512),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.22),
            torch.nn.SiLU(),
            
            torch.nn.Linear(512, 256),
            torch.nn.LayerNorm(256),
            torch.nn.Dropout(0.22),
            torch.nn.SiLU(),
        )
        self.softmax_reg1 = torch.nn.Linear(256, num_classes[0])
        
        
        self.cat2cfr = torch.nn.Sequential(
            torch.nn.Linear(self.img_dim + self.txt_dim, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.Dropout(0.22),
            torch.nn.SiLU(),
            
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.Dropout(0.22),
            torch.nn.SiLU(),
        )
        self.softmax_reg2 = torch.nn.Linear(num_classes[0] + 512, num_classes[1])
        
        
        self.cat3cfr = torch.nn.Sequential(
            torch.nn.Linear(self.img_dim + self.txt_dim, 1440),
            torch.nn.LayerNorm(1440),
            torch.nn.Dropout(0.22),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(1440),
            
            torch.nn.Linear(1440, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.Dropout(0.22),
            torch.nn.SiLU(),
            # torch.nn.BatchNorm1d(1024),
            
            torch.nn.Linear(1024, 768),
            torch.nn.LayerNorm(768),
            torch.nn.Dropout(0.22),
            torch.nn.SiLU(),
        )
        self.softmax_reg3 = torch.nn.Linear(num_classes[0]+num_classes[1]+768, num_classes[2])
        

    def forward(self, img, txt_input_ids, txt_attention_mask):
        batch_size = txt_input_ids.shape[0]
        img = self.img_main(img).logits
        img = self.activate(img)
        img = self.img_main_cfr(img)
        # txt = self.txt_main(txt_input_ids, txt_attention_mask).last_hidden_state.reshape(batch_size, 256 * 768)      # B, 256, 768 -> B, 256 * 768
        txt = self.txt_main(txt_input_ids, txt_attention_mask).last_hidden_state[:, 0, :]    # B, 256, 768 -> B, 256 * 768
        txt = self.activate(txt)
        txt = self.txt_main_cfr(txt)
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