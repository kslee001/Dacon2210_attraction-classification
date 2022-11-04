import os
os.chdir("..")
from config import *

from model.effnet import LoadEffnetv2
import torch

class HierachicalLossNetwork2(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "HierachicalLossNetwork2"

        """utils"""
        self.activate = torch.nn.functional.relu
        # self.bn768    = torch.nn.BatchNorm1d(768) 
        # self.bn1000   = torch.nn.BatchNorm1d(1000) 
        self.dropout  = torch.nn.Dropout1d(p=0.35)
        self.softmax = torch.nn.functional.softmax
        self.argmax = torch.argmax

        """hierachical filters"""
        self.filter_12 = torch.zeros(CFG["num_class1"], CFG["num_class2"])
        for i in range(len(CFG["subcat_for1"])):
            for j in CFG["subcat_for1"][i]:
                self.filter_12[i][j] = 1 # mask for cat1 -> cat2

        self.filter_23 = torch.zeros(CFG["num_class2"], CFG["num_class3"])
        for i in range(len(CFG["subcat_for2"])):
            for j in CFG["subcat_for2"][i]:
                self.filter_23[i][j] = 1 # mask for cat1 -> cat2


        """image classifier"""
        self.img_main = LoadEffnetv2(m_size="m", m_type="main")
        
        self.img1_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=128, origin_input_channels=1792)
        self.img2_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=256, origin_input_channels=1792)
        self.img3_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=512, origin_input_channels=1792)
        
        self.linear_lvl1_img = torch.nn.Linear(128, num_classes[0])
        self.linear_lvl2_img = torch.nn.Linear(256, num_classes[1])
        self.linear_lvl3_img = torch.nn.Linear(512, num_classes[2])

        """text classifier"""
        self.txt_main = torch.nn.Linear(768, 768)
        self.txt1_branch = torch.nn.Linear(768, 128)
        self.txt2_branch = torch.nn.Linear(768, 256)
        self.txt3_branch = torch.nn.Linear(768, 512)

        self.linear_lvl1_txt = torch.nn.Linear(128, num_classes[0])
        self.linear_lvl2_txt = torch.nn.Linear(256, num_classes[1])
        self.linear_lvl3_txt = torch.nn.Linear(512, num_classes[2])

        """final classifier"""

        self.softmax_reg1 = torch.nn.Linear(num_classes[0]*2, num_classes[0])
        self.softmax_reg2 = torch.nn.Linear(num_classes[0]+num_classes[1]*2, num_classes[1])
        self.softmax_reg3 = torch.nn.Linear(num_classes[0]+num_classes[1]+num_classes[2]*2, num_classes[2])


    def forward(self, x_img, x_txt):
        """image stage"""
        # main network (efficient net M)
        x_img       = self.img_main(x_img)
        
        # branch networks (efficient net S -> linear classifier)
        img_level_1 = self.img1_branch(x_img)
        img_level_2 = self.img2_branch(x_img)
        img_level_3 = self.img3_branch(x_img)
        
        img_level_1 = self.linear_lvl1_img(img_level_1)
        img_level_2 = self.linear_lvl2_img(img_level_2)
        img_level_3 = self.linear_lvl3_img(img_level_3)

        """text stage"""
        # main network (Kobert monolog -> linear classifier)
        x_txt       = self.txt_main(x_txt)
        x_txt       = self.activate(x_txt)
        x_txt       = self.dropout(x_txt)
        
        # branch networks 
        txt_level_1 = self.txt1_branch(x_txt)
        txt_level_1 = self.activate(txt_level_1)

        txt_level_2 = self.txt2_branch(x_txt)
        txt_level_2 = self.activate(txt_level_2)

        txt_level_3 = self.txt3_branch(x_txt)
        txt_level_3 = self.activate(txt_level_3)
        
        txt_level_1 = self.linear_lvl1_img(txt_level_1)
        txt_level_2 = self.linear_lvl2_img(txt_level_2)
        txt_level_3 = self.linear_lvl3_img(txt_level_3)

        """final classification stage"""
        cat1 = torch.cat((img_level_1, txt_level_1), dim=1)
        cat2 = torch.cat((img_level_2, txt_level_2), dim=1)
        cat3 = torch.cat((img_level_3, txt_level_3), dim=1)
        
        cat1 = self.softmax_reg1(cat1)
        cat2 = self.softmax_reg2(torch.cat((cat1, cat2), dim=1))
        cat3 = self.softmax_reg3(torch.cat((cat1, cat2, cat3), dim=1))
        
        return cat1, cat2, cat3
    
    
    
class HLNCat1(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "HLNCat1"

        """utils"""
        self.activate = torch.nn.functional.relu
        # self.bn768    = torch.nn.BatchNorm1d(768) 
        # self.bn1000   = torch.nn.BatchNorm1d(1000) 
        self.dropout  = torch.nn.Dropout1d(p=0.35)
        self.softmax = torch.nn.functional.softmax
        self.argmax = torch.argmax
        
        self.image_weight = 8   
        self.text_weight  = 4

        """image classifier"""
        self.img_branch = LoadEffnetv2(m_size="m", m_type="branch", 
                                        num_classes=num_classes[0]*self.image_weight, origin_input_channels=3)
        
        """text classifier"""
        self.txt_main = torch.nn.Linear(768, num_classes[0]*self.text_weight)

        """final classifier"""
        self.fin_classifier1 = torch.nn.Linear(
            num_classes[0]*(self.image_weight + self.text_weight), 
            num_classes[0]*(self.image_weight + self.text_weight)//2
        )
        self.fin_classifier2 = torch.nn.Linear(
            num_classes[0]*(self.image_weight + self.text_weight)//2, 
            num_classes[0]
        )

    def forward(self, x_img, x_txt):
        """image stage"""
        # branch networks (efficient net S -> linear classifier)
        x_img = self.img_branch(x_img)
        
        """text stage"""
        # main network (Kobert monolog -> linear classifier)
        x_txt = self.txt_main(x_txt)

        """final classification stage"""
        cat1 = torch.cat((x_img, x_txt), dim=1)
        cat1 = self.fin_classifier1(cat1)
        cat1 = self.activate(cat1)
        cat1 = self.dropout(cat1)
        cat1 = self.fin_classifier2(cat1)

        return cat1
