import os
os.chdir("..")
from config_continue1 import *

from model.effnet import LoadEffnetv2
import torch

# "HierachicalLossNetwork"
class HLNc(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "HLN_c"

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
        self.txt_main = torch.nn.Linear(CFG["embedding_dim"], 2048)
        self.txt1_branch = torch.nn.Linear(2048, 128)
        self.txt2_branch = torch.nn.Linear(2048, 256)
        self.txt3_branch = torch.nn.Linear(2048, 512)

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
    
    

class HLNc2(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "HLNc2"

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
                                        num_classes=512*4, origin_input_channels=1792)
        self.img2_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=1024*4, origin_input_channels=1792)
        self.img3_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=4096*4, origin_input_channels=1792)
        
        """text classifier"""
        self.txt_main = torch.nn.Linear(CFG["embedding_dim"], 2048)
        self.txt1_branch = torch.nn.Linear(2048, 512)
        self.txt2_branch = torch.nn.Linear(2048, 1024)
        self.txt3_branch = torch.nn.Linear(2048, 4096)

        """final classifier"""
        self.linear_lvl1 = torch.nn.Linear(512*5, num_classes[0])
        self.linear_lvl2 = torch.nn.Linear(1024*5, num_classes[1])
        self.linear_lvl3 = torch.nn.Linear(4096*5, num_classes[2])

        self.softmax_reg1 = torch.nn.Linear(num_classes[0], num_classes[0])
        self.softmax_reg2 = torch.nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
        self.softmax_reg3 = torch.nn.Linear(num_classes[0]+num_classes[1]+num_classes[2], num_classes[2])


    def forward(self, x_img, x_txt):
        """image stage"""
        # main network (efficient net M)
        x_img       = self.img_main(x_img)
        
        # branch networks (efficient net S -> linear classifier)
        img_level_1 = self.img1_branch(x_img)
        img_level_2 = self.img2_branch(x_img)
        img_level_3 = self.img3_branch(x_img)

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
        

        """final classification stage"""
        cat1 = torch.cat((img_level_1, txt_level_1), dim=1)
        cat2 = torch.cat((img_level_2, txt_level_2), dim=1)
        cat3 = torch.cat((img_level_3, txt_level_3), dim=1)
        
        cat1 = self.linear_lvl1(cat1)
        cat2 = self.linear_lvl2(cat2)
        cat3 = self.linear_lvl3(cat3)
        
        cat1 = self.softmax_reg1(cat1)
        cat2 = self.softmax_reg2(torch.cat((cat1, cat2), dim=1))
        cat3 = self.softmax_reg3(torch.cat((cat1, cat2, cat3), dim=1))
        
        return cat1, cat2, cat3



class CustomModel(torch.nn.Module):
    def __init__(self, num_classes=CFG["num_class3"]):
        super(CustomModel, self).__init__()
        # Image
        self.cnn_extract = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Text
        self.nlp_extract = torch.nn.Sequential(
            torch.nn.Linear(4096, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
        )
        # Classifier
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(21760, num_classes)
        )
            

    def forward(self, img, text):
        img_feature = self.cnn_extract(img)
        img_feature = torch.flatten(img_feature, start_dim=1)
        text_feature = self.nlp_extract(text)
        feature = torch.cat([img_feature, text_feature], axis=1)
        output = self.classifier(feature)
        return output
    
    