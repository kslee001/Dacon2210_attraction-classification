from config import *

# from effnet import *
import torch
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
# from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


class TinyHierachicalLossNetwork(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "TinyHierachicalLossNetwork"

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
        self.image_main = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.linear_lvl1_img = torch.nn.Linear(1000, num_classes[0])
        self.linear_lvl2_img = torch.nn.Linear(1000, num_classes[1])
        self.linear_lvl3_img = torch.nn.Linear(1000, num_classes[2])

        """text classifier"""
        self.txt_classifier1 = torch.nn.Linear(768, 768)
        self.txt_classifier2 = torch.nn.Linear(768, 1000)

        self.linear_lvl1_txt = torch.nn.Linear(1000, num_classes[0])
        self.linear_lvl2_txt = torch.nn.Linear(1000, num_classes[1])
        self.linear_lvl3_txt = torch.nn.Linear(1000, num_classes[2])

        """final classifier"""

        self.softmax_reg1 = torch.nn.Linear(num_classes[0], num_classes[0])
        self.softmax_reg2 = torch.nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
        self.softmax_reg3 = torch.nn.Linear(num_classes[0]+num_classes[1]+num_classes[2], num_classes[2])


    def forward(self, x_img, x_txt):
        """image stage"""
        x_img       = self.image_main(x_img)

        img_level_1 = self.linear_lvl1_img(x_img)
        img_level_2 = self.linear_lvl2_img(x_img)
        img_level_3 = self.linear_lvl3_img(x_img)

        """text stage"""
        x_txt       = self.txt_classifier1(x_txt)
        # x_txt       = self.bn768(x_txt)
        x_txt       = self.activate(x_txt)
        x_txt       = self.dropout(x_txt)
        
        x_txt       = self.txt_classifier2(x_txt)
        # x_txt       = self.bn1000(x_txt)
        x_txt       = self.activate(x_txt)
        x_txt       = self.dropout(x_txt)
        
        txt_level_1 = self.linear_lvl1_img(x_txt)
        txt_level_2 = self.linear_lvl2_img(x_txt)
        txt_level_3 = self.linear_lvl3_img(x_txt)

        """final classification stage"""
        
        cat1 = img_level_1 + txt_level_1
        cat2 = img_level_2 + txt_level_2
        cat3 = img_level_3 + txt_level_3

        cat1 = self.softmax_reg1(cat1)
        cat2 = self.softmax_reg2(torch.cat((cat1, cat2), dim=1))
        cat3 = self.softmax_reg3(torch.cat((cat1, cat2, cat3), dim=1))
        
        return cat1, cat2, cat3
    
    
    








class TNBranchC1(torch.nn.Module):  # for category 1
    def __init__(self):
        super().__init__()
        self.name = "TNBranchC1"
        
        """utils"""
        self.dropout  = torch.nn.Dropout(0.35)
        self.activate = torch.nn.functional.relu
        self.bn1024   = torch.nn.BatchNorm1d(1024, eps=0.001)
        self.bn768    = torch.nn.BatchNorm1d(768, eps=0.001)
        self.bn512    = torch.nn.BatchNorm1d(512, eps=0.001)
        self.bn256    = torch.nn.BatchNorm1d(256, eps=0.001)
        
        """image classifier"""
        self.img_mainnet = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        
        self.img_classifier1 = torch.nn.Linear(1000, 768)
        self.img_classifier2 = torch.nn.Linear(768, 768)
        self.img_classifier3 = torch.nn.Linear(768, 768)
        self.img_classifier4 = torch.nn.Linear(768, 512)

        """text classifier"""
        self.txt_classifier1 = torch.nn.Linear(768, 768)
        self.txt_classifier2 = torch.nn.Linear(768, 768)
        self.txt_classifier3 = torch.nn.Linear(768, 768)
        self.txt_classifier4 = torch.nn.Linear(768, 512)

        """final classifier"""
        self.fin_classifier1 = torch.nn.Linear(512*2, 768)
        self.fin_classifier2 = torch.nn.Linear(768, 512)
        self.fin_classifier3 = torch.nn.Linear(512, 256)
        self.fin_classifier4 = torch.nn.Linear(256, CFG["num_class1"])

    # Batchnorm -> Activate -> Dropout
    def BAD(self, x, batchnorm, do_batchnorm=False):
        if do_batchnorm:
            x = batchnorm(x)
            x = self.activate(x, inplace=True)
            x = self.dropout(x)            
        else:
            x = self.activate(x, inplace=True)
            x = self.dropout(x)
        return x
    

    def forward(self, x_img, x_txt):
        """image stage"""
        img1 = self.img_mainnet(x_img)
        
        img1 = self.img_classifier1(img1)
        img1 = self.BAD(img1, batchnorm=self.bn768)
        
        img1 = self.img_classifier2(img1)
        img1 = self.BAD(img1, batchnorm=self.bn768)
        
        img1 = self.img_classifier3(img1)
        img1 = self.BAD(img1, batchnorm=self.bn768)
        
        img1 = self.img_classifier4(img1)
        img1 = self.BAD(img1, batchnorm=self.bn512)


        """text stage"""
        txt1 = self.txt_classifier1(x_txt)
        txt1 = self.BAD(txt1, batchnorm=self.bn768)
        
        txt1 = self.txt_classifier2(txt1)
        txt1 = self.BAD(txt1, batchnorm=self.bn768)
        
        txt1 = self.txt_classifier3(txt1)
        txt1 = self.BAD(txt1, batchnorm=self.bn768)
        
        txt1 = self.txt_classifier4(txt1)
        txt1 = self.BAD(txt1, batchnorm=self.bn512)
        

        """final classification stage"""
        cat = torch.hstack([img1, txt1]).cuda()
        
        cat = self.fin_classifier1(cat)
        cat = self.BAD(cat, batchnorm=self.bn768)

        cat = self.fin_classifier2(cat)
        cat = self.BAD(cat, batchnorm=self.bn512)

        cat = self.fin_classifier3(cat)
        cat = self.BAD(cat, batchnorm=self.bn256)

        cat = self.fin_classifier4(cat)

        return cat
    
    
    
