import os
os.chdir("..")
from config import *


from model.effnet import LoadEffnetv2
import torch

# "HierachicalLossNetwork"
class HLNd(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "HLNd"
        
        self.cat1_dim = 256
        self.cat2_dim = 512
        self.cat3_dim = 1024

        """utils"""
        self.activate = torch.nn.functional.relu
        # self.bn768    = torch.nn.BatchNorm1d(768) 
        # self.bn1000   = torch.nn.BatchNorm1d(1000) 
        self.dropout  = torch.nn.Dropout1d(p=0.35)
        self.softmax = torch.nn.functional.softmax
        self.argmax = torch.argmax
        

        """image classifier"""
        self.img_main = LoadEffnetv2(m_size="m", m_type="main")
        
        self.img1_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=1024, origin_input_channels=1792)
        self.img2_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=2048, origin_input_channels=1792)
        self.img3_branch = LoadEffnetv2(m_size="s", m_type="branch", 
                                        num_classes=8192, origin_input_channels=1792)
        
        """text classifier"""
        # word count embedding
        self.txt_cntvector_main = torch.nn.Linear(CFG["embedding_dim"], 2048)
        self.txt1_cntvector_branch = torch.nn.Linear(2048, self.cat1_dim)
        self.txt2_cntvector_branch = torch.nn.Linear(2048, self.cat2_dim)
        self.txt3_cntvector_branch = torch.nn.Linear(2048, self.cat3_dim)
        

        """final classifier"""
        self.linear_lvl1 = torch.nn.Linear(1024+self.cat1_dim, num_classes[0])
        self.linear_lvl2 = torch.nn.Linear(2048+self.cat2_dim, num_classes[1])
        self.linear_lvl3 = torch.nn.Linear(8192+self.cat3_dim, num_classes[2])

        self.softmax_reg1 = torch.nn.Linear(num_classes[0], num_classes[0])
        self.softmax_reg2 = torch.nn.Linear(num_classes[0]+num_classes[1], num_classes[1])
        self.softmax_reg3 = torch.nn.Linear(num_classes[0]+num_classes[1]+num_classes[2], num_classes[2])


    def forward(self, x_img, x_txt_cntvector):
        """image stage"""
        # main network (efficient net M)
        x_img       = self.img_main(x_img)
        
        # branch networks (efficient net S -> linear classifier)
        img_level_1 = self.img1_branch(x_img)
        img_level_2 = self.img2_branch(x_img)
        img_level_3 = self.img3_branch(x_img)

        """text stage (word count vector)"""        
        # main network for word count
        x_txt_cntvector  = self.txt_cntvector_main(x_txt_cntvector)
        x_txt_cntvector  = self.activate(x_txt_cntvector)
        x_txt_cntvector  = self.dropout(x_txt_cntvector)

        # branch networks 
        txt_cnt_level_1 = self.txt1_cntvector_branch(x_txt_cntvector)
        txt_cnt_level_1 = self.activate(txt_cnt_level_1)

        txt_cnt_level_2 = self.txt2_cntvector_branch(x_txt_cntvector)
        txt_cnt_level_2 = self.activate(txt_cnt_level_2)

        txt_cnt_level_3 = self.txt3_cntvector_branch(x_txt_cntvector)
        txt_cnt_level_3 = self.activate(txt_cnt_level_3)

        """final classification stage"""
        cat1 = torch.cat((img_level_1, txt_cnt_level_1), dim=1)
        cat2 = torch.cat((img_level_2, txt_cnt_level_2), dim=1)
        cat3 = torch.cat((img_level_3, txt_cnt_level_3), dim=1)
        
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
    
    