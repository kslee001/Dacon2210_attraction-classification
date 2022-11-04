import os
os.chdir("..")
from config import *

from model.effnet import LoadEffnetv2
import torch


class BasicConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device="cuda" if torch.cuda.is_available() else "cpu", **kwargs):
        super(BasicConv2d, self).__init__()
        self.device = device
        self.conv = torch.nn.Conv2d(in_channels, out_channels, bias=False, **kwargs, device=self.device)
        self.bn   = torch.nn.BatchNorm2d(out_channels, eps=0.001, device=self.device)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return torch.nn.functional.relu(x)


class InceptionResidual(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv_a = BasicConv2d(input_dim, 96, kernel_size=1, padding="same")
        self.conv_b1 = BasicConv2d(input_dim, 64, kernel_size=1, padding="same")
        self.conv_b2 = BasicConv2d(64, 80, kernel_size=[1,5], padding="same")
        self.conv_b3 = BasicConv2d(80, 96, kernel_size=[5,1], padding="same")
        self.conv_fin = BasicConv2d(96+96, input_dim, kernel_size=1, padding="same")
        
    def forward(self, x):
        resid = x
        a = self.conv_a(x)
        b = self.conv_b1(x)
        b = self.conv_b2(b)
        b = self.conv_b3(b)
        x = torch.cat((a,b), 1)
        x = self.conv_fin(x)
        x = resid + x
        x = torch.nn.functional.relu(x)
        return x  


class InceptionReduction(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.pool = torch.nn.AvgPool2d(kernel_size=3)
        self.flatten = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(p=0.35)

        self.maxpool_a = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_b1 = BasicConv2d(input_dim, 128, kernel_size=1, padding="same")
        self.conv_b2 = BasicConv2d(128, 192, kernel_size=3, stride=2)

        self.conv_c1 = BasicConv2d(input_dim, 128, kernel_size=1, padding="same")
        self.conv_c2 = BasicConv2d(128, 144, kernel_size=3, stride=2)

        self.conv_d1 = BasicConv2d(input_dim, 128, kernel_size=1, padding="same")
        self.conv_d2 = BasicConv2d(128, 144, kernel_size=3, padding="same")
        self.conv_d3 = BasicConv2d(144, 160, kernel_size=3, stride=2)
        
        self.classifier = torch.nn.Linear(2288, output_dim)

    def forward(self, x):
        a = self.maxpool_a(x)
        
        b = self.conv_b1(x)
        b = self.conv_b2(b)

        c = self.conv_c1(x)
        c = self.conv_c2(c)

        d = self.conv_d1(x)
        d = self.conv_d2(d)
        d = self.conv_d3(d)

        x = torch.cat((a,b,c,d), 1)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x
    

# input : 1792 * 10 * 10
class BranchNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.name = "brnachnet"
        
        """layers"""
        self.residcell  = InceptionResidual(input_dim)
        self.classifier = InceptionReduction(input_dim, output_dim)   
            
    def forward(self, x):
        x = self.residcell(x)
        x = self.residcell(x)
        x = self.residcell(x)
        x = self.residcell(x)

        x = self.classifier(x)
        return x    



class TN(torch.nn.Module):
    def __init__(self, num_classes=[6,18,128]):
        super().__init__()
        self.name = "TN"
        
        self.cat1_dim = 256
        self.cat2_dim = 512
        self.cat3_dim = 1024

        """utils"""
        self.activate = torch.nn.functional.relu
        self.dropout  = torch.nn.Dropout1d(p=0.35)
        self.softmax = torch.nn.functional.softmax
        self.argmax = torch.argmax
        

        """image classifier"""
        self.img_main = LoadEffnetv2(m_size="m", m_type="main")
        
        self.img1_branch = BranchNet(1792, 1024)
        self.img2_branch = BranchNet(1792, 2048)
        self.img3_branch = BranchNet(1792, 8192)
        
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
