from config import *

# from effnet import *
import torch
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights
# from torchvision.models import efficientnet_v2_s as effnet_s


class Cat1Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Cat1Net"
        
        model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
        self.img_main= model
        self.img_classifier = torch.nn.Linear(1000, CFG["num_class1"])
        self.txt_classifier = torch.nn.Linear(768, CFG["num_class1"])
        
    def forward(self, img,txt):
        img = self.img_main(img)
        print(img.shape)
        img = self.img_classifier(img)
        print(img.shape)
        txt = txt.cpu()
        txt = self.txt_classifier(txt)
        print(txt.shape)
        
        return img, txt


class TinyNet4(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "TinyNet4"

        """utils"""
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

        self.image_classifier1 = torch.nn.Linear(1000, CFG["num_class1"])
        self.image_classifier2 = torch.nn.Linear(1000, CFG["num_class2"])
        self.image_classifier3 = torch.nn.Linear(1000, CFG["num_class3"])

        """text classifier"""
        self.txt_classifier1 = torch.nn.Linear(768, CFG["num_class1"])
        self.txt_classifier2 = torch.nn.Linear(768, CFG["num_class2"])
        self.txt_classifier3 = torch.nn.Linear(768, CFG["num_class3"])

        """final classifier"""
        self.fin_classifier1_1 = torch.nn.Linear(CFG["num_class1"]*2, CFG["num_class1"])
        self.fin_classifier2_1 = torch.nn.Linear(CFG["num_class2"]*2, CFG["num_class2"])
        self.fin_classifier3_1 = torch.nn.Linear(CFG["num_class3"]*2, CFG["num_class3"])

        self.fin_classifier1_2 = torch.nn.Linear(CFG["num_class1"], CFG["num_class1"])
        self.fin_classifier2_2 = torch.nn.Linear(CFG["num_class2"], CFG["num_class2"])
        self.fin_classifier3_2 = torch.nn.Linear(CFG["num_class3"], CFG["num_class3"])

        self.fin_classifier1_con1 = torch.nn.Linear(CFG["num_class1"], CFG["num_class1"])
        self.fin_classifier2_con1 = torch.nn.Linear(CFG["num_class2"], CFG["num_class2"])
        self.fin_classifier3_con1 = torch.nn.Linear(CFG["num_class3"], CFG["num_class3"])

        self.fin_classifier1_con2 = torch.nn.Linear(CFG["num_class1"], CFG["num_class1"])
        self.fin_classifier2_con2 = torch.nn.Linear(CFG["num_class2"], CFG["num_class2"])
        self.fin_classifier3_con2 = torch.nn.Linear(CFG["num_class3"], CFG["num_class3"])


    def forward(self, x_img, x_txt):
        main_features = self.image_main(x_img)

        """image stage"""

        img1 = self.image_classifier1(main_features)
        
        # transfer information 1 to 2 (masking)
        img1_softmax = self.softmax(img1, dim=1)
        img1_labels = self.argmax(img1_softmax, dim=1)
        img1_mask = self.filter_12[img1_labels]

        # transfer information 1 to 2 (filtering)
        img2 = self.image_classifier2(main_features)
        img2 = img2*img1_mask.cuda()

        # transfer information 2 to 3 (masking)
        img2_softmax = self.softmax(img2, dim=1)
        img2_labels = self.argmax(img2_softmax, dim=1)
        img2_mask = self.filter_23[img2_labels]

        # transfer information 2 to 30 (filtering)
        img3 = self.image_classifier3(main_features)
        img3 = img3*img2_mask.cuda()
        

        """text stage"""
        txt1 = self.txt_classifier1(x_txt)

        # transfer information 1 to 2 (masking)
        txt1_softmax = self.softmax(txt1, dim=1)
        txt1_labels = self.argmax(txt1_softmax, dim=1)
        txt1_mask = self.filter_12[txt1_labels]

        # transfer information 1 to 2 (filtering)
        txt2 = self.txt_classifier2(x_txt)
        txt2 = txt2*txt1_mask.cuda()

        # transfer information 2 to 3 (masking)
        txt2_softmax = self.softmax(txt2, dim=1)
        txt2_labels = self.argmax(txt2_softmax, dim=1)
        txt2_mask = self.filter_23[txt2_labels]

        # transfer information 2 to 3 (filtering)
        txt3 = self.txt_classifier3(x_txt)
        txt3 = txt3*txt2_mask.cuda()


        """final classification stage"""
        cat1 = torch.hstack([img1, txt1]).cuda()
        cat1 = self.fin_classifier1_1(cat1)
        cat1 = self.fin_classifier1_2(cat1)
        resid1 = cat1.clone()
        cat1 = self.fin_classifier1_con1(cat1)
        cat1 = self.fin_classifier1_con2(cat1)
        cat1 = resid1 + cat1


        cat2 = torch.hstack([img2, txt2]).cuda()
        cat2 = self.fin_classifier2_1(cat2)
        cat2 = self.fin_classifier2_2(cat2)
        resid2 = cat2.clone()
        cat2 = self.fin_classifier2_con1(cat2)
        cat2 = self.fin_classifier2_con2(cat2)
        cat2 = resid2 + cat2


        cat3 = torch.hstack([img3, txt3]).cuda()
        cat3 = self.fin_classifier3_1(cat3)
        cat3 = self.fin_classifier3_2(cat3)
        resid3 = cat3.clone()
        cat3 = self.fin_classifier3_con1(cat3)
        cat3 = self.fin_classifier3_con2(cat3)
        cat3 = resid3 + cat3


        return cat1, cat2, cat3
    
    
    
