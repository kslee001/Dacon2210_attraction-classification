"""
Creates a EfficientNetV2 Model as defined in:
Mingxing Tan, Quoc V. Le. (2021). 
EfficientNetV2: Smaller Models and Faster Training
arXiv preprint arXiv:2104.00298.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch
import torch.nn as nn
import math
import re

# private
from config import *


# text models

# txt_model = BertModel.from_pretrained('skt/kobert-base-v1').to(CFG["device"])
# tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})




# __all__ = ['effnetv2_s', 'effnetv2_m', 'effnetv2_l', 'effnetv2_xl']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# SiLU (Swish) activation function
if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

 
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        SiLU()
    )


class MBConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_se):
        super(MBConv, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if use_se:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                SELayer(inp, hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # fused
                nn.Conv2d(inp, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class EffNetV2(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


class EffNetV2_main(nn.Module):
    def __init__(self, cfgs, num_classes=1000, width_mult=1.):
        super(EffNetV2_main, self).__init__()
        self.cfgs = cfgs

        # building first layer
        input_channel = _make_divisible(24 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = MBConv
        for t, c, n, s, use_se in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, use_se))
                input_channel = output_channel

        self.features = nn.Sequential(*layers)

        # building last several layers
        output_channel = _make_divisible(1792 * width_mult, 8) if width_mult > 1.0 else 1792
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.001)
                m.bias.data.zero_()


def effnetv2_s(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_m(**kwargs):
    """
    Constructs a EfficientNetV2-M model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  3, 1, 0],
        [4,  48,  5, 2, 0],
        [4,  80,  5, 2, 0],
        [4, 160,  7, 2, 1],
        [6, 176, 14, 1, 1],
        [6, 304, 18, 2, 1],
        [6, 512,  5, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_l(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)


def effnetv2_xl(**kwargs):
    """
    Constructs a EfficientNetV2-XL model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  8, 2, 0],
        [4,  96,  8, 2, 0],
        [4, 192, 16, 2, 1],
        [6, 256, 24, 1, 1],
        [6, 512, 32, 2, 1],
        [6, 640,  8, 1, 1],
    ]
    return EffNetV2(cfgs, **kwargs)





def effnetv2_s_cat1(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, num_classes=CFG["num_class1"],**kwargs)

def effnetv2_s_cat2(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, num_classes=CFG["num_class2"],**kwargs)

def effnetv2_s_cat3(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2(cfgs, num_classes=CFG["num_class3"],**kwargs)



def effnetv2_s_main(**kwargs):
    """
    Constructs a EfficientNetV2-S model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  24,  2, 1, 0],
        [4,  48,  4, 2, 0],
        [4,  64,  4, 2, 0],
        [4, 128,  6, 2, 1],
        [6, 160,  9, 1, 1],
        [6, 256, 15, 2, 1],
    ]
    return EffNetV2_main(cfgs, **kwargs)


def effnetv2_l_main(**kwargs):
    """
    Constructs a EfficientNetV2-L model
    """
    cfgs = [
        # t, c, n, s, SE
        [1,  32,  4, 1, 0],
        [4,  64,  7, 2, 0],
        [4,  96,  7, 2, 0],
        [4, 192, 10, 2, 1],
        [6, 224, 19, 1, 1],
        [6, 384, 25, 2, 1],
        [6, 640,  7, 1, 1],
    ]
    return EffNetV2_main(cfgs, **kwargs)


class TinyMultiNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "TinyMultiNet"


        """utils"""
        self.activate = torch.nn.ReLU(inplace=True)
        self.flatten  = torch.nn.Flatten()
        self.normalizer = torch.nn.functional.normalize
        self.softmax = torch.nn.functional.softmax
        self.dropout = torch.nn.Dropout(0.35)

        """for images"""
        self.tiny_conv = [
            torch.nn.Sequential(
                torch.nn.Conv2d(1792, 1024, 3, padding="same", bias=False),
                # torch.nn.BatchNorm2d(1024),
                self.activate,
                self.dropout,

                torch.nn.Conv2d(1024, 512, 3, padding="same", bias=False),
                # torch.nn.BatchNorm2d(512),
                self.activate,
                self.dropout,

                torch.nn.Conv2d(512, 256, 3, padding="same", bias=False),
                # torch.nn.BatchNorm2d(256),
                self.activate,      
                self.dropout,

                # torch.nn.Conv2d(256, 128, 3, padding="same", bias=False),
                # torch.nn.BatchNorm2d(128),
                # self.activate,      
                # self.dropout,

                self.flatten
            ) for _ in range(3)
        ]
        self.image_main   = effnetv2_l_main()
        self.branch1 = self.tiny_conv[0]
        self.branch2 = self.tiny_conv[1]
        self.branch3 = self.tiny_conv[2]

        self.image_classifier1 = torch.nn.Linear(256*10*10, CFG["num_class1"])
        self.image_classifier2 = torch.nn.Linear(256*10*10, CFG["num_class2"])
        self.image_classifier3 = torch.nn.Linear(256*10*10, CFG["num_class3"])

        """for texts"""
        self.txt_classifier1 = torch.nn.Linear(768, CFG["num_class1"])
        self.txt_classifier2 = torch.nn.Linear(768, CFG["num_class2"])
        self.txt_classifier3 = torch.nn.Linear(768, CFG["num_class3"])

        """final classification"""
        self.fin_classifier1 = torch.nn.Linear(CFG["num_class1"]*2, CFG["num_class1"])
        self.fin_classifier2 = torch.nn.Linear(CFG["num_class2"]*2, CFG["num_class2"])
        self.fin_classifier3 = torch.nn.Linear(CFG["num_class3"]*2, CFG["num_class3"])


    def forward(self, image, txt):
        
        """Image stage"""
        # main network : 1792 features (1792,10,10)
        main_features = self.image_main(image)

        # branche networks (1, 2, 3)
        img_cat1 = self.branch1(main_features) 
        img_cat2 = self.branch2(main_features) 
        img_cat3 = self.branch3(main_features) 
 
        # predict cat1 & generate mask1                    # (N, 6)
        img_cat1 = self.image_classifier1(img_cat1)        
        img_mask1 = self.normalizer(img_cat1, p=1.0, dim=1, eps=1e-6)

        # predict cat2 considering mask1 & generate mask2  # (N, 18)
        img_cat2 = self.image_classifier2(img_cat2)     
        img_norm2 = img_cat2.norm().item()   
        for i in CFG["subcat_for1"]:
            img_cat2[ :, CFG["subcat_for1"][i] ] *= img_mask1[:, i].view(-1,1)
        img_cat2 = self.normalizer(img_cat2, p=img_norm2, dim=1, eps=1e-6)
        img_mask2 = self.normalizer(img_cat2, p=1.0, dim=1, eps=1e-6)

        # predict cat3 considering mask2                   # (N, 128)
        img_cat3 = self.image_classifier3(img_cat3)        
        img_norm3 = img_cat3.norm().item()
        for i in CFG["subcat_for2"]:
            img_cat3[ :, CFG["subcat_for2"][i] ] *= img_mask2[:, i].view(-1,1)  
        img_cat3 = self.normalizer(img_cat3, p=img_norm3, dim=1, eps=1e-6)


        """Text stage"""
        # predict cat1 & generate mask1                      # (N, 6)
        txt_cat1 = self.txt_classifier1(txt)
        txt_mask1 = self.normalizer(txt_cat1, p=1.0, dim=1, eps=1e-6)

        # predict cat2 considering mask1 & generate mask2    # (N, 18)
        txt_cat2 = self.txt_classifier2(txt)
        txt_norm2 = txt_cat2.norm().item()
        for i in CFG["subcat_for1"]:
            txt_cat2[ :, CFG["subcat_for1"][i] ] *= txt_mask1[:, i].view(-1,1)
        txt_cat2 = self.normalizer(txt_cat2, p=txt_norm2, dim=1, eps=1e-6)
        txt_mask2 = self.normalizer(txt_cat2, p=1.0, dim=1, eps=1e-6)

        # predict cat3 considering mask2                     # (N, 128)
        txt_cat3 = self.txt_classifier3(txt)
        txt_norm3 = txt_cat3.norm().item()
        for i in CFG["subcat_for2"]:
            txt_cat3[ :, CFG["subcat_for2"][i] ] *= txt_mask2[:, i].view(-1, 1)
        txt_cat3 = self.normalizer(txt_cat3, p=txt_norm3, dim=1, eps=1e-6)


        """final classification stage"""
        cat1 = torch.hstack([img_cat1, txt_cat1]).cuda()
        cat1 = self.fin_classifier1(cat1)

        cat2 = torch.hstack([img_cat2, txt_cat2]).cuda()
        cat2 = self.fin_classifier2(cat2)

        cat3 = torch.hstack([img_cat3, txt_cat3]).cuda()
        cat3 = self.fin_classifier3(cat3)

        return cat1, cat2, cat3




# real : (1792,10,10) -> (256*10*10)
# test : (3,299,299) -> (256*10*10)
class MainNet(torch.nn.Module):  
    def __init__(self):
        super().__init__()
        self.name = "MainNet"
        self.test = True

        """utils"""
        self.activate = torch.nn.ReLU(inplace=True)
        self.flatten  = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.35)

        if self.test:
            self.conv1 = torch.nn.Conv2d(3,   128, 3, stride=4, bias=False)  # -> 128 * 75 * 75
            self.conv2 = torch.nn.Conv2d(128, 128, 3, stride=4, bias=False)  # -> 128 * 18 * 18 
            self.conv3 = torch.nn.Conv2d(128, 256, 9, stride=1, bias=False)  # -> 256 * 10 * 10
            self.bn1   = torch.nn.BatchNorm2d(128)
            self.bn2   = torch.nn.BatchNorm2d(128)
            self.bn3   = torch.nn.BatchNorm2d(256)

        else:
            self.conv1 = torch.nn.Conv2d(1792, 1024, 3, padding="same", bias=False)
            self.conv2 = torch.nn.Conv2d(1024, 512 , 3, padding="same", bias=False)
            self.conv3 = torch.nn.Conv2d(512 , 256 , 3, padding="same", bias=False)
            self.bn1   = torch.nn.BatchNorm2d(1024)
            self.bn2   = torch.nn.BatchNorm2d(512)
            self.bn3   = torch.nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activate(x)
        x = self.dropout(x)

        x = self.flatten(x)
        return x
        

 
class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "TineNet(test)"

        self.image_main = MainNet()
        self.image_classifier1 = torch.nn.Linear(30976, CFG["num_class1"])
        self.image_classifier2 = torch.nn.Linear(30976, CFG["num_class2"])
        self.image_classifier3 = torch.nn.Linear(30976, CFG["num_class3"])

        """for texts"""
        self.txt_classifier1 = torch.nn.Linear(768, CFG["num_class1"])
        self.txt_classifier2 = torch.nn.Linear(768, CFG["num_class2"])
        self.txt_classifier3 = torch.nn.Linear(768, CFG["num_class3"])

        """final classification"""
        self.fin_classifier1 = torch.nn.Linear(CFG["num_class1"]*2, CFG["num_class1"])
        self.fin_classifier2 = torch.nn.Linear(CFG["num_class2"]*2, CFG["num_class2"])
        self.fin_classifier3 = torch.nn.Linear(CFG["num_class3"]*2, CFG["num_class3"])

    def forward(self, x_img, x_txt):
        main_features = self.image_main(x_img)

        """image stage"""
        img1 = self.image_classifier1(main_features)
        img2 = self.image_classifier2(main_features)
        img3 = self.image_classifier3(main_features)

        """text stage"""
        txt1 = self.txt_classifier1(x_txt)
        txt2 = self.txt_classifier2(x_txt)
        txt3 = self.txt_classifier3(x_txt)

        """final classification stage"""
        cat1 = torch.hstack([img1, txt1]).cuda()
        cat1 = self.fin_classifier1(cat1)

        cat2 = torch.hstack([img2, txt2]).cuda()
        cat2 = self.fin_classifier2(cat2)

        cat3 = torch.hstack([img3, txt3]).cuda()
        cat3 = self.fin_classifier3(cat3)

        return cat1, cat2, cat3



class BranchNet(torch.nn.Module):  
    def __init__(self):
        super().__init__()
        self.name = "MainNet"

        """utils"""
        self.activate = torch.nn.ReLU(inplace=True)
        self.flatten  = torch.nn.Flatten()
        self.dropout = torch.nn.Dropout(0.35)

        self.conv1 = torch.nn.Conv2d(1792, 1024, 3, padding="same", bias=False)
        self.conv2 = torch.nn.Conv2d(1024, 512 , 3, padding="same", bias=False)
        self.conv3 = torch.nn.Conv2d(512 , 256 , 3, padding="same", bias=False)
        self.bn1   = torch.nn.BatchNorm2d(1024)
        self.bn2   = torch.nn.BatchNorm2d(512)
        self.bn3   = torch.nn.BatchNorm2d(256)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activate(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.activate(x)
        x = self.dropout(x)

        x = self.flatten(x)
        return x


class TinyNet3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "TineNet3"

        """utils"""
        self.normalizer = torch.nn.functional.normalize

        """image classifier"""
        self.image_main = MainNet()
        self.image_classifier1 = torch.nn.Linear(30976, CFG["num_class1"])
        self.image_classifier2 = torch.nn.Linear(30976, CFG["num_class2"])
        self.image_classifier3 = torch.nn.Linear(30976, CFG["num_class3"])

        """image transferer from cat1 to cat3"""
        self.image_transferer1 = torch.nn.Linear(CFG["num_class1"], CFG["num_class2"])
        self.image_tranhelper1 = torch.nn.Linear(CFG["num_class2"], CFG["num_class2"])

        self.image_transferer2 = torch.nn.Linear(CFG["num_class2"], CFG["num_class3"])
        self.image_tranhelper2 = torch.nn.Linear(CFG["num_class3"], CFG["num_class3"])



        """text classifier"""
        self.txt_classifier1 = torch.nn.Linear(768, CFG["num_class1"])
        self.txt_classifier2 = torch.nn.Linear(768, CFG["num_class2"])
        self.txt_classifier3 = torch.nn.Linear(768, CFG["num_class3"])

        """text transferer from cat1 to cat3"""
        self.txt_transferer1 = torch.nn.Linear(CFG["num_class1"], CFG["num_class2"])
        self.txt_tranhelper1 = torch.nn.Linear(CFG["num_class2"], CFG["num_class2"])

        self.txt_transferer2 = torch.nn.Linear(CFG["num_class2"], CFG["num_class3"])
        self.txt_tranhelper2 = torch.nn.Linear(CFG["num_class3"], CFG["num_class3"])


        """final classifier"""
        self.fin_classifier1_1 = torch.nn.Linear(CFG["num_class1"]*2, CFG["num_class1"])
        self.fin_classifier2_1 = torch.nn.Linear(CFG["num_class2"]*2, CFG["num_class2"])
        self.fin_classifier3_1 = torch.nn.Linear(CFG["num_class3"]*2, CFG["num_class3"])

        self.fin_classifier1_2 = torch.nn.Linear(CFG["num_class1"], CFG["num_class1"])
        self.fin_classifier2_2 = torch.nn.Linear(CFG["num_class2"], CFG["num_class2"])
        self.fin_classifier3_2 = torch.nn.Linear(CFG["num_class3"], CFG["num_class3"])


    def forward(self, x_img, x_txt):
        main_features = self.image_main(x_img)

        """image stage"""
        img1 = self.image_classifier1(main_features)
        img2 = self.image_classifier2(main_features)
        img3 = self.image_classifier3(main_features)

        # transfer information
        img_transfer1 = self.image_transferer1(img1)
        img_transfer1 = self.image_tranhelper1(img_transfer1)
        img2 = img_transfer1*img2
        
        img_transfer2 = self.image_transferer2(img2)
        img_transfer2 = self.image_tranhelper2(img_transfer2)
        img3 = img_transfer2*img3
        

        """text stage"""
        txt1 = self.txt_classifier1(x_txt)
        txt2 = self.txt_classifier2(x_txt)
        txt3 = self.txt_classifier3(x_txt)

        # transfer information
        txt_transfer1 = self.txt_transferer1(txt1)
        txt_transfer1 = self.txt_tranhelper1(txt_transfer1)
        txt2 = txt_transfer1*txt2
        
        txt_transfer2 = self.txt_transferer2(txt2)
        txt_transfer2 = self.txt_tranhelper2(txt_transfer2)
        txt3 = txt_transfer2*txt3


        """final classification stage"""
        cat1 = torch.hstack([img1, txt1]).cuda()
        cat1 = self.fin_classifier1_1(cat1)
        cat1 = self.fin_classifier1_2(cat1)

        cat2 = torch.hstack([img2, txt2]).cuda()
        cat2 = self.fin_classifier2_1(cat2)
        cat2 = self.fin_classifier2_2(cat2)


        cat3 = torch.hstack([img3, txt3]).cuda()
        cat3 = self.fin_classifier3_1(cat3)
        cat3 = self.fin_classifier3_2(cat3)

        return cat1, cat2, cat3


class TinyNet3_HLN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "TinyNet3_HLN"

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
        self.image_main = effnetv2_l_main()

        self.image_branch1 = BranchNet()
        self.image_branch2 = BranchNet()
        self.image_branch3 = BranchNet()

        self.image_classifier1 = torch.nn.Linear(256*10*10, CFG["num_class1"])
        self.image_classifier2 = torch.nn.Linear(256*10*10, CFG["num_class2"])
        self.image_classifier3 = torch.nn.Linear(256*10*10, CFG["num_class3"])

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
        img1 = self.image_branch1(main_features)
        img2 = self.image_branch2(main_features)
        img3 = self.image_branch3(main_features)

        img1 = self.image_classifier1(img1)
        
        # transfer information 1 to 2 (masking)
        img1_softmax = self.softmax(img1, dim=1)
        img1_labels = self.argmax(img1_softmax, dim=1)
        img1_mask = self.filter_12[img1_labels]

        # transfer information 1 to 2 (filtering)
        img2 = self.image_classifier2(img2)
        img2 = img2*img1_mask.cuda()

        # transfer information 2 to 3 (masking)
        img2_softmax = self.softmax(img2, dim=1)
        img2_labels = self.argmax(img2_softmax, dim=1)
        img2_mask = self.filter_23[img2_labels]

        # transfer information 2 to 30 (filtering)
        img3 = self.image_classifier3(img3)
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