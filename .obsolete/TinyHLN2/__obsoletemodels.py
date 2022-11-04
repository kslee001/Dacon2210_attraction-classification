import re
import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer


__all__ = ["InceptionNet", "InceptionNet2", "InceptionNet3", "InceptionNet4", "MultiNet", "MultiNet2", "MultiNetLight"]


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


class ResidualCell(torch.nn.Module):
    def __init__(self, sublayer:torch.nn.Module):
        super().__init__()
        self.sublayer = sublayer
    def forward(self, x):
        resid = x
        x = self.sublayer(x)    
        x = x+resid
        x = torch.nn.functional.relu(x)
        return x


class _Stem(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            BasicConv2d(3, 32, kernel_size=3, stride=2),
            BasicConv2d(32, 32, kernel_size=3),
            BasicConv2d(32, 64, kernel_size=3, padding="same"),
            torch.nn.MaxPool2d(kernel_size=3, stride=2),
            BasicConv2d(64, 80, kernel_size=1, padding="same"),
            BasicConv2d(80, 192, kernel_size=3),
            BasicConv2d(192, 384, kernel_size=3, stride=2)
        ]
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _InceptionA_Res(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_a = BasicConv2d(384, 32, kernel_size=1, padding="same")

        self.conv_b1 = BasicConv2d(384, 32, kernel_size=1, padding="same")
        self.conv_b2 = BasicConv2d(32, 32, kernel_size=3, padding="same")

        self.conv_c1 = BasicConv2d(384, 32, kernel_size=1, padding="same")
        self.conv_c2 = BasicConv2d(32, 32, kernel_size=3, padding="same")
        self.conv_c3 = BasicConv2d(32, 32, kernel_size=3, padding="same")

        self.conv_fin = BasicConv2d(32*3, 384, kernel_size=1, padding="same")
        

    def forward(self, x):
        resid = x
        a = self.conv_a(x)

        b = self.conv_b1(x)
        b = self.conv_b2(b)

        c = self.conv_c1(x)
        c = self.conv_c2(c)
        c = self.conv_c3(c)

        x = torch.cat((a,b,c), 1)
        x = self.conv_fin(x)
        
        x = resid + x
        x = torch.nn.functional.relu(x)
        return x


class _ReductionA_Res(torch.nn.Module):
    def __init__(self):
        super().__init__()
        k=256; l=256; m=384; n=384
        self.maxpool_a = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_b = BasicConv2d(384, n, kernel_size=3, stride=2)

        self.conv_c1 = BasicConv2d(384, k, kernel_size=1, padding="same")
        self.conv_c2 = BasicConv2d(k, l, kernel_size=3, padding="same")
        self.conv_c3 = BasicConv2d(l, m, kernel_size=3, stride=2)

    def forward(self, x):
        a = self.maxpool_a(x)
        b = self.conv_b(x)

        c = self.conv_c1(x)
        c = self.conv_c2(c)
        c = self.conv_c3(c)

        x = torch.cat((a,b,c), 1)
        return x


class _InceptionB_Res(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_a = BasicConv2d(1152, 192, kernel_size=1, padding="same")

        self.conv_b1 = BasicConv2d(1152, 128, kernel_size=1, padding="same")
        self.conv_b2 = BasicConv2d(128, 160, kernel_size=[1,7], padding="same")
        self.conv_b3 = BasicConv2d(160, 192, kernel_size=[7,1], padding="same")

        self.conv_fin = BasicConv2d(384, 1152, kernel_size=1, padding="same")
        

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


# 7->5 second filter size
class _InceptionB_Res2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_a = BasicConv2d(1152, 192, kernel_size=1, padding="same")

        self.conv_b1 = BasicConv2d(1152, 128, kernel_size=1, padding="same")
        self.conv_b2 = BasicConv2d(128, 160, kernel_size=[1,5], padding="same")
        self.conv_b3 = BasicConv2d(160, 192, kernel_size=[5,1], padding="same")

        self.conv_fin = BasicConv2d(384, 1152, kernel_size=1, padding="same")
        

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


class _ReductionB_Res(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool_a = torch.nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv_b1 = BasicConv2d(1152, 256, kernel_size=1, padding="same")
        self.conv_b2 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.conv_c1 = BasicConv2d(1152, 256, kernel_size=1, padding="same")
        self.conv_c2 = BasicConv2d(256, 288, kernel_size=3, stride=2)

        self.conv_d1 = BasicConv2d(1152, 256, kernel_size=1, padding="same")
        self.conv_d2 = BasicConv2d(256, 288, kernel_size=3, padding="same")
        self.conv_d3 = BasicConv2d(288, 320, kernel_size=3, stride=2)

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
        # out_channels : 2016
        return x


class _InceptionC_Res(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_a = BasicConv2d(2144, 192, kernel_size=1, padding="same")

        self.conv_b1 = BasicConv2d(2144, 192, kernel_size=1, padding="same")
        self.conv_b2 = BasicConv2d(192, 224, kernel_size=[1,3], padding="same")
        self.conv_b3 = BasicConv2d(224, 256, kernel_size=[3,1], padding="same")

        self.conv_fin = BasicConv2d(448, 2144, kernel_size=1, padding="same")
        

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




class InceptionNet(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.layers = torch.nn.Sequential(
            _Stem(),
            # 5 Inception A
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),

            _ReductionA_Res().to(self.device),

            # 10 Inception B
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),
            _InceptionB_Res().to(self.device),

            _ReductionB_Res().to(self.device),

            # 5 Inception C
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),

            torch.nn.AvgPool2d(kernel_size=3),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.80),
            torch.nn.Flatten(),
            torch.nn.Linear(8576, num_class).to(self.device),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.layers(x).to(self.device)
        return x


class InceptionNet2(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.layers = torch.nn.Sequential(
            _Stem(),
            # 5 Inception A
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),

            _ReductionA_Res().to(self.device),

            # 10 Inception B
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),

            _ReductionB_Res().to(self.device),

            # 5 Inception C
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),

            torch.nn.MaxPool2d(kernel_size=3),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.75),
            torch.nn.Flatten(),
            torch.nn.Linear(8576, 512).to(self.device),
            torch.nn.Linear(512, num_class).to(self.device)
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.layers(x).to(self.device)
        return x


class InceptionNet3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.layers = torch.nn.Sequential(
            _Stem(),
            # 5 Inception A
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),

            _ReductionA_Res().to(self.device),

            # 10 Inception B
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),

            _ReductionB_Res().to(self.device),

            # 5 Inception C
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),

            torch.nn.MaxPool2d(kernel_size=3),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.75),
            torch.nn.Flatten(),
            torch.nn.Linear(8576, 2304).to(self.device),

            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.layers(x).to(self.device)
        return x   # (N, 2304)



class InceptionNet4(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.layers = torch.nn.Sequential(
            _Stem(),
            # 5 Inception A -> 3
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            _InceptionA_Res().to(self.device),
            # _InceptionA_Res().to(self.device),
            # _InceptionA_Res().to(self.device),

            _ReductionA_Res().to(self.device),

            # 10 Inception B -> 5
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            _InceptionB_Res2().to(self.device),
            # _InceptionB_Res2().to(self.device),
            # _InceptionB_Res2().to(self.device),
            # _InceptionB_Res2().to(self.device),
            # _InceptionB_Res2().to(self.device),
            # _InceptionB_Res2().to(self.device),

            _ReductionB_Res().to(self.device),

            # 5 Inception C -> 3
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            _InceptionC_Res().to(self.device),
            # _InceptionC_Res().to(self.device),
            # _InceptionC_Res().to(self.device),

            torch.nn.MaxPool2d(kernel_size=3),
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.75),
            torch.nn.Flatten(),
            torch.nn.Linear(8576, 2304).to(self.device),

            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(self.device)
        x = self.layers(x).to(self.device)
        return x   # (N, 2304)



class MultiNet(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.name = "MultiNet"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_model = InceptionNet3().to(self.device)
        self.txt_model = BertModel.from_pretrained('skt/kobert-base-v1').to(self.device)
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
        self.classifier = torch.nn.Linear(3072, num_class).to(self.device)
        self.txt_length = 128

    def embed(self, x2:list): # X2[idx]
        txt_vectors = []
        for txt in x2:
            cur = self.tokenizer.batch_encode_plus([txt])
            cur = (torch.tensor(cur["input_ids"]).to(self.device), torch.tensor(cur["attention_mask"]).to(self.device))
            cur = self.txt_model(cur[0], cur[1]).pooler_output.to(self.device)
            txt_vectors.append(cur)
        txt_vectors = torch.vstack(txt_vectors).float()
        
        return txt_vectors.to(self.device)

    def forward(self, x1, x2):
        x1 = x1.to(self.device).float()
        x1 = self.img_model(x1) # (N, 2304)
        x2 = self.embed(x2)     # (N, 768)

        # concatenate image vector and text vector
        x = torch.cat([x1, x2], 1).to(self.device)
        x = self.classifier(x)
        return x


class MultiNet2(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.name = "MultiNet2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_model = InceptionNet2(num_class).to(self.device)
        self.txt_model = BertModel.from_pretrained('skt/kobert-base-v1').to(self.device)
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
        
        self.img_classifier = torch.nn.Linear(2304, 256).to(self.device)
        self.txt_classifier = torch.nn.Linear(768,  256).to(self.device)
        self.con_classifier = torch.nn.Linear(3072, 256).to(self.device)
        self.fin_classifier = torch.nn.Linear(256, num_class).to(self.device)

        self.relu = torch.nn.functional.relu().to(self.device)

    def embed(self, x2:list): # X2[idx]
        txt_vectors = []
        for txt in x2:
            cur = self.tokenizer.batch_encode_plus([txt])
            cur = (torch.tensor(cur["input_ids"]).to(self.device), torch.tensor(cur["attention_mask"]).to(self.device))
            cur = self.txt_model(cur[0], cur[1]).pooler_output.to(self.device)
            txt_vectors.append(cur)
        txt_vectors = torch.vstack(txt_vectors).float()
        
        return txt_vectors.to(self.device)

    def forward(self, x1, x2):
        # image
        x1 = x1.to(self.device)
        x1 = self.img_model(x1)        # (N, 2304)
        
        # text
        x2 = self.embed(x2).to(device) # (N, 768)

        # 1. concatenated  vector
        x = torch.cat([x1, x2], 1).to(self.device)
        x = self.con_classifier(x)
        x = self.relu(x)

        # 2. image's choice
        x1 = self.img_classifier(x1)
        x1 = self.relu(x1)

        # 3. text's choice
        x2 = self.txt_classifier(x2)
        x2 = self.relu(x2)

        x = self.fin_classifier(x+x1+x2)        
        return x



class MultiNetLight(torch.nn.Module):
    def __init__(self, num_class):
        super().__init__()
        self.name = "MultiNet2"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.img_model1 = InceptionNet4(num_class).to(self.device)
        self.img_model2 = InceptionNet4(num_class).to(self.device)
        self.img_model3 = InceptionNet4(num_class).to(self.device)
        
        self.txt_model = BertModel.from_pretrained('skt/kobert-base-v1').to(self.device)
        self.tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1', sp_model_kwargs={'nbest_size': -1, 'alpha': 0.6, 'enable_sampling': True})
        
        self.img_classifier1 = torch.nn.Linear(2304, 256).to(self.device)
        self.img_classifier2 = torch.nn.Linear(2304, 256).to(self.device)
        self.img_classifier3 = torch.nn.Linear(2304, 256).to(self.device)
        self.txt_classifier = torch.nn.Linear(768,  256).to(self.device)
        self.con_classifier = torch.nn.Linear(3072, 256).to(self.device)
        self.fin_classifier = torch.nn.Linear(256, num_class).to(self.device)

        self.relu = torch.nn.functional.relu().to(self.device)

    def embed(self, x2:list): # X2[idx]
        txt_vectors = []
        for txt in x2:
            cur = self.tokenizer.batch_encode_plus([txt])
            cur = (torch.tensor(cur["input_ids"]).to(self.device), torch.tensor(cur["attention_mask"]).to(self.device))
            cur = self.txt_model(cur[0], cur[1]).pooler_output.to(self.device)
            txt_vectors.append(cur)
        txt_vectors = torch.vstack(txt_vectors).float()
        
        return txt_vectors.to(self.device)

    def forward(self, x1, x2):
        # image
        x1 = x1.to(self.device)
        x1 = self.img_model(x1)        # (N, 2304)
        
        # text
        x2 = self.embed(x2).to(device) # (N, 768)

        # 1. concatenated  vector
        x = torch.cat([x1, x2], 1).to(self.device)
        x = self.con_classifier(x)
        x = self.relu(x)

        # 2. image's choice
        x1 = self.img_classifier(x1)
        x1 = self.relu(x1)

        # 3. text's choice
        x2 = self.txt_classifier(x2)
        x2 = self.relu(x2)

        x = self.fin_classifier(x+x1+x2)        
        return x


