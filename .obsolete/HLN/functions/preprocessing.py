
import warnings
warnings.filterwarnings(action="ignore")

import sys
import os
import pickle
sys.path.append(".")
# private modules
from config import * 

# for texts
from konlpy.tag import Okt
current_working_dir = os.getcwd()
os.chdir(CFG["text_eda_dir"])
from .KorEDA import eda
os.chdir(current_working_dir)

# for NNs
import torch
from torch.utils.data import Dataset, DataLoader

# utils
import re
import pickle
import pandas as pd
import numpy as np
import random

# from tqdm.auto import tqdm as tq
import warnings
warnings.filterwarnings(action="ignore")

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# for images
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_dir(directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory)

def resize(x):
    H, W, C = x.shape
    if H>W:
        tf = A.Compose([
            A.Resize(CFG["IMG_SIZE"]*H//W, CFG["IMG_SIZE"]),
        ])
    else:
        tf = A.Compose([
            A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]*W//H),
        ])
    return tf(image=x)["image"]

def to_tensor(x):
    return A.pytorch.ToTensorV2()(image=x)["image"].float()

def normalize(x):
    return np.round(x/255, 4)

def transform(x:np.array):
    seed_everything(CFG['SEED'])
    # soft transformation
    tf_sequence = A.Compose([
        A.RandomCrop(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5)
        ]),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.GaussNoise(p=0.5),
        ])
    ])
    return tf_sequence(image=x)["image"]

def augment(x:np.array):
    seed_everything(CFG['SEED'])
    # hard transformation (for augmentation)
    aug_sequence = A.Compose([
        A.RandomCrop(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.MotionBlur(p=0.5),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.Blur(blur_limit=3, p=0.5),
        A.ISONoise(p=0.5),
        A.GaussNoise(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.ColorJitter(p=0.5),
    ])
    return aug_sequence(image=x)["image"]


def load_data(do_augmentation=False, do_embedding=False):
    seed_everything(CFG['SEED']) # Seed 고정

    def execute_augmentation(data:pd.DataFrame):
        seed_everything(CFG['SEED'])
        make_dir(data_path+"augmented")
        
        print("execute data augmentation...(it takes about 10~15 minutes to complete)")
        
        # for counting
        nums = data.groupby("cat3").count()[["id"]]
        nums.columns = ["num"]
        nums = nums.reset_index()
        target_50   = []
        target_100  = []
        target_300  = []
        target_500  = []
        targets      = [target_50, target_100, target_300, target_500]
        targets_cnts = [50,        100,        300,        500]
        for i in range(len(nums)):
            cur = nums.iloc[i]
            cnt = cur["num"]; name = cur["cat3"]
            if(cnt < 50):
                targets[0].append(name)
            elif(cnt < 100):
                targets[1].append(name)
            elif(cnt < 300):
                targets[2].append(name)
            elif(cnt < 500):
                targets[3].append(name)


        # prepare tokenizer and text data for "text augmentation"
        tokenizer = Okt()
        
        
        idx = 0
        augmented_text = []
        augmented_cat1 = []
        augmented_cat2 = []
        augmented_cat3 = []
        new_random_idx = 0 
        augmented_img = []
        for a in (range(len(targets))):
            for cat_name in (targets[a]):
                cur_df = data[data["cat3"]==cat_name]
                img_index = cur_df.index.tolist()
                needed = targets_cnts[a]-len(img_index)
                cur_idx = 0
                while(cur_idx <= needed):
                    new_random_idx +=1
                    # 1. image augmentation 
                    # 1) sampling
                    random.seed(CFG["SEED"] + new_random_idx *5) # random + random
                    
                    i = random.sample(img_index, 1)[0]    
                    path = cur_df.loc[i]["img_path"]
                    cur_img = cv2.imread(path)
                    cur_img = resize(cur_img)
                    cur_img = normalize(augment(cur_img))
                    cur_img = to_tensor(cur_img).float()
                    
                    # 2) save augmented image
                    augmented_img.append(cur_img)

                    # 2. text augmentation - word replacement
                    # 1) text augmentation
                    cur_txt = cur_df.loc[i]["overview"]
                    if len(cur_txt)>=1:
                        augmented_txt_list = eda.EDA(cur_txt) # EDA package
                        cur_txt = random.choice(augmented_txt_list)
                    else:
                        pass               
                    
                    # save augmented text data
                    augmented_text.append(cur_txt)
                    augmented_cat1.append(cur_df.loc[i]["cat1_enc"])
                    augmented_cat2.append(cur_df.loc[i]["cat2_enc"])
                    augmented_cat3.append(cur_df.loc[i]["cat3_enc"])
                    cur_idx += 1
                    idx += 1
                random.seed(CFG["SEED"])

        # save augmented imgs                
        with open(CFG["augimg_dir"], "wb") as f:
            pickle.dump(augmented_img, f)
        # save augmented text 
        with open(CFG["augtxt_dir"], "wb") as f:
            pickle.dump(augmented_text, f)
            
        # save augmented data's categories
        with open(CFG["augcat1_dir"], "wb") as f:
            pickle.dump(augmented_cat1, f)
        with open(CFG["augcat2_dir"], "wb") as f:
            pickle.dump(augmented_cat2, f)
        with open(CFG["augcat3_dir"], "wb") as f:
            pickle.dump(augmented_cat3, f)
 
        return idx


    def execute_cnt_embedding(data:pd.DataFrame, augmented_text:list, max_features=CFG["embedding_dim"]):
        tokenizer  = Okt()
        vectorizer = CountVectorizer(max_features=max_features)
        texts = data["overview"].tolist() + augmented_text    
        new_texts = []
        print("tokenizing...(it takes about 8~10 minutes to complete.)")
        for i in (range(len(texts))):
            # preprocess special characters
            texts[i] = texts[i].replace("\n", " ").replace("br", "")
            texts[i] = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", texts[i])
            
            # tokenize using Okt tokenizer (only extract Noun or Adjective tokens in normalized form)
            tokenized = []
            tokenized = tokenizer.pos(texts[i], norm=True, stem=True)
            tokenized = [ token[0] for token in tokenized if ( token[1] in ["Noun","Adjective","Verb","Exclamation"]) ]
            new_texts.append(" ".join(tokenized))
        
        print("verctorizing...")
        new_texts = vectorizer.fit_transform(new_texts).todense()
        new_texts = torch.tensor(new_texts).float()
        torch.save(new_texts, CFG["embedding_dir"])
        return 
    
    # load data (original data)
    data = pd.read_csv(data_path + "train.csv")
    data["img_path"] = data["img_path"].str.replace("./image/train/", CFG["org_img_train"], regex=False)
    data["overview"] = data["overview"].str.replace("<br>", "").fillna(" ")
    
    # text preprocessing
    for i in range(len(data["overview"])):
        cur_txt = data.loc[i]["overview"]
        cur_txt = cur_txt.replace("\n\n", "").replace("br", "").replace("strong", "")
        cur_txt = cur_txt.split(".")
        cur_txt = [re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", t) for t in cur_txt]
        random.shuffle(cur_txt)
        cur_txt = [t.lstrip().rstrip() for t in cur_txt]
        cur_txt[:] = list(filter(None, cur_txt))
        cur_txt = " ".join(cur_txt)
        data.loc[i]["overview"] = cur_txt

    # label encoding
    enc = LabelEncoder()
    data["cat1_enc"] = enc.fit_transform(data["cat1"])
    data["cat2_enc"] = enc.fit_transform(data["cat2"])
    data["cat3_enc"] = enc.fit_transform(data["cat3"])
    cat1_enc = data["cat1_enc"].tolist()
    cat2_enc = data["cat2_enc"].tolist()
    cat3_enc = data["cat3_enc"].tolist()
    cat1_set = list(set(cat1_enc))
    cat2_set = list(set(cat2_enc))
    cat3_set = list(set(cat3_enc))
    subcat_for1 = dict()
    subcat_for2 = dict()
    for cat in cat1_set:
        subcat_for1[cat] = list()
    for cat in cat2_set:
        subcat_for2[cat] = list()
    for cat in cat1_set:
        subcat_for1[cat] += sorted(data[data["cat1_enc"]==cat]["cat2_enc"].unique().tolist())
    for cat in cat2_set:
        subcat_for2[cat] += sorted(data[data["cat2_enc"]==cat]["cat3_enc"].unique().tolist())
    CFG["subcat_for1"] = subcat_for1
    CFG["subcat_for2"] = subcat_for2
        
    print("data loaded...")
    # (execute augmentation)
    if do_augmentation:
        execute_augmentation(data)

    # load augmented imgs  ( List of tensors )
    with open(CFG["augimg_dir"], "rb") as f:
        augmented_img = pickle.load(f)
    # load augmented text  ( List of strings )
    with open(CFG["augtxt_dir"], "rb") as f:
        augmented_text = pickle.load(f)
        
    # load augmented data's categories ( List of integers )
    with open(CFG["augcat1_dir"], "rb") as f:
        augmented_cat1 = pickle.load(f)
    with open(CFG["augcat2_dir"], "rb") as f:
        augmented_cat2 = pickle.load(f)
    with open(CFG["augcat3_dir"], "rb") as f:
        augmented_cat3 = pickle.load(f)


    aug_image_indices = [f'{i}.pt' for i in range(len(augmented_img))] # list of indices of each aug img tensor

    # original data + augmented data
    image_paths = data["img_path"].tolist() + aug_image_indices

    # (execute embedding)
    if do_embedding:
        execute_cnt_embedding(data, augmented_text)

    # load text vector data (original + augmented)
    texts_cntvector = torch.load(CFG["embedding_dir"])
    
    # y1, y2, y3 (labels)
    y1 = data["cat1_enc"].tolist() + augmented_cat1
    y2 = data["cat2_enc"].tolist() + augmented_cat2
    y3 = data["cat3_enc"].tolist() + augmented_cat3

    # imge path, text count vector, raw text (~5 lines)
    return image_paths, texts_cntvector, y1, y2, y3, augmented_img # augmented_img : list of tensors


class AugmentDataset(Dataset):   # X1 : img path  / X2 : txt cntvector / X3 : txt raw
    def __init__(self, X1:list, X2:torch.tensor, y1:list, y2:list, y3:list, augmented_img:list, infer_flag=False):
        self.X_images          = X1
        self.X_texts_cntvector = X2
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.infer_flag = infer_flag
        self.augmented_img = torch.stack(augmented_img) # list of tensors

    def resize(self, x):
        H, W, C = x.shape
        if H>W:
            tf = A.Compose([
                A.Resize(CFG["IMG_SIZE"]*H//W, CFG["IMG_SIZE"]),
            ])
        else:
            tf = A.Compose([
                A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]*W//H),
            ])
        return tf(image=x)["image"]

    def to_tensor(self, x):
        return A.pytorch.ToTensorV2()(image=x)["image"].float()

    def normalize(self, x):
        return np.round(x/255, 4)

    def transform(self, x:np.array):
        # soft transformation
        tf_sequence = A.Compose([
            A.RandomCrop(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
            A.OneOf([
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=20, p=0.5)
            ]),
            A.OneOf([
                A.MotionBlur(p=0.5),
                A.GaussNoise(p=0.5),
            ])
        ])
        return tf_sequence(image=x)["image"]

    def __len__(self):
        return len(self.X_images)
    
    def __getitem__(self, idx):
        # transform image from str to tensor
        image    = self.X_images[idx]
        data_type= image.split(".")[1]
        if(data_type =="jpg"):
            image = cv2.imread(image)
            image = self.resize(image)
            image = self.transform(image)
            image = self.normalize(image)
            image = self.to_tensor(image)
        elif(data_type =="pt"):
            image = self.augmented_img[int(self.X_images[idx].split(".pt")[0])]

        # text
        text_cntvector = self.X_texts_cntvector[idx]
        
        if self.infer_flag :
            return image, text_cntvector
        else:
            # load label
            y1 = torch.tensor(self.y1[idx])
            y2 = torch.tensor(self.y2[idx])
            y3 = torch.tensor(self.y3[idx])
            y = torch.stack([y1,y2,y3]).T
            return image, text_cntvector, y
        
        
                        # X1 : img path  / X2 : txt cntvector / X3 : txt raw
def generate_dataloader(X1:list, X2:torch.tensor, y1:list, y2:list, y3:list, augmented_img:list):
    seed_everything(CFG["SEED"])
    # CFG["num_class"] = len(set(y))

    X_img_train, X_img_valid, X_text_cntvector_train, X_text_cntvector_valid, Y1_train, Y1_valid, Y2_train, Y2_valid, Y3_train, Y3_valid = train_test_split(
        X1,
        X2,
        y1,
        y2,
        y3,
        test_size=CFG["test_size"], 
        random_state=CFG["SEED"],
        stratify=y3
    )

    train_dataset = AugmentDataset(
        X1=X_img_train,
        X2=X_text_cntvector_train,
        y1=Y1_train,
        y2=Y2_train,
        y3=Y3_train, 
        augmented_img=augmented_img,
        infer_flag=False
    )
    valid_dataset = AugmentDataset(
        X1=X_img_valid,
        X2=X_text_cntvector_valid,
        y1=Y1_valid,
        y2=Y2_valid,
        y3=Y3_valid, 
        augmented_img=augmented_img,
        infer_flag=False
    )
    loader_cfg = {
        "batch_size"    : CFG["BATCH_SIZE"], 
        "shuffle"       : True, 
        # "num_workers"   : 4
    }
    train_loader  = DataLoader(dataset=train_dataset, **loader_cfg)
    val_loader  = DataLoader(dataset=valid_dataset, **loader_cfg)

    return train_loader, val_loader

