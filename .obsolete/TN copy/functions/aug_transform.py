from config import * 
import os
import sys
sys.path.append(".")
# from tqdm.auto import tqdm as tq

import warnings
warnings.filterwarnings(action="ignore")

import re
import random
import pandas as pd
import numpy as np
import pickle

# for images
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# for texts
from sklearn.feature_extraction.text import CountVectorizer
from konlpy.tag import Okt
current_working_dir = os.getcwd()
os.chdir(CFG["text_eda_dir"])
from .KorEDA import eda
os.chdir(current_working_dir)


"""
UTILS
"""
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


"""
PREPARE AUGMENTATION
"""
def prepare_augmentation(data:pd.DataFrame):
    seed_everything(CFG['SEED'])
    make_dir(data_path+"augmented")
    print("execute data augmentation...(it takes about 10~15 minutes to complete)")
    # for counting
    nums = data.groupby("cat3").count()[["id"]]
    nums.columns = ["num"]
    nums = nums.reset_index()
    target_50   = []
    target_100  = []
    target_200  = []

    targets      = [target_50, target_100, target_200]
    targets_cnts = [50,        100,        200]
    for i in range(len(nums)):
        cur = nums.iloc[i]
        cnt = cur["num"]; name = cur["cat3"]
        if(cnt < 50):
            targets[0].append(name)
        elif(cnt < 100):
            targets[1].append(name)
        elif(cnt < 200):
            targets[2].append(name)

            
    idx = 0
    new_random_idx = 0 
    augmented_cat1 = []
    augmented_cat2 = []
    augmented_cat3 = []

    for a in (range(len(targets))):
        for cat_name in (targets[a]):
            cur_df = data[data["cat3"]==cat_name]
            img_index = cur_df.index.tolist()
            needed = targets_cnts[a]-len(img_index)
            cur_idx = 0
            while(cur_idx <= needed):
                new_random_idx +=1
                # 1) sampling
                random.seed(CFG["SEED"] + new_random_idx *5) # random + random
                i = random.sample(img_index, 1)[0]    
                augmented_cat1.append(cur_df.loc[i]["cat1_enc"])
                augmented_cat2.append(cur_df.loc[i]["cat2_enc"])
                augmented_cat3.append(cur_df.loc[i]["cat3_enc"])
                cur_idx += 1
                idx += 1
            random.seed(CFG["SEED"])
            
    # save augmented data's categories
    with open(CFG["augcat1_dir"], "wb") as f:
        pickle.dump(augmented_cat1, f)
    with open(CFG["augcat2_dir"], "wb") as f:
        pickle.dump(augmented_cat2, f)
    with open(CFG["augcat3_dir"], "wb") as f:
        pickle.dump(augmented_cat3, f)
    
    args = {
        "targets":targets,
        "targets_cnts":targets_cnts,
        "augmented_cat1":augmented_cat1,
        "augmented_cat2":augmented_cat2,
        "augmented_cat3":augmented_cat3,
    }
    # args = [targets, targets_cnts, augmented_cat1, augmented_cat2, augmented_cat3]
    
    return args


"""
IMAGE AUGMENTATION & TRANSFORMATION
"""
def execute_img_augmentation(data:pd.DataFrame, args):
    seed_everything(CFG['SEED'])
    targets        = args["targets"]
    targets_cnts   = args["targets_cnts"]
    augmented_cat1 = args["augmented_cat1"]
    augmented_cat2 = args["augmented_cat2"]
    augmented_cat3 = args["augmented_cat3"]
    
    idx = 0
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
                
                # 2) append augmented image
                augmented_img.append(cur_img)

                cur_idx += 1
                idx += 1
            random.seed(CFG["SEED"])
    
    # save augmented imgs                
    with open(CFG["augimg_dir"], "wb") as f:
        pickle.dump(augmented_img, f)
    
    
def execute_img_transformation(image:str):
    image = cv2.imread(image)
    image = resize(image)
    image = transform(image)
    image = normalize(image)
    image = to_tensor(image)
    return image

# sub functions
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
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
        ]),
        A.OneOf([
            A.MotionBlur(p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ]),
        A.OneOf([
            A.ISONoise(p=0.5),
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.ColorJitter(p=0.5),
        ]),
    ])
    return aug_sequence(image=x)["image"]

def to_tensor(x):
    return A.pytorch.ToTensorV2()(image=x)["image"].float()


"""
TEXT AUGMENTATION & TRANSFORMATION
"""
def execute_txt_augmentation(data:pd.DataFrame, args):
    seed_everything(CFG['SEED'])
    targets        = args["targets"]
    targets_cnts   = args["targets_cnts"]
    augmented_cat1 = args["augmented_cat1"]
    augmented_cat2 = args["augmented_cat2"]
    augmented_cat3 = args["augmented_cat3"]
    
    # prepare tokenizer and text data for "text augmentation"
    tokenizer = Okt()
    
    idx = 0
    augmented_text = []
    new_random_idx = 0     
    
    for a in (range(len(targets))):
        for cat_name in (targets[a]):
            cur_df = data[data["cat3"]==cat_name]
            img_index = cur_df.index.tolist()
            needed = targets_cnts[a]-len(img_index)
            cur_idx = 0
            while(cur_idx <= needed):
                new_random_idx +=1
                # 1) sampling
                random.seed(CFG["SEED"] + new_random_idx *5) # random + random
                i = random.sample(img_index, 1)[0]    
                # 2) text augmentation
                cur_txt = cur_df.loc[i]["overview"]
                if len(cur_txt)>=1:
                    augmented_txt_list = eda.EDA(cur_txt) # EDA package
                    cur_txt = random.choice(augmented_txt_list)
                else:
                    pass               
                
                # save augmented text data
                augmented_text.append(cur_txt)
                cur_idx += 1
                idx += 1
            random.seed(CFG["SEED"]) 

    # save augmented text 
    with open(CFG["augtxt_dir"], "wb") as f:
        pickle.dump(augmented_text, f)    


def execute_cnt_embedding(data:pd.DataFrame, augmented_text:list, max_features=CFG["embedding_dim"], infer=False):
    tokenizer  = Okt()
    vectorizer = CountVectorizer(max_features=max_features)
    
    # load data
    if not infer:
        texts = data["overview"].tolist() + augmented_text    
    else:
        texts = data["overview"].tolist()
    
    # tokenizing
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
    
    # vectorizing
    print("verctorizing...")
    new_texts = vectorizer.fit_transform(new_texts).todense()
    new_texts = torch.tensor(new_texts).float()
    
    # save tensor
    if not infer:
        torch.save(new_texts, CFG["embedding_dir"])
    else:
        torch.save(new_texts, CFG["infer_embedding_dir"])    