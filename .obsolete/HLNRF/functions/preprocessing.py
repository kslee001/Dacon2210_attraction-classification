# private modules
from config import * 
import functions.aug_transform as aug_transform

# public modules
import warnings
warnings.filterwarnings(action="ignore")
import os
import sys
sys.path.append(".")
import re
import pickle
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from tqdm.auto import tqdm as tq


# for NNs
import torch
from torch.utils.data import Dataset, DataLoader

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
        

def preprocess_txt(data:pd.DataFrame):
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
        
        
def label_encoding(data:pd.DataFrame):
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


def load_data(do_augmentation=False, do_embedding=False):
    seed_everything(CFG['SEED']) # Seed 고정
    # load data (original data)
    data = pd.read_csv(data_path + "train.csv")
    data["img_path"] = data["img_path"].str.replace("./image/train/", CFG["org_img_train"], regex=False)
    data["overview"] = data["overview"].str.replace("<br>", "").fillna(" ")
    # text preprocessing
    preprocess_txt(data)
    # label encoding
    label_encoding(data)
    print("data loaded...")
    
    # (execute augmentation)
    if do_augmentation:
        args = aug_transform.prepare_augmentation(data)
        aug_transform.execute_img_augmentation(data, args)
        aug_transform.execute_txt_augmentation(data, args)

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
 
    # list of indices of each aug img tensor
    aug_image_indices = [f'{i}.pt' for i in range(len(augmented_img))] 
    """LOAD IMAGE (original + augmented)"""
    # original data + augmented data
    image_paths = data["img_path"].tolist() + aug_image_indices

    """LOAD TEXT (original + augmented)"""
    # (execute embedding if there is no embedding matrix for text)
    if do_embedding:
        aug_transform.execute_cnt_embedding(data, augmented_text)
    # load text vector data (original + augmented)
    texts_cntvector = torch.load(CFG["embedding_dir"])
    
    """LOAD LABELS (original + augmented)"""
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
        if not self.infer_flag:
            self.augmented_img = torch.stack(augmented_img) # list of tensors

    def __len__(self):
        return len(self.X_images)
    
    def __getitem__(self, idx):
        # transform image from str to tensor
        image    = self.X_images[idx]
        data_type= image.split(".")[1]
        if(data_type =="jpg"):
            image = aug_transform.execute_img_transformation(image)
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
def generate_dataloader(X1:list, X2:torch.tensor, y1:list, y2:list, y3:list, augmented_img:list, infer_flag=False):
    seed_everything(CFG["SEED"])
    # CFG["num_class"] = len(set(y))
    if not infer_flag:
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

    else:
        test_dataset = AugmentDataset(
            X1=X1,
            X2=X2,
            y1=y1,
            y2=y2,
            y3=y3, 
            augmented_img=augmented_img,
            infer_flag=True
        )
        loader_cfg = {
            "batch_size"    : CFG["INFER_BATCH_SIZE"], 
            "shuffle"       : False, 
            # "num_workers"   : 4
        }
        test_loader  = DataLoader(dataset=test_dataset, **loader_cfg)
        return test_loader



