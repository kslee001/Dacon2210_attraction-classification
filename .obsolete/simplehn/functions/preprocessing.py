# private modules
from config import * 

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


# for images
import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# for texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import ElectraTokenizer

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
        cur_txt = [t.lstrip().rstrip() for t in cur_txt]
        cur_txt[:] = list(filter(None, cur_txt))
        cur_txt = " ".join(cur_txt)
        data.loc[i]["overview"] = cur_txt
        
        
def execute_embedding(data:pd.DataFrame):
    print("execute embedding... (may take 20~30 minutes to complete.)")
    tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator')
    tagged_data = [
        TaggedDocument(
            words=tokenizer.tokenize(txt[:1024]), 
            tags=[str(i)]
        ) for i, txt in enumerate(data["overview"])
    ]
    vec_size = CFG["embedding_dim"]
    alpha = 0.025
    model = Doc2Vec(tagged_data,
                    vector_size=vec_size,
                    alpha=alpha,
                    min_alpha=0.00025,
                    min_count=1,
                    seed = CFG["SEED"],
                    dm =1,
                    workers=4,
                    dbow_words=1,
                    dm_concat=1,
                    )
    model.save(CFG["embedding_dir"])


def execute_img_transformation(image:str):
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

    def to_tensor(x):
        return A.pytorch.ToTensorV2()(image=x)["image"].float()
    
    image = cv2.imread(image)
    image = resize(image)
    image = transform(image)
    image = normalize(image)
    image = to_tensor(image)
    return image
        
        
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


def load_data(do_embedding=False):
    seed_everything(CFG['SEED']) # Seed 고정
    # load data (original data)
    data = pd.read_csv(data_path + "train.csv")
    data["img_path"] = data["img_path"].str.replace("./image/train/", CFG["org_img_train"], regex=False)
    # text preprocessing
    preprocess_txt(data)
    # label encoding
    label_encoding(data)
    print("data loaded...")
 
    image_paths = data["img_path"].tolist()
    if do_embedding:
        execute_embedding(data)
        
    txt_model = Doc2Vec.load(CFG["embedding_dir"])
    vectorizer = txt_model.infer_vector
    
    texts_vector = []
    for txt in data["overview"].tolist():
        embedding = vectorizer(txt.split("."))
        texts_vector.append(embedding)
    texts_vector = torch.tensor(texts_vector)

    """LOAD LABELS"""
    # y1, y2, y3 (labels)
    y1 = data["cat1_enc"].tolist()
    y2 = data["cat2_enc"].tolist()
    y3 = data["cat3_enc"].tolist()

    # imge path, text count vector, raw text (~5 lines)
    return image_paths, texts_vector, y1, y2, y3

class AugmentDataset(Dataset):   # X1 : img path  / X2 : txt vector / X3 : txt raw
    def __init__(self, X1:list, X2:torch.tensor, y1:list, y2:list, y3:list, infer_flag=False):
        self.X_images       = X1
        self.X_texts_vector = X2
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.infer_flag = infer_flag

    def __len__(self):
        return len(self.X_images)
    
    def __getitem__(self, idx):
        # transform image from str to tensor
        image    = self.X_images[idx]
        data_type= image.split(".")[1]
        image = execute_img_transformation(image)

        # text
        text_vector = self.X_texts_vector[idx]
        
        if self.infer_flag :
            return image, text_vector
        else:
            # load label
            y1 = torch.tensor(self.y1[idx])
            y2 = torch.tensor(self.y2[idx])
            y3 = torch.tensor(self.y3[idx])
            y = torch.stack([y1,y2,y3]).T
            return image, text_vector, y
        
        
                        # X1 : img path  / X2 : txt vector / X3 : txt raw
def generate_dataloader(X1:list, X2:torch.tensor, y1:list, y2:list, y3:list, infer_flag=False):
    seed_everything(CFG["SEED"])

    # CFG["num_class"] = len(set(y))
    if not infer_flag:
        X_img_train, X_img_valid, X_text_vector_train, X_text_vector_valid, Y1_train, Y1_valid, Y2_train, Y2_valid, Y3_train, Y3_valid = train_test_split(
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
            X2=X_text_vector_train,
            y1=Y1_train,
            y2=Y2_train,
            y3=Y3_train, 
            infer_flag=False
        )
        valid_dataset = AugmentDataset(
            X1=X_img_valid,
            X2=X_text_vector_valid,
            y1=Y1_valid,
            y2=Y2_valid,
            y3=Y3_valid, 
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
            infer_flag=True
        )
        loader_cfg = {
            "batch_size"    : CFG["INFER_BATCH_SIZE"], 
            "shuffle"       : False, 
            # "num_workers"   : 4
        }
        test_loader  = DataLoader(dataset=test_dataset, **loader_cfg)
        return test_loader



