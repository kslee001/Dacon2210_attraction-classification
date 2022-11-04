import os
os.chdir("..")
# for NNs
import torch
from torch.utils.data import Dataset, DataLoader

# utils
import re
import pickle
import pandas as pd
import numpy as np
import random

from tqdm.auto import tqdm as tq
import warnings
warnings.filterwarnings(action="ignore")

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# for images
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# private modules
from config import * 

# txt models
# https://github.com/monologg/KoBERT-Transformers
from transformers import BertModel
from model.tokenization_kobert import KoBertTokenizer
txt_model = BertModel.from_pretrained("monologg/kobert").cuda()
txt_model.eval()
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

def embed(txt, txt_model, tokenizer):
    # embedding
    with torch.no_grad():
        txt = tokenizer.batch_encode_plus(
            txt,
            add_special_tokens=True,
            max_length=128,
            pad_to_max_length='longest',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        # main network : 768 features
        txt = txt_model( torch.tensor(txt["input_ids"]).cuda(), torch.tensor(txt["attention_mask"]).cuda() ).pooler_output.cuda()
    return txt


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
        print("execute data augmentation...")
        # for counting
        nums = data.groupby("cat3").count()[["id"]]
        nums.columns = ["num"]
        nums = nums.reset_index()
        target_50  = []
        target_100 = []
        target_300 = []
        target_500 = []
        target_800 = []
        targets      = [target_50, target_100, target_300, target_500, target_800]
        targets_cnts = [50,        100,        300,        500,        800]
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
            elif(cnt < 800):
                targets[4].append(name)
            
        idx = 0
        augmented_text = []
        augmented_cat1 = []
        augmented_cat2 = []
        augmented_cat3 = []
        for a in tq(range(len(targets))):
            for cat_name in tq(targets[a]):
                cur_df = data[data["cat3"]==cat_name]
                img_index = cur_df.index.tolist()
                needed = targets_cnts[a]-len(img_index)
                cur_idx = 0
                while(cur_idx <= needed):
                    # sample
                    i = random.sample(img_index, 1)[0]    
                    path = cur_df.loc[i]["img_path"]
                    cur_img = cv2.imread(path)
                    cur_img = resize(cur_img)
                    cur_img = normalize(augment(cur_img))
                    cur_img = to_tensor(cur_img).float()
                    # save image
                    make_dir(CFG['aug_img_train'])
                    torch.save(cur_img, f"{CFG['aug_img_train']}{idx}.pt")

                    # text augmentation
                    cur_txt = cur_df.loc[i]["overview"].split(" ")
                    cur_txt = " ".join(random.sample(cur_txt, (len(cur_txt)//5)*4))
                    
                    augmented_text.append(cur_txt)
                    augmented_cat1.append(cur_df.loc[i]["cat1_enc"])
                    augmented_cat2.append(cur_df.loc[i]["cat2_enc"])
                    augmented_cat3.append(cur_df.loc[i]["cat3_enc"])

                    cur_idx += 1
                    idx += 1

        augmented_data = pd.DataFrame([augmented_text,
        augmented_cat1,
        augmented_cat2,
        augmented_cat3]).T
        augmented_data.columns=["overview", "cat1_enc", "cat2_enc", "cat3_enc"]  
        augmented_data.to_csv(data_path+"augmented_data.csv", index=False)    
        
        return idx

    def execute_embedding(data:pd.DataFrame, augmented_data):
        print("generate embedding...")
        texts = data["overview"].tolist() + augmented_data["overview"].tolist()

        # preprocess texts
        print("word tokenizing...")
        for i in (range(len(texts))):
            texts[i] = texts[i].replace("\n", " ").replace("br", "")
            texts[i] = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", texts[i])
            texts[i] = texts[i].split(" ")[:128]
            texts[i] = " ".join(texts[i])
            # vectorize texts
        batch = 128
        steps = len(texts)//batch + 1
        print("sentence embedding...")
        for i in (range(steps)):
            texts[i*batch:(i+1)*batch] = embed(texts[i*batch:(i+1)*batch], txt_model, tokenizer)

        torch.save(texts, CFG["embedding_dir"])
    
    # load data
    data = pd.read_csv(data_path + "train.csv")
    data["img_path"] = data["img_path"].str.replace("./image/train/", CFG["org_img_train"], regex=False)
    data["overview"] = data["overview"].fillna(" ")
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
        num_aug_data = execute_augmentation(data)
    else:
        num_aug_data = len(os.listdir(CFG["aug_img_train"]))

    # load augmented image data
    augmented_data = pd.read_csv(data_path+"augmented_data.csv")
    augmented_data["overview"] = augmented_data["overview"].fillna(" ")
    aug_images = [f'{CFG["aug_img_train"]}{i}.pt' for i in range(num_aug_data)]

    # original data + augmented data
    image_paths = data["img_path"].tolist() + aug_images

    # (execute embedding)
    if do_embedding:
        execute_embedding(data, augmented_data)

    # load text data (original + augmented)
    texts = torch.load(CFG["embedding_dir"])
    
    # y1, y2, y3 (labels)
    y1 = data["cat1_enc"].tolist() + augmented_data["cat1_enc"].tolist()
    y2 = data["cat2_enc"].tolist() + augmented_data["cat2_enc"].tolist()
    y3 = data["cat3_enc"].tolist() + augmented_data["cat3_enc"].tolist()


    return image_paths, texts, y1, y2, y3



class AugmentDataset(Dataset):
    def __init__(self, X1:list, X2:list, y1:list, y2:list, y3:list, infer_flag=False):
        self.X_images   = X1
        self.X_texts    = X2
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.infer_flag = infer_flag

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
            image = torch.load(image)

        # text
        text = self.X_texts[idx]

        if self.infer_flag :
            return image, text
        else:
            # load label
            y1 = torch.tensor(self.y1[idx])
            y2 = torch.tensor(self.y2[idx])
            y3 = torch.tensor(self.y3[idx])
            y = torch.stack([y1,y2,y3]).T
            return image, text, y
        

def generate_dataloader(X1:list, X2:list, y1:list, y2:list, y3:list):
    random.seed(CFG["SEED"])
    # CFG["num_class"] = len(set(y))

    X_img_train, X_img_valid, X_text_train, X_text_valid, Y1_train, Y1_valid, Y2_train, Y2_valid, Y3_train, Y3_valid = train_test_split(
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
        X2=X_text_train,
        y1=Y1_train,
        y2=Y2_train,
        y3=Y3_train, 
        infer_flag=False
    )
    valid_dataset = AugmentDataset(
        X1=X_img_valid,
        X2=X_text_valid,
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

