# private modules
from .denoise import * 
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
from transformers import AutoTokenizer
current_working_dir = os.getcwd()
os.chdir(CFG["text_eda_dir"])
from .KorEDA import eda
os.chdir(current_working_dir)


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
        # cur_txt = [re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z.,]", "", t) for t in cur_txt]
        cur_txt = [t.lstrip().rstrip() for t in cur_txt]
        cur_txt[:] = list(filter(None, cur_txt))
        cur_txt = " ".join(cur_txt)
        data.loc[i]["overview"] = cur_txt    
        

"""
AUGMENTATION & TRANSFORMATION
"""
# sub functions
def resize(x):
    H, W, C = x.shape
    if H>W:
        tf = A.Compose([
            A.Resize(CFG["IMG_RESIZE"]*H//W, CFG["IMG_RESIZE"]),
        ])
    else:
        tf = A.Compose([
            A.Resize(CFG["IMG_RESIZE"], CFG["IMG_RESIZE"]*W//H),
        ])
    return tf(image=x)["image"]

def normalize(x):
    return np.round(x/255, 4)

def transform(x:np.array):
    seed_everything(CFG['SEED'])
    # soft transformation
    tf_sequence = A.Compose([
        A.CenterCrop(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
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


def execute_augmentation(data:pd.DataFrame):
    # make directory
    make_dir(f"{data_path}augmented/{folder_name}{seed_number}")
    # fix random seed
    seed_everything(CFG['SEED'])
    
    print("execute data augmentation...(it takes about 10~15 minutes to complete)")
    
    augmented_img = []
    augmented_text = []
    augmented_cat1 = []
    augmented_cat2 = []
    augmented_cat3 = []
    
    # step 1. 50개, 120개, 250개 이하 이미지/텍스트 augmentation
    nums = data.groupby("cat3").count()[["id"]]
    nums.columns = ["num"]
    nums = nums.reset_index()
    target_50   = []
    target_120  = []
    target_250  = []
    targets      = [target_50, target_120, target_250]
    targets_cnts = [50,        120,        250]
    for i in range(len(nums)):
        cur = nums.iloc[i]
        cnt = cur["num"]; name = cur["cat3"]
        if(cnt < 50):
            targets[0].append(name)
        elif(cnt < 120):
            targets[1].append(name)
        elif(cnt < 250):
            targets[2].append(name)
            
    for i in (range(len(targets))):
        for cat_name in targets[i]:
            # 50~250개 이하 카테고리 선택
            cur_df    = data[data["cat3"]==cat_name]
            img_index = cur_df.index.tolist()
            needed    = targets_cnts[i] - len(img_index)
            cur_idx   = 0
            new_random_idx = 0
            while(cur_idx <=needed):
                new_random_idx +=1
                # sampling
                new_aug_seed = CFG["SEED"] + new_random_idx # new random seed
                random.seed(new_aug_seed)
                sample_index = random.sample(img_index, 1)[0]
                
                # labels
                cur_cat1 = cur_df.loc[sample_index]["cat1_enc"]
                cur_cat2 = cur_df.loc[sample_index]["cat2_enc"]
                cur_cat3 = cur_df.loc[sample_index]["cat3_enc"]
                augmented_cat1.append(cur_cat1)
                augmented_cat2.append(cur_cat2)
                augmented_cat3.append(cur_cat3)
                
                # img augmentation
                path    = cur_df.loc[sample_index]["img_path"]
                cur_img = cv2.imread(path)
                cur_img = resize(cur_img)
                cur_img = normalize(augment(cur_img))
                cur_img = to_tensor(cur_img).float()
                augmented_img.append(cur_img)
                
                # text augmentation
                cur_txt    = cur_df.loc[sample_index]["overview"]
                candidates = eda.EDA(cur_txt, random_state=new_aug_seed)
                cur_txt    = random.choice(candidates)
                augmented_text.append(cur_txt)
                
                cur_idx +=1
                
    # step 2. 한식, 야영지 제외 전체 이미지 텍스트 augmentation -> 7500개만
    samples = random.sample(data.index.tolist(), 10000)
    new_random_idx = 0
    for i in (range(len(samples))):
        new_random_idx +=1
        new_aug_seed = (CFG["SEED"] + new_random_idx)
        random.seed(new_aug_seed)
        
        cur = data.loc[i]
        cur_cat1 = cur["cat1_enc"]
        cur_cat2 = cur["cat2_enc"]
        cur_cat3 = cur["cat3_enc"]
        cur_img_path  = cur["img_path"]
        cur_txt  = cur["overview"]
        
        cur_cat3_str = cur["cat3"]
        
        if (cur_cat3_str == "한식") | (cur_cat3_str == "야영장,오토캠핑장"):
            continue
        else:
            # image
            cur_img = cv2.imread(cur_img_path)
            cur_img = resize(cur_img)
            cur_img = normalize(augment(cur_img))
            cur_img = to_tensor(cur_img).float()
            augmented_img.append(cur_img)     

            # text
            candidates = eda.EDA(cur_txt, random_state=new_aug_seed)
            cur_txt    = random.choice(candidates)
            augmented_text.append(cur_txt)
            
            # labels
            augmented_cat1.append(cur_cat1)
            augmented_cat2.append(cur_cat2)
            augmented_cat3.append(cur_cat3)

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
        

def execute_img_transformation(image:str):
    image = cv2.imread(image)
    image = resize(image)
    image = transform(image)
    image = normalize(image)
    image = to_tensor(image)
    return image
        
 
"""
DATA LOADING
"""
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
           
           
def denoise_data(data):
    data["error"] = False
    data["check"] = False
    data["valid"] = True
    cat3 = data.cat3.unique().tolist()
    
    errors = []
    
    # for restaurants
    for idx in range(len(restaurant_target)):
    #     print(restaurant_target[idx])
        cur_target = data[data["cat3"]==restaurant_target[idx]]
        cur_error_words = restaurant_error_words[idx]
        cur_words_off = restaurant_words_off[idx]

        # error check
        for i in range(len(cur_error_words)):
            cur_target["error"] += cur_target["overview"].str.contains(cur_error_words[i])
        cur_errors = cur_target[cur_target["error"]==True]

        # False positive check
        for i in range(len(cur_words_off)):
            cur_errors["error"] = np.where(cur_errors["overview"].str.contains(cur_words_off[i]), False, cur_errors["error"])
        cur_errors = cur_errors[cur_errors["error"]==True]
        errors.extend(cur_errors.id.tolist())

    # for etc    
    for idx in range(len(etc_target)):
    #     print(etc_target[idx])
        cur_target = data[data["cat3"]==etc_target[idx]]
        cur_words_off = etc_error_words_off[idx]
        for i in range(len(cur_words_off)):
            cur_target["check"] += cur_target.overview.str.contains(cur_words_off[i])
        cur_errors = cur_target[cur_target["check"]==False].id.tolist()
        errors.extend(cur_errors)
        
    # based on inference result on train data (low accuracy)
    for idx in range(len(infer_result_based_target)):
    #     print(infer_result_based_target[idx])
        cur_target = data[data["cat3"]==infer_result_based_target[idx]]
        cur_error_words = infer_result_based_error_words[idx]
        cur_words_off = infer_result_based_words_off[idx]

        # error check
        for i in range(len(cur_error_words)):
            cur_target["error"] += cur_target["overview"].str.contains(cur_error_words[i])
        cur_errors = cur_target[cur_target["error"]==True]

        # False positive check
        for i in range(len(cur_words_off)):
            cur_errors["error"] = np.where(cur_errors["overview"].str.contains(cur_words_off[i]), False, cur_errors["error"])
        cur_errors = cur_errors[cur_errors["error"]==True]
        errors.extend(cur_errors.id.tolist())

    # delete data based on inference result
    errors.extend(delete_target)    
    data["valid"] = np.where(data["id"].isin(errors), False, True)
    data = data[data["valid"]]

    # re-label data
    for i in range(len(rename_target)):
        data.loc[rename_target[i], 'cat3'] = rename_label[i]
    # reset index
    data = data.reset_index()
    data = data.drop(columns = ["index", "error", "check", "valid"])



def load_data(do_augmentation=False):
    seed_everything(CFG['SEED']) # Seed 고정
    """1. load original data (original data)"""
    data = pd.read_csv(data_path + "train.csv")
    denoise_data(data)
    
    # text preprocessing
    preprocess_txt(data)
    
    # label encoding
    label_encoding(data)

    data["img_path"] = data["img_path"].str.replace("./image/train/", CFG["org_img_train"], regex=False)
    
    """2. (generate) load augmented data"""
    if(do_augmentation):
        execute_augmentation(data)

    # load augmented imgs  ( List of tensors )
    with open(CFG["augimg_dir"], "rb") as f:
        augmented_img = pickle.load(f)
        # list of indices of each aug img tensor
        aug_image_indices = [f'{i}.pt' for i in range(len(augmented_img))] 
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
        
    """3. concat original data and augmented data"""
    # image
    image_paths = data["img_path"].tolist() + aug_image_indices
    # text
    texts = data["overview"].tolist() + augmented_text
    texts = np.array(texts)
    # y1, y2, y3 (labels)
    y1 = data["cat1_enc"].tolist() + augmented_cat1
    y2 = data["cat2_enc"].tolist() + augmented_cat2
    y3 = data["cat3_enc"].tolist() + augmented_cat3

    return image_paths, texts, y1, y2, y3, augmented_img


class AugmentDataset(Dataset):   # X1 : img path  / X2 : txt vector / X3 : txt raw
    def __init__(self, X1:list, X2, y1:list, y2:list, y3:list, augmented_img:torch.tensor, infer_flag=False):
        self.X_images = X1
        self.X_texts  = X2
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.infer_flag = infer_flag
        self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
        self.augmented_img = augmented_img

    def __len__(self):
        return len(self.X_images)
    
    def __getitem__(self, idx):
        # transform image from str to tensor
        image    = self.X_images[idx]
        data_type= image.split(".")[1]
        if(data_type =="jpg"):
            image = execute_img_transformation(image)
        elif(data_type =="pt"):
            image = self.augmented_img[int(self.X_images[idx].split(".pt")[0])]

        text  = self.X_texts[idx]
        text = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=256,
            return_token_type_ids=False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask=True,
            return_tensors='pt'
        )   
        text_input_ids = text["input_ids"].squeeze(0)
        text_attention_mask = text["attention_mask"].squeeze(0)
        
        if self.infer_flag :
            return image, text_input_ids, text_attention_mask
        else:
            # load label
            y1 = torch.tensor(self.y1[idx])
            y2 = torch.tensor(self.y2[idx])
            y3 = torch.tensor(self.y3[idx])
            y = torch.stack([y1,y2,y3]).T
            return image, text_input_ids, text_attention_mask, y
        
        
                        # X1 : img path  / X2 : txt vector / X3 : txt raw
def generate_dataloader(X1:list, X2:torch.tensor, y1:list, y2:list, y3:list, augmented_img:torch.tensor, infer_flag=False):
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
            augmented_img=augmented_img,
            infer_flag=False
        )
        valid_dataset = AugmentDataset(
            X1=X_img_valid,
            X2=X_text_vector_valid,
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
            augmented_img=[],
            infer_flag=True
        )
        loader_cfg = {
            "batch_size"    : CFG["INFER_BATCH_SIZE"], 
            "shuffle"       : False, 
            # "num_workers"   : 4
        }
        test_loader  = DataLoader(dataset=test_dataset, **loader_cfg)
        return test_loader



