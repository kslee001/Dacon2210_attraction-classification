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
    return data
        

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


def prepare_augmentation(data:pd.DataFrame):
    seed_everything(CFG['SEED'])
    make_dir(f"{data_path}augmented/{folder_name}{seed_number}")
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
    
    return args


def execute_augmentation(data:pd.DataFrame, args):
    seed_everything(CFG['SEED'])
    targets        = args["targets"]
    targets_cnts   = args["targets_cnts"]
    augmented_cat1 = args["augmented_cat1"]
    augmented_cat2 = args["augmented_cat2"]
    augmented_cat3 = args["augmented_cat3"]
    
    idx = 0
    new_random_idx = 0 
    augmented_img = []
    augmented_text = []

    for a in (range(len(targets))):
        for cat_name in (targets[a]):
            cur_df = data[data["cat3"]==cat_name]
            img_index = cur_df.index.tolist()
            needed = targets_cnts[a]-len(img_index)
            cur_idx = 0
            while(cur_idx <= needed):
                new_random_idx +=1
                # 1. sampling
                txt_aug_seed = random.seed(CFG["SEED"] + new_random_idx) # random + random
                i = random.sample(img_index, 1)[0]    
                
                # 2. image augmentation 
                path = cur_df.loc[i]["img_path"]
                cur_img = cv2.imread(path)
                cur_img = resize(cur_img)
                cur_img = normalize(augment(cur_img))
                cur_img = to_tensor(cur_img).float()
                augmented_img.append(cur_img)
                
                # 3. text augmentation
                curtxt = cur_df.loc[i]["overview"]
                augmented_txt_list = eda.EDA(curtxt, random_state=txt_aug_seed)
                cur_txt = random.choice(augmented_txt_list)
                augmented_text.append(cur_txt)
                
                cur_idx += 1
                idx += 1
            random.seed(CFG["SEED"])
    
    # save augmented imgs                
    with open(CFG["augimg_dir"], "wb") as f:
        pickle.dump(augmented_img, f)
        
    # save augmented text 
    with open(CFG["augtxt_dir"], "wb") as f:
        pickle.dump(augmented_text, f)  
        

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
    restaurant_target = ["한식", "중식", "일식", "서양식"]
    restaurant_error_words = [
        [
        "서양식", "양식전문점", "양식전문", "양식 전문점", "양식 전문", "양식요리", "양식 요리", "서양요리", "서양 요리", "서양음식", "서양 음식",
        "일식", "일식전문점", "일식전문", "일식 전문점", "일식 전문", "일식요리", "일식 요리", "일본요리", "일본 요리", "일본음식", "일본 음식",
        "중식", "중식전문점", "중식전문", "중식 전문점", "중식 전문", "중식요리", "중식 요리", "중국요리", "중국 요리", "중국음식", "중국 음식"
        ],   
        [
        "서양식", "양식전문점", "양식전문", "양식 전문점", "양식 전문", "양식요리", "양식 요리", "서양요리", "서양 요리", "서양음식", "서양 음식",
        "일식", "일식전문점", "일식전문", "일식 전문점", "일식 전문", "일식요리", "일식 요리", "일본요리", "일본 요리", "일본음식", "일본 음식",
        "한식", "한식전문점", "한식전문", "한식 전문점", "한식 전문", "한식요리", "한식 요리", "한국요리", "한국 요리", "한국음식", "한국 음식"
        ],
        [
        "서양식", "양식전문점", "양식전문", "양식 전문점", "양식 전문", "양식요리", "양식 요리", "서양요리", "서양 요리", "서양음식", "서양 음식",
        "중식", "중식전문점", "중식전문", "중식 전문점", "중식 전문", "중식요리", "중식 요리", "중국요리", "중국 요리", "중국음식", "중국 음식",
        "한식", "한식전문점", "한식전문", "한식 전문점", "한식 전문", "한식요리", "한식 요리", "한국요리", "한국 요리", "한국음식", "한국 음식"
        ],
        [
        "일식", "일식전문점", "일식전문", "일식 전문점", "일식 전문", "일식요리", "일식 요리", "일본요리", "일본 요리", "일본음식", "일본 음식",
        "중식", "중식전문점", "중식전문", "중식 전문점", "중식 전문", "중식요리", "중식 요리", "중국요리", "중국 요리", "중국음식", "중국 음식",
        "한식", "한식전문점", "한식전문", "한식 전문점", "한식 전문", "한식요리", "한식 요리", "한국요리", "한국 요리", "한국음식", "한국 음식"
        ]
    ]
    restaurant_words_off = [
        ["한식", "보양식"],
        ["중식"],
        ["일식"],
        ["서양식", "스테이크"]
    ]

    etc_target = [ "바/까페", "야영장,오토캠핑장", "모텔"]
    etc_error_words_off = [
        [
            "카페", "까페", "찻집", "바", 
            "bar", "cafe", "커피", "쿠키", 
        "빵", "베이커리", "디저트","도넛", 
        "요거트", "전통차", "음료", "컨피", 
        "아메리카노", "카푸치노", "차 한잔", "차를 한잔", "녹차", "빙수"
        ],
        ["야영장", "캠핑장", "캠핑", "캠프", "야영", "텐트","글램핑", "카라반", "휴양림"],
        ["모텔", "호텔", "숙박", "게스트", "객실", "온천", 
                    "무인텔", "호스텔", "MOTEL", "motel", "hotel","HOTEL", "스키텔", "골프텔", 
                    "하우스", "방 안"
        ]
    ]
    data["error"] = False
    data["check"] = False
    data["valid"] = True
    errors = []
    for idx in range(len(restaurant_target)):
        cur_target = data[data["cat3"]==restaurant_target[idx]]
        cur_error_words = restaurant_error_words[idx]
        cur_words_off = restaurant_words_off[idx]

        for i in range(len(cur_error_words)):
            cur_target["error"] += cur_target.overview.str.contains(cur_error_words[i])
        cur_errors = cur_target[cur_target["error"]].id.tolist()
        errors.extend(cur_errors)

    for idx in range(len(etc_target)):
        cur_target = data[data["cat3"]==etc_target[idx]]
        cur_words_off = etc_error_words_off[idx]
        for i in range(len(cur_words_off)):
            cur_target["check"] += cur_target.overview.str.contains(cur_words_off[i])
        cur_errors = cur_target[cur_target["check"]==False].id.tolist()
        errors.extend(cur_errors)     
    data["valid"] = np.where(data["id"].isin(errors), False, True)
    data = data[data["valid"]].reset_index(); del data["index"]
    return data

def load_data(do_augmentation=False):
    seed_everything(CFG['SEED']) # Seed 고정
    """1. load original data (original data)"""
    data = pd.read_csv(data_path + "train.csv")
    data["img_path"] = data["img_path"].str.replace("./image/train/", CFG["org_img_train"], regex=False)
    data = denoise_data(data)
    # text preprocessing
    data = preprocess_txt(data)
    # label encoding
    label_encoding(data)

    """2. (generate) load augmented data"""
    if(do_augmentation):
        args = prepare_augmentation(data)
        execute_augmentation(data, args)

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



