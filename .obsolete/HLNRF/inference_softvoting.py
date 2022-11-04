from config import *
from functions.aug_transform import *
from model.hierarchical_loss import HierarchicalLossNetwork
import functions.preprocessing as preprocessing
import functions.train as train
from model.models import *

import torch
import pandas as pd
import time as t
from tqdm.auto import tqdm as tq

from sklearn.preprocessing import LabelEncoder


def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for img, text in tq(iter(test_loader)):
            img = img.float().to(device)
            text = text.to(device)
            _, _, model_pred = model(img, text)
            # model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            model_preds += model_pred.detach().cpu().numpy().tolist()
    return model_preds


if __name__ =="__main__":
    print("job started -- inference")
    print(f"target seeds : {inference_config['seeds']}\n")
    start_time = t.time()
    
    # Encode labels
    enc = LabelEncoder()
    train_data = pd.read_csv(train_data)
    encoded = enc.fit_transform(train_data["cat3"])
    

    # inference
    data = pd.read_csv(test_data)
    data["img_path"] = data["img_path"].str.replace("./image/test/", CFG["org_img_test"], regex=False)

    X1 = data["img_path"].tolist()
    
    if DATA["do_embedding"]:
        preprocessing.__aug_transform(data, infer=True)
    X2 = torch.load(CFG["infer_embedding_dir"])
    
    test_loader = preprocessing.generate_dataloader(X1, X2, [], [], [], [], True)
    
    model = HLN().cuda()
    model = torch.nn.DataParallel(model, device_ids = [0,1])
    model.load_state_dict(torch.load(
        inference_config["weight_dir1"] 
        + str(inference_config["seeds"][0])
        + inference_config["weight_dir2"] ))
    model.eval()
    
    pred_list = []    
    for i in range(len(inference_config["seeds"])):
        pred = inference(model, test_loader, device="cuda")
        pred = enc.inverse_transform(pred)
        pred_list.append(pred)
        
    
    
    submit = pd.read_csv(submit_dir)

    submit["cat3"] = pred
    submit.to_csv(submit_dir)
