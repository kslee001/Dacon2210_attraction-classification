from config import *
from model.hierarchical_loss import HierarchicalLossNetwork
import functions.preprocessing as preprocessing
import functions.train as train
from model.models import *

import torch
import pandas as pd
import numpy as np
import time as t
from tqdm.auto import tqdm as tq

from sklearn.preprocessing import LabelEncoder


def __inference(model, test_loader, encoder, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    
    with torch.no_grad():
        for img, text_input_ids, text_attention_mask in tq(iter(test_loader)):
            img = img.float().to(device)
            text_input_ids, text_attention_mask = text_input_ids.cuda(), text_attention_mask.cuda()
            _, _, model_pred = model(img, text_input_ids, text_attention_mask)
            pred =  model_pred.detach().cpu().numpy().argmax(1).tolist()
            # print(pred)
            pred = encoder.inverse_transform(pred)
            # print(pred)
            model_preds.extend(pred)

    return model_preds

def main():
    print("job started -- inference")
    print(f"target seeds : {inference_config['seeds']}\n")
    start_time = t.time()
    
    # Encode labels
    enc = LabelEncoder()
    data = pd.read_csv(train_data)
    encoded = enc.fit_transform(data["cat3"])
    

    # inference
    data = pd.read_csv(test_data)
    data["img_path"] = data["img_path"].str.replace("./image/test/", CFG["org_img_test"], regex=False)

    X1 = data["img_path"].tolist()
    X2 = data["overview"]
    
    test_loader = preprocessing.generate_dataloader(X1, X2, [], [], [], [], True)
    
    i = 0
    model = torch.load("/home/gyuseonglee/dacon/workplace/output/MR42aug42/MR42aug").cuda()
    # model.load_state_dict(torch.load(inference_config["weight_dir"]))
    model.eval()
    model_preds = __inference(model, test_loader, enc, device="cuda")
    del model
    torch.cuda.empty_cache()
    
    submit = pd.read_csv(submit_file)
    submit["cat3"] = model_preds
    submit.to_csv(submit_dir, index=False)

if __name__ =="__main__":
    main()