import torch
import pandas as pd
import numpy as np
import time as t
from tqdm.auto import tqdm as tq

from sklearn.preprocessing import LabelEncoder

import preprocessing
from model.models import *

def __inference(model, test_loader, encoder, device):
    model.to(device)
    model.eval()
    
    model_preds = []
    model_preds_probs = []
    
    with torch.no_grad():
        for img, text_input_ids, text_attention_mask in tq(iter(test_loader)):
            img = img.float().to(device)
            text_input_ids, text_attention_mask = text_input_ids.cuda(), text_attention_mask.cuda()
            _, _, model_pred = model(img, text_input_ids, text_attention_mask)
            pred_probs = model_pred.detach().cpu().numpy().tolist()
            pred =  model_pred.detach().cpu().numpy().argmax(1).tolist()
            # pred = encoder.inverse_transform(pred)
            model_preds.extend(pred)
            model_preds_probs.extend(pred_probs)

    return model_preds, np.array(model_preds_probs)


def most_freq(preds, target_data_index):
    key, val = np.unique(preds[:,target_data_index], return_counts=True)
    cnt = dict(zip(key, val))
    freq = sorted(cnt.items(), reverse=True, key=lambda x:x[1])[0][0]
    return freq


def main():
    print("job started -- inference")
    start_time = t.time()
    
    # Encode labels
    enc = LabelEncoder()
    data = pd.read_csv("/home/gyuseonglee/dacon/data/train.csv")
    encoded = enc.fit_transform(data["cat3"])
    
    # inference
    data = pd.read_csv("/home/gyuseonglee/dacon/data/test.csv")
    data["img_path"] = data["img_path"].str.replace("./image/test/", "/home/gyuseonglee/dacon/data/original/image/test/", regex=False)

    X1 = data["img_path"].tolist()
    X2 = data["overview"]
    
    test_loader = preprocessing.generate_dataloader(X1, X2, [], [], [], [], True)
    

    seeds = [1203, 33, 364, 42, 95, 210]#, 317, 918, 22364]    
    
    all_model_preds = []
    all_model_preds_probs = []
    for cur_seed in seeds:
        model = torch.load(f"/home/gyuseonglee/dacon/final/MR{cur_seed}aug").cuda()
        # model.load_state_dict(torch.load(inference_config["weight_dir"]))
        model.eval()
        model_preds, model_preds_probs = __inference(model, test_loader, enc, device="cuda")
        del model
        torch.cuda.empty_cache()
        all_model_preds.append(model_preds)
        all_model_preds_probs.append(model_preds_probs)    
    
    # hard voting
    all_model_preds = np.array(all_model_preds)
    num_data = len(data)
    res = np.array([most_freq(all_model_preds, idx) for idx in range(num_data)])
    res = enc.inverse_transform(res)
    submit = pd.read_csv("/home/gyuseonglee/dacon/data/sample_submission.csv")
    submit["cat3"] = res
    submit.to_csv("/home/gyuseonglee/dacon/final/hardvoting_pred_partial.csv", index=False)

if __name__ =="__main__":
    main()