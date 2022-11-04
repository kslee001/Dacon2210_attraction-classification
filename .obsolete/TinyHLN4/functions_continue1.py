import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_ as clip_grad

import os
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
# from tqdm.auto import tqdm as tq
from config import *



def make_dir(directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory)



def train(model, optimizers:list, criterions:list, warm_ups:list, schedulers:list, train_loader, val_loader, device=CFG["device"]):
    best_score = 0
    best_model = None
    
    # first scheduler
    for epoch in range(1, CFG["EPOCHS"]+1):
        model.train()
        train_loss = []
        train_loss1 = []
        train_loss2 = []
        train_loss3 = []
        train_loss12 = []
        train_loss23 = []

        for x_img, x_text, y in (iter(train_loader)):
            img  = x_img.float().cuda()
            text = x_text.cuda()
            
            for optimizer in optimizers:
                optimizer.zero_grad()

            model_pred = model(img, text)
            loss1 = criterions[0](model_pred[0].cuda(), y[:,0].cuda())
            loss2 = criterions[1](model_pred[1].cuda(), y[:,1].cuda())
            loss3 = criterions[2](model_pred[2].cuda(), y[:,2].cuda())  
            loss = loss1+loss2+loss3
            loss12 = loss1.item()+loss2.item()
            loss23 = loss2.item()+loss3.item()

            loss.backward()

            # gradient clipping
            clip_grad(model.parameters(), CFG["max_norm"])
            optimizers[0].step()
            
            # FREEZE backpropagtion for category 2 and 3 until classification for cat 1 stabilized
            # if(epoch >=8 | epoch < 6):
            optimizers[1].step()
            # if(epoch >=12 | epoch < 6):
            optimizers[2].step()
                
            # for optimizer in optimizers:
            #     optimizer.step()
            train_loss.append(loss.item())
            train_loss1.append(loss1.item())
            train_loss2.append(loss2.item())
            train_loss3.append(loss3.item())
            train_loss12.append(loss12)
            train_loss23.append(loss23)


        tr_loss = np.mean(train_loss)
        tr_loss1 = np.mean(train_loss1)
        tr_loss2 = np.mean(train_loss2)
        tr_loss3 = np.mean(train_loss3)

        val_loss, val_loss1, val_loss2, val_loss3, val_loss12, val_loss23, val_score = validation(model, criterions, val_loader)
        
        val_losses = [val_loss, val_loss1, val_loss2, val_loss3]
        print(f'Epoch [{epoch}]')
        print(f'Train Loss1 : [{tr_loss1:.5f}] | Val Loss1 : [{val_loss1:.5f}]')
        print(f'Train Loss2 : [{tr_loss2:.5f}] | Val Loss2 : [{val_loss2:.5f}]')
        print(f'Train Loss3 : [{tr_loss3:.5f}] | Val Loss3 : [{val_loss3:.5f}]')
        print(f'Train Loss  : [{tr_loss:.5f}] | Val Loss : [{val_loss:.5f}]')
        print(f'Val Score   : [{val_score:.5f}]')
        

        schedulers[0].step(val_loss)    # for main network of image
        # FREEZE backpropagtion for category 2 and 3 until classification for cat 1 stabilized
        # if(epoch >=8):                 # FREEZE scheduler step for val loss 2 and val loss 3
        schedulers[1].step(val_loss2)   # for cat 2
        # if(epoch >=12):
        schedulers[2].step(val_loss3)   # for cat 3


        if best_score < val_score:
            best_score = val_score
            best_model = model     

    
    return best_model

def score_function(real, pred):
    return f1_score(real, pred, average="weighted")

def validation(model, criterions, val_loader, device=CFG["device"]):
    model.eval()
    
    model_preds = []
    true_labels = []
    
    val_loss = []
    val_loss1 = []
    val_loss2 = []
    val_loss3 = []
    val_loss12 = []
    val_loss23 = []
    
    with torch.no_grad():
        for x_img, x_text, y in (iter(val_loader)):
            img  = x_img.float().cuda()
            text = text = x_text

            model_pred = model(img, text)
            loss1 = criterions[0](model_pred[0].cuda(), y[:,0].cuda())
            loss2 = criterions[1](model_pred[1].cuda(), y[:,1].cuda())
            loss3 = criterions[2](model_pred[2].cuda(), y[:,2].cuda())  
            loss  = loss1 + loss2 + loss3
            loss12 = loss1.item() + loss2.item()
            loss23 = loss2.item() + loss3.item()

            val_loss.append(loss.item())
            val_loss1.append(loss1.item())
            val_loss2.append(loss2.item())
            val_loss3.append(loss3.item())
            val_loss12.append(loss12)
            val_loss23.append(loss23)

            model_preds += model_pred[2].argmax(1).detach().cpu().numpy().tolist()
            true_labels += y[:,2].cpu().numpy().tolist()
        
    test_weighted_f1 = score_function(true_labels, model_preds)
    return np.mean(val_loss), np.mean(val_loss1), np.mean(val_loss2), np.mean(val_loss3), np.mean(val_loss12), np.mean(val_loss23), test_weighted_f1



def save_model(best_model):
    torch.save(
        best_model, 
        output_folder + "/" + model_name
    )
    torch.save(
        best_model.state_dict(),
        output_folder + "/" + model_states_name
    )


def save_configs():
    # save configs
    with open(output_folder + "/" + "configs.txt", "w") as f:
        
        # CFG
        f.write("CFG = {\n")
        for name, val in CFG.items():
            if((type(val)==int) | (type(val)==float)):
                cur = "'"+str(name)+"'"+ " : " + str(val) + ",\n"
            else:
                cur = "'"+str(name)+"'"+ " : " + "'" + str(val) + "'" + ",\n"                
            f.write(cur)
        f.write("}\n\n")

        # scheduler args
        f.write("scheduler_args = {\n")
        for name, val in scheduler_args.items():
            if((type(val)==int) | (type(val)==float)):
                cur = "'"+str(name)+"'"+ " : " + str(val) + ",\n"
            else:
                cur = "'"+str(name)+"'"+ " : " + "'" + str(val) + "'" + ",\n"                
            f.write(cur)
        f.write("}")


def count_parameters(model, trainable=False):
    if(trainable):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())
