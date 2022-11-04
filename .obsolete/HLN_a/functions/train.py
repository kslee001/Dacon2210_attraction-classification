import os
os.chdir("..")
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


def calculate_accuracy(predictions, labels):
    '''Calculates the accuracy of the prediction.
    '''
    num_data = labels.size()[0]
    predicted = torch.argmax(predictions, dim=1)

    correct_pred = torch.sum(predicted == labels)
    accuracy = correct_pred*(100/num_data)

    return accuracy.item()


def train(model, optimizer, criterion, warm_up, scheduler, train_loader, val_loader, device=CFG["device"]):
    best_score = 0
    best_model = None
    
    # first scheduler
    for epoch in range(1, CFG["EPOCHS"]+1):
        model.train()
        train_loss = []

        superclass_accuracy = []
        subclass1_accuracy = []
        subclass2_accuracy = []

        test_epoch_loss = []
        test_epoch_superclass_accuracy = []
        test_epoch_subclass_accuracy = []

        for x_img, x_text, y in (iter(train_loader)):
            img  = x_img.float().cuda()
            text = x_text.cuda()
            y1   = y[:, 0].cuda()
            y2   = y[:, 1].cuda()
            y3   = y[:, 2].cuda() 
            ygt  = [y1, y2, y3]
            optimizer.zero_grad()

            yhat1, yhat2, yhat3 = model(img, text)
            pred = [yhat1, yhat2, yhat3]
            dloss = criterion.calculate_dloss(pred, ygt)
            lloss = criterion.calculate_lloss(pred, ygt)

            total_loss = dloss + lloss
            total_loss.backward()

            # gradient clipping
            clip_grad(model.parameters(), CFG["max_norm"])
            optimizer.step()                
            train_loss.append(total_loss.item())
            superclass_accuracy.append(calculate_accuracy(predictions=pred[0].detach(), labels=y1))
            subclass1_accuracy.append(calculate_accuracy(predictions=pred[1].detach(), labels=y2))
            subclass2_accuracy.append(calculate_accuracy(predictions=pred[2].detach(), labels=y3))
            
        tr_loss = np.mean(train_loss)
        supperclass_acc = np.mean(superclass_accuracy)
        subclass1_acc = np.mean(subclass1_accuracy)
        subclass2_acc = np.mean(subclass2_accuracy)
        

        val_loss, val_supperclass_acc,val_subclass1_acc, val_subclass2_acc, val_score = validation(model, criterion, val_loader)

        print(f'Epoch [{epoch}]')
        print(f'Train Loss : [{tr_loss:.5f}] | Val Loss : [{val_loss:.5f}]')
        print(f'Val Score  : [{val_score:.5f}]')
        print(f'Train superclass acc : {supperclass_acc:.5f}')
        print(f'Train subclass1 acc : {subclass1_acc:.5f}')
        print(f'Train subclass2 acc : {subclass2_acc:.5f}')
        print(f'Valid superclass acc : {val_supperclass_acc:.5f}')
        print(f'Valid subclass1 acc : {val_subclass1_acc:.5f}')
        print(f'Valid subclass2 acc : {val_subclass2_acc:.5f}')
        
        if(epoch < 6):
            warm_up.step()
        else:
            scheduler.step(val_loss)    # for main network of image

        if best_score < val_score:
            best_score = val_score
            best_model = model     

    return best_model



def score_function(real, pred):
    return f1_score(real, pred, average="weighted")

def validation(model, criterion, val_loader, device=CFG["device"]):
    model.eval()
    model_preds = []
    true_labels = []
    val_loss = []
    
    superclass_accuracy = []
    subclass1_accuracy = []
    subclass2_accuracy = []
    with torch.no_grad():
        for x_img, x_text, y in (iter(val_loader)):
            img  = x_img.float().cuda()
            text = text = x_text
            y1   = y[:, 0].cuda()
            y2   = y[:, 1].cuda()
            y3   = y[:, 2].cuda() 
            ygt  = [y1, y2, y3]

            yhat1, yhat2, yhat3 = model(img, text)
            pred = [yhat1, yhat2, yhat3]
            dloss = criterion.calculate_dloss(pred, ygt)
            lloss = criterion.calculate_lloss(pred, ygt)

            total_loss = dloss + lloss
            
            val_loss.append(total_loss.item())
            superclass_accuracy.append(calculate_accuracy(predictions=pred[0].detach(), labels=y1))
            subclass1_accuracy.append(calculate_accuracy(predictions=pred[1].detach(), labels=y2))
            subclass2_accuracy.append(calculate_accuracy(predictions=pred[2].detach(), labels=y3))


            model_preds += pred[2].argmax(1).detach().cpu().numpy().tolist()
            true_labels += y3.cpu().numpy().tolist()
    
    supperclass_acc = np.mean(superclass_accuracy)
    subclass1_acc = np.mean(subclass1_accuracy)
    subclass2_acc = np.mean(subclass2_accuracy)

    test_weighted_f1 = score_function(true_labels, model_preds)
    return np.mean(val_loss), supperclass_acc, subclass1_acc, subclass2_acc, test_weighted_f1


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
