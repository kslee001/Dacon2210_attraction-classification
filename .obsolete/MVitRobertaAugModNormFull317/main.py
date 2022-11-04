# config & dataset
# import os
# os.chdir("/home/gyuseonglee/.Dacon/workplace/TNBranchC1")

from config import *
from model.hierarchical_loss import HierarchicalLossNetwork
import functions.preprocessing as preprocessing
import functions.train as train
from model.models import *

import torch
import time as t
import subprocess
import json
import pprint
DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)
def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]

def gpu_mem():
    pprint.pprint(get_gpu_info())


    
if __name__ == "__main__":
    # try:
    print(f"job started -- current model : {folder_name}{seed_number}")
    start_time = t.time()

    # load data
    X1, X2, y1, y2, y3, augmented_img  = preprocessing.load_data(
        do_augmentation = DATA["do_augmentation"]
    )
    train_loader, val_loader = preprocessing.generate_dataloader(X1, X2, y1, y2, y3, augmented_img, False)

    # define model
    model     = MVitRobertaAugModNormFull317().cuda()
    optimizer = torch.optim.Adam(
        params       = model.parameters(), 
        lr           = CFG["LEARNING_RATE"],
    )
    criterion = [torch.nn.CrossEntropyLoss().cuda() for _ in range(3)]

    # parallelize model
    model = torch.nn.DataParallel(model)
    warm_up = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=CFG["LEARNING_RATE"]/5, 
        end_factor=CFG["LEARNING_RATE"], 
        total_iters=5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        "max", 
        **scheduler_args
    )


    """train Phase"""
    print("train started...")
    train.make_dir(output_folder)
    best_model = train.train(
        mode="best",
        model=model, 
        optimizer=optimizer,
        criterion=criterion,
        warm_up=warm_up,
        scheduler=scheduler,
        train_loader=train_loader, 
        val_loader=val_loader
    )

    # """Second train Phase"""
    # con_optimizer = torch.optim.Adam(
    #     params       = best_model.parameters(), 
    #     lr           = CFG["LEARNING_RATE"]*0.75,
    # )
    # con_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     con_optimizer, 
    #     "max", 
    #     **scheduler_args
    # )
    
    # print("second train started...")
    # best_model = train.train(
    #     mode="second",
    #     model=best_model, 
    #     optimizer=con_optimizer,
    #     criterion=criterion,
    #     warm_up=None,
    #     scheduler=con_scheduler,
    #     train_loader=train_loader, 
    #     val_loader=val_loader
    # )

    # save model
    train.save_model(best_model)
    # save configs
    train.save_configs()
    
    

    end_time = t.time()
    duration = (end_time-start_time)
    
    h = int(duration//3600)
    m = int(duration//60 - h*60)
    s = round(duration%60,3)

    print("job finished")
    print(f"duration : {h} h {m} m {s} s")
    gpu_mem()
    
    # except:

