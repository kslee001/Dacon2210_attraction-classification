# config & dataset
import os
os.chdir("/home/gyuseonglee/dacon/workplace/TinyHLN2")
from config import *
import ldata
import functions
from models import *
from optimizers import Lamb

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
    print("job started")
    start_time = t.time()

    # load data
    X1, X2, y1,y2,y3         = ldata.load_data(do_augmentation=False, do_embedding=True)
    train_loader, val_loader = ldata.generate_dataloader(X1, X2, y1,y2,y3)
    print("data loaded...")

    # define model
    model = TinyNet3_HLN().cuda()

    # params
    __param_main = list(model.image_main.parameters())
    __param1     = (
        list(model.image_branch1.parameters())
        + list(model.image_classifier1.parameters()) 
        + list(model.txt_classifier1.parameters())
        + list(model.fin_classifier1_1.parameters())
        + list(model.fin_classifier1_2.parameters())
        + list(model.fin_classifier1_con1.parameters())
        + list(model.fin_classifier1_con2.parameters())
    )
    __param2     = (
        list(model.image_branch2.parameters())
        + list(model.image_classifier2.parameters()) 
        + list(model.txt_classifier2.parameters())
        + list(model.fin_classifier2_1.parameters())
        + list(model.fin_classifier2_2.parameters())
        + list(model.fin_classifier2_con1.parameters())
        + list(model.fin_classifier2_con2.parameters())
    )
    __param3     = (
        list(model.image_branch3.parameters())
        + list(model.image_classifier3.parameters()) 
        + list(model.txt_classifier3.parameters())
        + list(model.fin_classifier3_1.parameters())
        + list(model.fin_classifier3_2.parameters())
        + list(model.fin_classifier3_con1.parameters())
        + list(model.fin_classifier3_con2.parameters())
    )
    # __param12    = (
    #     list(model.image_transferer1.parameters())
    #     + list(model.image_tranhelper1.parameters())
    #     + list(model.txt_transferer1.parameters())
    #     + list(model.txt_tranhelper1.parameters())
    # )
    # __param23    = (
    #     list(model.image_transferer2.parameters())
    #     + list(model.image_tranhelper2.parameters())
    #     + list(model.txt_transferer2.parameters())
    #     + list(model.txt_tranhelper2.parameters())
    # )
    P = [__param_main, __param1, __param2, __param3]#, __param12, __param23]

    optimizers =[
        torch.optim.Adam(
            params       = P[i], 
            lr           = CFG["LEARNING_RATE"],
        ) for i in range(len(P))
    ]

    criterions = [
        torch.nn.CrossEntropyLoss().cuda() for _ in range(3)
    ]

    # parallelize model
    model = torch.nn.DataParallel(model, device_ids = [0,1])
    model.eval()

    warm_ups = [
        torch.optim.lr_scheduler.LinearLR(optimizers[i], start_factor=CFG["LEARNING_RATE"]/5, end_factor=CFG["LEARNING_RATE"], total_iters=5) for i in range(len(P))
    ]
    schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizers[i], "max", **scheduler_args) for i in range(len(P)) 
    ]


    print("train started...")
    functions.make_dir(output_folder)
    best_model = functions.train(
        model=model, 
        optimizers=optimizers,
        criterions=criterions,
        warm_ups=warm_ups,
        schedulers=schedulers,
        train_loader=train_loader, 
        val_loader=val_loader
    )

    # save model
    functions.save_model(best_model)
    # save configs
    functions.save_configs()
    
    gpu_mem()

    end_time = t.time()
    duration = (end-start)*3000
    
    h = int(duration//3600)
    m = int(duration//60 - h*60)
    s = round(duration%60,3)

    print("job finished")
    print(f"duration : {h} h {m} m {s} s")

