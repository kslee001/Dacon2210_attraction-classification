
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
    print(f"job started -- current model : {folder_name}")
    start_time = t.time()

    # load data
    X1, X2, X3, y1, y2, y3   = preprocessing.load_data(
        do_augmentation=DATA["do_augmentation"], 
        do_embedding=DATA["do_embedding"]
    )
    train_loader, val_loader = preprocessing.generate_dataloader(X1, X2, X3, y1, y2, y3)
    
    print(X3)
    print(X3.shape)