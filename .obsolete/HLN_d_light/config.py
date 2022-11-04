LOCAL = False

# None will be filled run-time
import os
os.chdir("..")
from runtimedata.cat import cat1, cat2, cat3

folder_name = "HLN_d_light"
current_working_dir = f"/home/gyuseonglee/dacon/Dacon/{folder_name}/"
if not LOCAL:
    output_folder = f"/home/gyuseonglee/dacon/workplace/output/{folder_name}"
    data_path     = "/home/gyuseonglee/dacon/data/"
else:
    output_folder = f"C:/jupyter_notebook/dacon/workplace/output/{folder_name}"
    data_path     = f"C:/jupyter_notebook/dacon/data/"

model_name    = folder_name
model_states_name = folder_name+"state_dicts"


DATA = {
    "do_augmentation" : True,
    "do_embedding"    : True
}

CFG = {
    # directory
    "text_eda_dir"  : f"/home/gyuseonglee/dacon/Dacon/{folder_name}/functions/KorEDA",
    
    "org_img_train" : data_path + "original/image/train/",
    "org_img_test"  : data_path + "original/image/test/",
    "aug_img_train" : data_path + "augmented/image/train/",
    "aug_img_test"  : data_path + "augmented/image/test/",
    
    
    
    # text embedding directory
    "embedding_dir" : data_path + "embedding.pt",

    # image preprocessing
    "IMG_SIZE"      : 299,
    
    # text embedding dimension 
    "embedding_dim" : 4096,
    # "sentence_dim"  : 2048,

    # for run-time data
    "num_class1"     : 6,
    "num_class2"     : 18,
    "num_class3"     : 128,
    "subcat_for1"    : None,
    "subcat_for2"    : None,

    "cat1" : cat1,
    "cat2" : cat2,
    "cat3" : cat3,


    'BATCH_SIZE'     : 64,
    'EPOCHS'         : 30,
    'LEARNING_RATE'  : 0.0065,
    'SEED'           : 1203,
    "test_size"      : 0.18,
    "device"         : "cuda",
    
    "max_norm"       : 20.0,
}

scheduler_args = {
    "eps"       : 0.0001,
    # "cool_down" : 3,
    "patience"  : 2,     
}



