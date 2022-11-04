LOCAL = False

# None will be filled run-time
import os
os.chdir("..")
from runtimedata.cat import cat1, cat2, cat3

seed_number = 33
folder_name = f"MVitRobertaAugModNormFull33"

current_working_dir = f"/home/gyuseonglee/dacon/workplace/{folder_name}/"
if not LOCAL:
    output_folder = f"/home/gyuseonglee/dacon/workplace/output/{folder_name}{seed_number}"
    data_path     = "/home/gyuseonglee/dacon/data/"
    train_data    = "/home/gyuseonglee/dacon/data/train.csv"
    test_data     = "/home/gyuseonglee/dacon/data/test.csv"
    submit_file   = "/home/gyuseonglee/dacon/data/sample_submission.csv"
    submit_dir    = f"/home/gyuseonglee/dacon/data/sample_submission{folder_name}{seed_number}.csv"
else:
    output_folder = f"C:/jupyter_notebook/dacon/workplace/output/{folder_name}{seed_number}"
    data_path     = "C:/jupyter_notebook/dacon/data/"
    train_data    = "C:/jupyter_notebook/dacon/data/train.csv"
    test_data     = "C:/jupyter_notebook/dacon/data/test.csv"
    submit_file   = "C:/jupyter_notebook/dacon/data/sample_submission.csv"
    submit_dir    = f"C:/jupyter_notebook/dacon/data/sample_submission{folder_name}{seed_number}.csv"

model_name    = folder_name
model_states_name = folder_name+"state_dicts"


DATA = {
    "do_augmentation" : True,
}

CFG = {
    # directory
    "text_eda_dir"  : f"{current_working_dir}functions/KorEDA",
    
    "org_img_train" : data_path + "original/image/train/",
    "org_img_test"  : data_path + "original/image/test/",
    
    # runtime data directory
    "embedding_dir" : data_path + f"doc2vec{seed_number}.dv",
    "infer_embedding_dir" : data_path + f"embedding_infer{seed_number}.pt",
    
    
    "augimg_dir"    : data_path + f"augmented/{folder_name}{seed_number}/augimg.pk",
    "augtxt_dir"    : data_path + f"augmented/{folder_name}{seed_number}/augtxt.pk",
    "augcat1_dir"   : data_path + f"augmented/{folder_name}{seed_number}/augcat1.pk",
    "augcat2_dir"   : data_path + f"augmented/{folder_name}{seed_number}/augcat2.pk",
    "augcat3_dir"   : data_path + f"augmented/{folder_name}{seed_number}/augcat3.pk",

    # image preprocessing
    "IMG_RESIZE"    : 284,  # for resizing
    "IMG_SIZE"      : 256,  # for center/random crop
    
    # text embedding dimension 
    "embedding_dim" : 768,
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


    'BATCH_SIZE'     : 32,
    'INFER_BATCH_SIZE' : 128,
    'EPOCHS'         : 35,
    'LEARNING_RATE'  : 0.0023,
    'SEED'           : seed_number,
    "test_size"      : 0.08,
    "device"         : "cuda",
    
    "max_norm"       : 20.0,
}

scheduler_args = {
    "eps"       : 0.0001,
    # "cool_down" : 3,
    "patience"  : 2,     
}


inference_config = {
    "seeds"      : [33, 317, 1203],
    # "weight_dir" : f"/home/gyuseonglee/dacon/workplace/output/{folder_name}{seed_number}/{folder_name}state_dicts",
    "weight_dir" : f"/home/gyuseonglee/dacon/workplace/output/{folder_name}{seed_number}/{folder_name}state_dicts",
    
}    
    