# None will be filled run-time

folder_name = "TinyHLN"

output_folder = f"/home/gyuseonglee/dacon/workplace/output/{folder_name}"
data_path     = "/home/gyuseonglee/dacon/data/"
model_name    = folder_name
model_states_name = folder_name+"state_dicts"

CFG = {
    # image directory
    
    "org_img_train" : data_path + "original/image/train/",
    "org_img_test"  : data_path + "original/image/test/",
    "aug_img_train" : data_path + "augmented/image/train/",
    "aug_img_test"  : data_path + "augmented/image/test/",
    
    # text embedding directory
    "embedding_dir" : data_path + "embedding.pt",

    # for run-time data
    "num_class1"     : 6,
    "num_class2"     : 18,
    "num_class3"     : 128,
    "subcat_for1"    : None,
    "subcat_for2"    : None,


    'BATCH_SIZE'     : 32,
    'EPOCHS'         : 30,
    'LEARNING_RATE'  : 0.099,
    'SEED'           : 22364,
    "test_size"      : 0.25,
    "device"         : "cuda",
    
    "max_norm"       : 20.0,
}

scheduler_args = {
    "eps"       : 0.0001,
    # "cool_down" : 3,
    "patience"  : 3,     
}



