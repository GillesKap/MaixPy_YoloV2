# python script version of training.ipynb (i.e. copy paste)

import numpy as np
np.random.seed(111)
import os
import json
from yolo.frontend import create_yolo, get_object_labels\



def setup_training(config_file):
    """make directory to save weights & its configuration """
    import shutil

    with open(config_file) as config_buffer:
        config = json.loads(config_buffer.read())
    dirname = config['train']['saved_folder']

    if os.path.isdir(dirname):
        print("{} is already exists. Weight file in directory will be overwritten".format(dirname))
    else:
        print("{} is created.".format(dirname, dirname))
        os.makedirs(dirname)

    print("Weight file and Config file will be saved in \"{}\"".format(dirname))
    shutil.copyfile(config_file, os.path.join(dirname, "config.json"))

    return config, os.path.join(dirname, "weights.h5")


import warnings
if __name__ == "__main__":
    config_file = "configs/brio.json"
    config, weight_file = setup_training(config_file)

    if config['train']['is_only_detect']:
        labels = ["object"]
    else:
        if config['model']['labels']:
            labels = config['model']['labels']
        else:
            labels = get_object_labels(config['train']['train_annot_folder'])

    print(labels)

    # 1. Construct the model
    yolo = create_yolo(config['model']['architecture'],
                       labels,
                       config['model']['input_size'],
                       config['model']['anchors'],
                       config['model']['coord_scale'],
                       config['model']['class_scale'],
                       config['model']['object_scale'],
                       config['model']['no_object_scale'])

    # 2. Load the pretrained weights (if any)
    yolo.load_weights(config['pretrained']['full'], by_name=True)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 3. actual training
        yolo.train(config['train']['train_image_folder'],
                   config['train']['train_annot_folder'],
                   config['train']['actual_epoch'],
                   weight_file,
                   config["train"]["batch_size"],
                   config["train"]["jitter"],
                   config['train']['learning_rate'],
                   config['train']['train_times'],
                   config['train']['valid_times'],
                   config['train']['train_image_folder'],
                   config['train']['train_annot_folder'],
                   config['train']['first_trainable_layer'],
                   config['train']['is_only_detect'])

