import argparse
import json
import cv2
import numpy as np
from yolo.frontend import create_yolo
from yolo.backend.utils.box import draw_scaled_boxes
from yolo.backend.utils.annotation import parse_annotation
from yolo.backend.utils.eval.fscore import count_true_positives, calc_score

from pascal_voc_writer import Writer
from shutil import copyfile
import os
import yolo

evaluation_object = "brio"
DEFAULT_CONFIG_FILE = evaluation_object + "/config.json"
DEFAULT_WEIGHT_FILE = evaluation_object + "/weights.h5"
DEFAULT_THRESHOLD = 0.4

with open(DEFAULT_CONFIG_FILE) as config_buffer:
    config = json.loads(config_buffer.read())


def create_ann(filename, image, boxes, labels, label_list):
    if not os.path.exists('evaluation/imgs/'):
        os.makedirs('evaluation/imgs/')
    if not os.path.exists('evaluation/ann/'):
        os.makedirs('evaluation/ann/')
    copyfile(os.path.join('datasets', evaluation_object, 'images_test', filename), 'evaluation/imgs/' + filename)

    writer = Writer(os.path.join('datasets', evaluation_object, 'images_test', filename), 224, 224)
    writer.addObject(label_list[labels[0]], boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])
    name = filename.split('.')
    writer.save('evaluation/ann/' + name[0] + '.xml')


if __name__ == "__main__":
    # 2. create yolo instance & predict
    yolo = create_yolo(config['model']['architecture'],
                       config['model']['labels'],
                       config['model']['input_size'],
                       config['model']['anchors'])
    # yolo.load_weights(DEFAULT_WEIGHT_FILE)
    yolo.load_weights("./weights.h5")
    # 3. read image
    write_dname = "evaluation/detected"
    if not os.path.exists(write_dname): os.makedirs(write_dname)
    annotations = parse_annotation(config['train']['valid_annot_folder'],
                                   config['train']['valid_image_folder'],
                                   config['model']['labels'],
                                   is_only_detect=config['train']['is_only_detect'])

    # n_true_positives = 0
    # n_truth = 0
    # n_pred = 0
    # for i in range(len(annotations)):
    for filename in os.listdir('datasets/' + evaluation_object + '/images_test'):
        img_path = os.path.join('datasets', evaluation_object, 'images_test', filename)
        img_fname = filename
        image = cv2.imread(img_path)

        boxes, probs = yolo.predict(image, float(DEFAULT_THRESHOLD))
        labels = np.argmax(probs, axis=1) if len(probs) > 0 else []

        # 4. save detection result
        image = draw_scaled_boxes(image, boxes, probs, config['model']['labels'])
        output_path = os.path.join(write_dname, os.path.split(img_fname)[-1])
        label_list = config['model']['labels']
        # cv2.imwrite(output_path, image)
        print("{}-boxes are detected. {} saved.".format(len(boxes), output_path))
        if len(probs) > 0:
            create_ann(filename, image, boxes, labels, label_list)
            cv2.imwrite(output_path, image)
