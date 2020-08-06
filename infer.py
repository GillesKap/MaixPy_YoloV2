import json
import os
import numpy as np

import cv2

from evaluate import create_ann
from yolo.backend.utils.box import draw_scaled_boxes, to_minmax
from yolo.frontend import create_yolo

DEFAULT_THRESHOLD = 0.4

evaluation_object = "brio"
DEFAULT_CONFIG_FILE = evaluation_object + "/config.json"
DEFAULT_WEIGHT_FILE = evaluation_object + "/weights.h5"

with open(DEFAULT_CONFIG_FILE) as config_buffer:
    config = json.loads(config_buffer.read())


def _to_original_scale(boxes):
    height, width = image.shape[:2]
    minmax_boxes = to_minmax(boxes)
    minmax_boxes[:, 0] *= width
    minmax_boxes[:, 2] *= width
    minmax_boxes[:, 1] *= height
    minmax_boxes[:, 3] *= height
    return minmax_boxes.astype(np.int)


if __name__ == "__main__":
    threshold = 0.3
    yolo = create_yolo(config['model']['architecture'],
                       config['model']['labels'],
                       config['model']['input_size'],
                       config['model']['anchors'])
    # yolo.load_weights(DEFAULT_WEIGHT_FILE)
    yolo.load_weights("./weights.h5")

    img_fname = "brio_5.jpg"
    img_path = os.path.join('datasets', "brio", 'images_test', img_fname)

    image = cv2.imread(img_path)

    # boxes, probs = yolo.predict(image, float(DEFAULT_THRESHOLD))

    # for testing purposes, splitting up prediction
    netout = yolo._yolo_network.forward(image)
    # netout.shape = (7, 7, 5, 6) -- 7x7 = grid
    boxes, probs = yolo._yolo_decoder.run(netout, threshold)

    if len(boxes) > 0:
        boxes = _to_original_scale(boxes)
    else:
        boxes, probs = [],[]

    exit()

    # labels = np.argmax(probs, axis=1) if len(probs) > 0 else []
    #
    # # 4. save detection result
    # image = draw_scaled_boxes(image, boxes, probs, config['model']['labels'])
    # label_list = config['model']['labels']
    #
    # if len(probs) > 0:
    #     create_ann(img_fname, image, boxes, labels, label_list)
    #     output_path = "temp_output/inferred.png"
    #     cv2.imwrite(output_path, image)
