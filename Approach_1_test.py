import cv2
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt

from tensorflow.python.keras.layers import Input
# ** Original was tensorflow.keras.layers import Input

from src.yolo3.model import *
from src.yolo3.detect import *

from src.utils.image import *
from src.utils.datagen import *
from src.utils.fixes import *

import random
import json

fix_tf_gpu()


def prepare_model(approach):
    """
    Prepare the YOLO model
    """
    global input_shape, class_names, anchor_boxes, num_classes, num_anchors, model

    # shape (height, width) of the input image
    input_shape = (416, 416)

    # class names
    if approach == 1:
        class_names = ['H', 'V', 'W']

    elif approach == 2:
        class_names = ['W', 'WH', 'WV', 'WHV']

    elif approach == 3:
        class_names = ['W']

    else:
        raise NotImplementedError('Approach should be 1, 2, or 3')

    # anchor boxes
    if approach == 1:
        anchor_boxes = np.array(
            [
                np.array([[76, 59], [84, 136], [188, 225]]) / 32,  # output-1 anchor boxes
                np.array([[25, 15], [46, 29], [27, 56]]) / 16,  # output-2 anchor boxes
                np.array([[5, 3], [10, 8], [12, 26]]) / 8  # output-3 anchor boxes
            ],
            dtype='float64'
        )
    else:
        anchor_boxes = np.array(
            [
                np.array([[73, 158], [128, 209], [224, 246]]) / 32,  # output-1 anchor boxes
                np.array([[32, 50], [40, 104], [76, 73]]) / 16,  # output-2 anchor boxes
                np.array([[6, 11], [11, 23], [19, 36]]) / 8  # output-3 anchor boxes
            ],
            dtype='float64'
        )

    # number of classes and number of anchors
    num_classes = len(class_names)
    num_anchors = anchor_boxes.shape[0] * anchor_boxes.shape[1]

    # input and output
    input_tensor = Input(shape=(input_shape[0], input_shape[1], 3))  # input
    num_out_filters = (num_anchors // 3) * (5 + num_classes)  # output

    # build the model
    model = yolo_body(input_tensor, num_out_filters)

    # load weights
    weight_path = f'model-data\weights\pictor-ppe-v302-a{approach}-yolo-v3-weights.h5'
    model.load_weights(weight_path)


def get_detection(img):
    """
    detect the objects from images that have selected randomly
    :param img:
    :return:
    """
    # save a copy of the img
    act_img = img.copy()

    # shape of the image
    ih, iw = act_img.shape[:2]

    # preprocess the image
    img = letterbox_image(img, input_shape)
    img = np.expand_dims(img, 0)
    image_data = np.array(img) / 255.

    # raw prediction from yolo model
    prediction = model.predict(image_data)

    # process the raw prediction to get the bounding boxes
    boxes = detection(
        prediction,
        anchor_boxes,
        num_classes,
        image_shape=(ih, iw),
        input_shape=(416, 416),
        max_boxes=10,
        score_threshold=0.3,
        iou_threshold=0.45,
        classes_can_overlap=False)

    # convert tensor to numpy
    boxes = boxes[0].numpy()
    output_doc(boxes)

    # draw the detection on the actual image
    return draw_detection(act_img, boxes, class_names)


def plt_imshow(img):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis('off')


# *****************************************************
def output_doc(boxes):
    """
    read the boxes from and create a json file called doc_info.json in the img_src directory
    this file can be used as the results of the images that ran thorough the script
    :param boxes:
    :return:
    """
    # print(boxes)
    object_info = []
    o_info = []
    cat_id = -1

    # print(f"{len(boxes)} objects detected from {image_name}\n")
    # for item_no in range(len(boxes)):
    # ['W', 'WH', 'WV', 'WHV']

    for box in boxes:
        if box[5] == 0.0:
            detected_object = "person"
            cat_id = 1
        elif box[5] == 1.0:
            detected_object = "person-hat"
            cat_id = 2
        elif box[5] == 2.0:
            detected_object = "person-hivis"
            cat_id = 0
        elif box[5] == 3.0:
            detected_object = "person-hat-hivis"
            cat_id = 3

        o_info += [
            {
                "Object_Class": float(box[5]),
                "Object_name": detected_object,
                # if output need more details about the accuracy , uncomment the next two rows
                # "Certainty(raw)": float(box[4]),
                # "Certainty(%)": f'{round(box[4] * 100)}%',
            },
        ]

        imgID = image_name[1]

    read_data = []

    doc_data = {
        "Image_name": image_name[0],
        "Image_ID": image_name[1],
        "object_count": len(boxes),
        "object_info": o_info,
    }

    try:
        with open('img_src/doc_info.json', 'r+') as json_file:
            read_data = (json.load(json_file))

    except ValueError:
        pass

    read_data.append(doc_data)

    with open('img_src/doc_info.json', 'w') as json_file:
        json.dump(read_data, json_file)


def read_images():
    """
    read images form the rush_src , using ground_truth_labels.json
    and load it using json into a python set

    """
    # to list down all the names of the test images
    img_data = []
    # ano_data = []
    cat_data = []

    # use json file to map the test image data
    with open('img_src/ground_truth_labels.json') as data_json:
        dataSet = json.load(data_json)
        image_data = dataSet["images"]
        annotations_data = dataSet["annotations"]
        categories_data = dataSet["categories"]

        for item in image_data:
            img_data.append([item['file_name'], item['id']])

        for item in annotations_data:
            ano_data.append([item['image_id'], item['category_id']])

        for item in categories_data:
            cat_data.append([item['id'], item['supercategory']])

        img_data = sorted(img_data)

    info['image_data'] = img_data
    info['annotations_data'] = ano_data
    info['categories_data'] = cat_data

    return img_data

# *****************************************************


selected_img = []
selected_img_id = []
ano_data = []
processed_img = []
info = {}

with open('img_src/doc_info.json', 'w') as json_file:
    pass

# set number n to image amount you need to run through the script
image_count = 50

for i in range(image_count):
    selected_img.append(random.choice(read_images()))
    selected_img_id.append(selected_img[i][1])

count = 0

# set model_approach to number 1, 2, 3 as you wish to get the output
model_approach = 2
prepare_model(model_approach)

for image_name in selected_img:
    count += 1
    # read the image
    img = cv2.imread(f'img_src/test_images/{image_name[0]}')

    # resize
    img = letterbox_image(img, input_shape)

    # get the detection on the image
    img = get_detection(img)

    cv2.imshow(f'Img - {count} {image_name[0]}', img)
    cv2.waitKey(0)


# write the selected image data into another json file called selected_img_info
with open('img_src/selected_img_info.json', 'w') as json_file:
    json.dump(info, json_file)
