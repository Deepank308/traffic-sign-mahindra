import warnings
warnings.filterwarnings('ignore')
import numpy as np
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import glob as glob
import cv2
import time 
import sys

NMS_THRESH = 0.6

# Append your Tensorflow object detection and darkflow directories to your path
# sys.path.append('models-tf/research/object_detection/') # ~/tensorflow/models/research/object_detection # CHECK
# sys.path.append('darkflow-master/') # ~/darkflow
# from utils import label_map_util
# from utils import visualization_utils as vis_util

MODEL_NAME = 'rfcn_resnet101'
# Path to frozen detection graph. This is the actual model that is used for the traffic sign detection.
MODEL_PATH = os.path.join('models', MODEL_NAME)
PATH_TO_CKPT = os.path.join(MODEL_PATH,'inference_graph/frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('scripts', 'gtsdb3_label_map.pbtxt')
image_np_expanded = None
bridge = None
NUM_CLASSES = 3
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print("Loaded model")
# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
# print(label_map)

def iou(box1, box2):
    """
    box : ymin,xmin,ymax,xmax
    """
    boxA = [box1[1],box1[0],box1[3],box1[2]]
    boxB = [box2[1],box2[0],box2[3],box2[2]]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

def nms(boxes,thresh=0.5):
    boxes_filtered = []
    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[0]):
            if i == j:
                continue
            if iou(boxes[i],boxes[j]) > thresh:
                if i < j:
                    boxes_filtered.append(boxes[i])
        
    return boxes_filtered
            

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

IMAGE_SIZE = (20, 20)

def load_image_into_numpy_array(image):
    (im_width, im_height,_) = image.shape
    return np.array(image.reshape((im_height, im_width, 3)).astype(np.uint8))

def process_img_data(img):
    global image_np_expanded
    # image_np = load_image_into_numpy_array(img)
    image_np_expanded = np.expand_dims(img, axis=0)

def main():
    cap = cv2.VideoCapture(0)
    
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            while True:
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = detection_graph.get_tensor_by_name('detection_scores:0')
                classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                ret,img = cap.read()
                print(img.shape)
                # img = cv2.imread("test_images/a.png")

                process_img_data(img)

                # Make Predictions
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                valid_boxes = []
                boxes = np.squeeze(boxes)
                scores = np.squeeze(scores)
                
                for i in range(boxes.shape[0]):
                    if scores[i] > 0.5:
                        valid_boxes.append(boxes[i])
                    else:
                        break
                
                print(np.max(scores))

                print(boxes)
                cv2.namedWindow("OUTPUT",cv2.WINDOW_NORMAL)
                if len(valid_boxes) < 1:
                    print("NO TRAFFIC SIGN")
                    cv2.imshow("OUTPUT",img)
                    cv2.waitKey(1)
                    continue

                print(valid_boxes)
                
                valid_boxes = np.vstack(valid_boxes)
                
                boxes_filtered = valid_boxes

                # boxes_filtered = nms(valid_boxes,NMS_THRESH)

                print(boxes_filtered.shape)
                for box in boxes_filtered:
                    cv2.rectangle(img,(int(box[1] * img.shape[1]),int(box[0] * img.shape[0])),(int(box[3] * img.shape[1]),int(box[2] * img.shape[0])),(0,255,0),3)
                
                cv2.imshow("OUTPUT",img)
                cv2.waitKey(1)
main()