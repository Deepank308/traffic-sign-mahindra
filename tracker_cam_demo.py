import warnings
import math
warnings.filterwarnings('ignore')
import numpy as np
from numpy.linalg import inv
import os
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import glob as glob
import cv2
import time 
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from classifier.classifier import Net, evaluate_single

NMS_THRESH = 0.6
NUM_CLASSES_classifier = 7


MODEL_NAME = 'ssd_inception_v2'
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
M = np.asarray([[660.50915392, 0. , 314.49448395],[0. , 663.22638251 , 271.43837282] , [0. , 0. , 1.]])
def distance(x1,y1,x2,y2):
    global M
    # print("\ntyfy\n")
    # print(x1,y1,x2,y2)
    a1, b1, c1 = np.matmul(inv(M) , np.asarray([[x1],[y1],[1]]))
    a2, b2, c2 = np.matmul(inv(M) , np.asarray([[x2],[y2],[1]]))
    # print("\n\nhfjkldsh\n\n")
    # print(a1,b1,c1,a2,b2,c2)
    #d = 0.185
    d = 0.300
    s = d/math.sqrt((a1-a2*c1/c2)**2 + (b1-b2*c1/c2)**2)
    return s

def iou(box1, box2):
    """
    box : ymin,xmin,ymax,xmax
    """
    boxA = [box1[1],box1[0],box1[3],box1[2]]
    boxB = [box2[1],box2[0],box2[3],box2[2]]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
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
    image_np_expanded = np.expand_dims(img, axis=0)

def create_bb(point, offset):
    bbox = (point[0]-offset, point[1]-offset, offset, offset)
    return bbox

sign_dict = {0 : "Right", 1 : "Left", 2 : "Stop", 3 : "Right-Ahead", 4 : "Forward", 5 : "Left-Ahead", 6 : "Noise"}

def main():
    net = Net(NUM_CLASSES_classifier)
    net.load_state_dict(torch.load('classifier/models/sign_classifier2.pt'))
    net = net.eval().cuda()

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("../Videos/Webcam/stop_test.webm")
    to_track = False
    frame = 0
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            
            while True:
                time1 = time.time()
                sign_present = False
                ret,img = cap.read()
                img_blk = np.zeros((img.shape[0], img.shape[1], 3))
                roi_to_classify = None
                if ret is False:
                    continue

                process_img_data(img)
                img_clone = img.copy()

                if to_track == False:
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                    # Make Predictions
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    frame = frame + 1
                    valid_boxes = []
                    boxes = np.squeeze(boxes)
                    scores = np.squeeze(scores)
                    
                    for i in range(boxes.shape[0]):
                        if scores[i] > 0.5:
                            valid_boxes.append(boxes[i])
                        else:
                            break
                    
                    cv2.namedWindow("OUTPUT",cv2.WINDOW_NORMAL)

                    try:
                        valid_boxes = np.vstack(valid_boxes)
                    except:
                        pass           
                    boxes_filtered = valid_boxes

                    for box in boxes_filtered:
                        sign_present = True
                        cv2.rectangle(img,(int(box[1] * img.shape[1]),int(box[0] * img.shape[0])),(int(box[3] * img.shape[1]),int(box[2] * img.shape[0])),(0,255,0),3)
                        roi_to_classify = img[int(box[0] * img.shape[0]):int(box[2]*img.shape[0]), int(box[1] * img.shape[0]):int(box[3]*img.shape[0])]

                    if sign_present:
                        print("FOUND TRAFFIC SIGN")
                        bbox = (boxes_filtered[0][1]*img.shape[1], boxes_filtered[0][0]*img.shape[0],
                                (boxes_filtered[0][3] - boxes_filtered[0][1])*img.shape[1],
                                (boxes_filtered[0][2] - boxes_filtered[0][0])*img.shape[0])
                       
                        # cv2.imwrite("mar.png",img) 
                        img_clone = cv2.resize(img_clone,(img.shape[1]//2,img.shape[0]//2))
                        bbox = (bbox[0]//2,bbox[1]//2,bbox[2]//2,bbox[3]//2)
                        tracker = cv2.TrackerKCF_create()
                        ok = tracker.init(img_clone, bbox)
                        to_track = True
                    # else:
                    #     continue

                else:
                    img_clone = cv2.resize(img_clone,(img.shape[1]//2,img.shape[0]//2))
                    if(frame %15 == 0):
                        to_track = False
                    ok, bbox = tracker.update(img_clone)

                    print("updating Tracker")
                    bbox = (int(bbox[0]*2),int(bbox[1]*2),int(bbox[2]*2),int(bbox[3]*2))
                    if ok:
                        print("Ok, tracking")
                        frame = frame + 1
                        roi_to_classify = img[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
                        cv2.rectangle(img,(bbox[0],bbox[1]),(bbox[0]+bbox[2], bbox[1]+bbox[3]),(0,0,255),3)
                        # try:
                        #     dist = distance(bbox[0],bbox[1],bbox[0] + bbox[2],bbox[1])
                        #     print("DISTANCE : {}".format(dist))
                        # except:
                        #     print("Distance me error")
                        #     cv2.imshow('OUTPUT', img)
                        #     cv2.waitKey(1)
                        #     continue
                    else:
                        to_track = False

                if roi_to_classify is not None :
                    sign = evaluate_single(roi_to_classify, net)
                    cv2.putText(img, sign_dict[sign], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0 , 0), 2, cv2.LINE_AA)
                ctime = time.time()
                print('FPS: ',1/(ctime - time1))
                cv2.imshow("OUTPUT",img)
                cv2.waitKey(1)
main()
