# traffic-sign-mahindra

## Introduction
This package simultneously detects the traffic signs from urban environments and classifies them and also estimates the distance of the sign from the camera. Currently designed for Indian traffic signs. TO increase robustness and fps, a tracker (KCF) is also integrated


## Dataset
No dataset for Indian Traffic Signs. The German signs are quite similar to Indian signs. So for detection GTSDB dataset is used and for classification GTSRB dataset is used. Since the number of images in GTSDB dataset is very small so a very robust detector could not be made that simultaneously classifies the signs as well

## Pretrained model

Detector:
Download [models](https://drive.google.com/open?id=1sFRgTOEs2SsJ6WsOA6VkHWk260S5EEat) in the cloned folder.

You can use any of the three pretrained models, rfcn_resnet101, ssd_inception_v2, ssd_mobilenet_v1.

Classifier:
Download the [models](https://drive.google.com/open?id=1JBkNJS86w05VJIlaD6Rf5XXg95FOieUF) in the classifier folder.

## Run the code

tracker_cam_demo.py is the code that runs on either live camera input or a recorded video (nessecary changes are to be made)

traffic_sign_ros.py consists of the ros integrated code. It subscribes images and publishes the name of the sign along with the estimated distance encoded in a string. The names of the topics can be changed as per convenience. 
