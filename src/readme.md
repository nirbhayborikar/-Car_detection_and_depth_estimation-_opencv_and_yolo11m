FILE: Main_car_detection_with_Yolo.py
-------------------------
This file is the main file where YOLO Model is used to detect car, and also with ground truth the green boxes around the car is created.
The YOLO detection marked with red.

The IOU and AP is calculated in order to check the efficiency of the YOLO 11m Model.

-------------------------

FILE: depth_estimation.py

Here I load the intrinsic matrix from already calibrated file given from KITTI Dataset.
Then on each detected boxes the depth is measured based on the given data, along camera car and detected boxes.

