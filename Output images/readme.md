Folder_1: Averageprecision

This folder contain the images with Average precision at the bottom, the average precision is bascially :
=(total right detection)/ (total right and wrong detection)

Also it includes the IOU at top and name of detection here Car.
IOU: Intersection over Union = common area enclosed by green(groundtruth boxes) and red (yolo detection boxes) / area enclosed by both box the UNION


Folder_2: Filtering

Here I removed the detection boxes by YOLO, that car position is not present in the already given KITTI DATASET files.

This way we can reduce error margin the goal was to train YOLO Model, missing data will create disturbance.


Folder_3: NO_filtering

Here the YOLO detected the car with RED BOXES, but on checking with already given data we find there are some car whose data missing.

The car whose data is present are marked with Green boxes, and also you can see some car YOLO Boxes overlapping to groundtruth Green boxes.


Folder_4: only_yolo

This folder contain images, where only YOLO model is run to detect car.
The red boxes are YOLO Model Detections.

Folder_5: Depth_Ground_and_YOLO

This folder contains the images with car and 'GT': Ground truth distance given in already existing file from KITTI DATASET, and 'YOLO' the measured depth from YOLO Model detection. Distance here measured in meter

