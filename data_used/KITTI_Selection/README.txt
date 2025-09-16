KITTI Selected Dataset:
20 Frames from the KITTI Dataset

Folderstructure:
KITTI_Selected/
├── calib/            # Calibration files
│   ├── 006037.txt
│   ├── ...
├── images/           # Image files
│   ├── 006037.jpg
│   ├── ...
├── labels/           # Label files
│   ├── 006037.txt
│   ├── ...
└── README.txt        # This document

Content Description:
calib  - contains the intrinsic camera matrix in shape 3 x 3
images - contains the camera images
labels - contains the groundtruth labels:
         object type: Type of the object (e.g., car, pedestrian).
         xmin, ymin:  Coordinates of the top-left corner of the bounding box.
         xmax, ymax:  Coordinates of the bottom-right corner of the bounding box.
         gt_distance: Ground truth distance from the camera to the object.
                      The grount truth distance was calculated by the 3D Label.
                      It is the minimal distance of the four 3D footpoints and the
                      central point on the four footedges to the camera center.
