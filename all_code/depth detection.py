import cv2
import math
import cvzone
import numpy as np
from ultralytics import YOLO

# Load the vehicle image
image_path = 'C:\\Users\\Lenovo\\Desktop\\rwu\\computer vision\\TASK_2_YOLO\\KITTI_Selection (1)\\KITTI_Selection\\images\\006374.png'
vehicle = cv2.imread(image_path)
ground_truth_file = "C:\\Users\\Lenovo\\Desktop\\rwu\\computer vision\\TASK_2_YOLO\\KITTI_Selection (1)\\KITTI_Selection\\labels\\006374.txt"

if vehicle is None:
    print("Error: Image file not found or could not be loaded.")
    exit()

# Load YOLO model
model = YOLO("yolo11m.pt")

# Load ground truth bounding boxes
ground_truth_boxes = []
try:
    with open(ground_truth_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x1, y1, x2, y2, depth = map(float, parts[1:6])
            ground_truth_boxes.append((int(x1), int(y1), int(x2), int(y2), depth))
except FileNotFoundError:
    print("Error: Ground truth file not found.")
    exit()

# IoU calculation function
def calculate_iou(box1, box2):
    xl = max(box1[0], box2[0])
    yl = max(box1[1], box2[1])
    xr = min(box1[2], box2[2])
    yr = min(box1[3], box2[3])

    intersection_width = max(0, xr - xl)
    intersection_height = max(0, yr - yl)
    AOI = intersection_width * intersection_height

    width_box1, height_box1 = box1[2] - box1[0], box1[3] - box1[1]
    area_box1 = width_box1 * height_box1

    width_box2, height_box2 = box2[2] - box2[0], box2[3] - box2[1]
    area_box2 = width_box2 * height_box2

    union_area = area_box1 + area_box2 - AOI
    return AOI / union_area if union_area != 0 else 0

# Load intrinsic matrix
def load_intrinsic_matrix(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    matrix_values = []
    for line in lines:
        matrix_values.extend(map(float, line.split()))

    intrinsic_matrix = np.array(matrix_values).reshape(3, 3)
    return intrinsic_matrix

# Load intrinsic matrix
intrinsic_matrix_path = "C:\\Users\\Lenovo\\Desktop\\rwu\\computer vision\\TASK_2_YOLO\\KITTI_Selection (1)\\KITTI_Selection\\calib\\006374.txt"
K = load_intrinsic_matrix(intrinsic_matrix_path)

# Camera parameters
camera_height = 1.65  # in meters
fx, fy = K[0, 0], K[1, 1]  # Focal length
cx, cy = K[0, 2], K[1, 2]  # Principal point

# Detect objects
results = model(vehicle, stream=True)
detected_boxes = []

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        if cls == 2:  # Class 2 corresponds to "car" in COCO dataset
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            detected_boxes.append((x1, y1, x2, y2, conf))

# Sort detected boxes by confidence
detected_boxes.sort(key=lambda x: x[4], reverse=True)

# IoU threshold for matching
iou_threshold = 0.5

# Iterate over each detection
for detected_box in detected_boxes:
    x1, y1, x2, y2, conf = detected_box
    y_lower = y2  # Bottom of the bounding box
    detected_depth = None

    # Depth estimation using the midpoint of the lower bound
    if y_lower != cy:
        car_height = 1.65  # Real height of a car in meters
        detected_depth = (fx * car_height) / (y_lower - cy)
    
    # Match with ground truth boxes
    max_iou = 0
    best_gt_idx = -1
    for gt_idx, gt_box in enumerate(ground_truth_boxes):
        iou = calculate_iou(gt_box[:4], detected_box[:4])
        if iou > max_iou:
            max_iou = iou
            best_gt_idx = gt_idx

    # Only display if matched with a ground truth box
    if max_iou >= iou_threshold:
        gt_x1, gt_y1, gt_x2, gt_y2, gt_depth = ground_truth_boxes[best_gt_idx]

        # Draw detection bounding box and YOLO depth
        cv2.rectangle(vehicle, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for detection
        text_yolo = f"YOLO - {detected_depth:.2f} m"
        text_gt = f"GT - {gt_depth:.2f} m"
        combined_text = f"{text_yolo}\n{text_gt}"
        
        # Draw white rectangle for text background
        cv2.rectangle(vehicle, (x1, y1 - 50), (x1 + 200, y1), (255, 255, 255), -1)
        
        # Write text for YOLO and GT inside the rectangle
        lines = combined_text.split("\n")
        for i, line in enumerate(lines):
            cv2.putText(vehicle, line, (x1 + 5, y1 - 5 - (i * 15)), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Draw ground truth bounding box
        cv2.rectangle(vehicle, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)  # Green for ground truth

# Save and display the final image
output_image_path = "C:\\Users\\Lenovo\\Desktop\\rwu\\computer vision\\TASK_2_YOLO\\runs\\depth_estimation\\006374.png"
try:
    cv2.imwrite(output_image_path, vehicle)
    print(f"Output saved to: {output_image_path}")
except Exception as e:
    print(f"Error saving output image: {e}")

cv2.imshow("Final Output", vehicle)
cv2.waitKey(0)
cv2.destroyAllWindows()
