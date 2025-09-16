import cv2   # OpenCV library for image processing and visualization
import math  # Provides mathematical functions like ceil, floor, etc.
import cvzone  # CVZone - helper functions for computer vision tasks
import numpy as np  # NumPy library for numerical computations and matrices
from ultralytics import YOLO  # Import YOLO object detection model from ultralytics

# Load the vehicle image
image_path = "/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/images//006374.png"
vehicle = cv2.imread(image_path)  # Reads the image from the given path into a numpy array
ground_truth_file = "/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/labels/006374.txt"

# Check if image loaded successfully
if vehicle is None:
    print("Error: Image file not found or could not be loaded.")
    exit()

# Load YOLO model
model = YOLO("yolo11m.pt")  # Load pretrained YOLOv8 model with medium size weights

# Load ground truth bounding boxes
ground_truth_boxes = []  # List to store ground truth bounding boxes
try:
    with open(ground_truth_file, 'r') as file:  # Open label file for reading
        for line in file:  # Loop through each line in label file
            parts = line.strip().split()  # Split line into components
            x1, y1, x2, y2, depth = map(float, parts[1:6])  # Extract bounding box and depth
            ground_truth_boxes.append((int(x1), int(y1), int(x2), int(y2), depth))  # Store in list
except FileNotFoundError:  # Handle missing label file
    print("Error: Ground truth file not found.")
    exit()

# IoU calculation function
def calculate_iou(box1, box2):
    """
    This function calculates Intersection over Union (IoU) between two bounding boxes.
    Input: box1 and box2 as (x1, y1, x2, y2)
    Output: IoU score (float between 0 and 1)
    """
    xl = max(box1[0], box2[0])  # Left boundary of intersection
    yl = max(box1[1], box2[1])  # Top boundary of intersection
    xr = min(box1[2], box2[2])  # Right boundary of intersection
    yr = min(box1[3], box2[3])  # Bottom boundary of intersection

    intersection_width = max(0, xr - xl)  # Width of intersection area
    intersection_height = max(0, yr - yl)  # Height of intersection area
    AOI = intersection_width * intersection_height  # Area of intersection

    # Area of first bounding box
    width_box1, height_box1 = box1[2] - box1[0], box1[3] - box1[1]
    area_box1 = width_box1 * height_box1

    # Area of second bounding box
    width_box2, height_box2 = box2[2] - box2[0], box2[3] - box2[1]
    area_box2 = width_box2 * height_box2

    # Union = sum of areas - intersection
    union_area = area_box1 + area_box2 - AOI
    return AOI / union_area if union_area != 0 else 0  # Return IoU

# Load intrinsic matrix
def load_intrinsic_matrix(filepath):
    """
    This function loads the camera intrinsic matrix from a file.
    Input: filepath (string) - path to calibration file
    Output: intrinsic matrix (3x3 numpy array)
    """
    with open(filepath, 'r') as f:  # Open calibration file
        lines = f.readlines()  # Read all lines

    matrix_values = []  # Store numeric values
    for line in lines:
        matrix_values.extend(map(float, line.split()))  # Convert line to float values

    intrinsic_matrix = np.array(matrix_values).reshape(3, 3)  # Convert to 3x3 matrix
    return intrinsic_matrix

# Load intrinsic matrix
intrinsic_matrix_path = "home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/calib/006374.txt"
K = load_intrinsic_matrix(intrinsic_matrix_path)  # Get camera intrinsic matrix

# Camera parameters
camera_height = 1.65  # in meters (height of camera above ground)
fx, fy = K[0, 0], K[1, 1]  # Extract focal lengths from intrinsic matrix
cx, cy = K[0, 2], K[1, 2]  # Extract principal point from intrinsic matrix

# Detect objects
results = model(vehicle, stream=True)  # Run YOLO inference on the input image, returns stream of results
detected_boxes = []  # List to store detected bounding boxes

# Loop through results to extract bounding boxes
for r in results:  # Each r contains detection results for one frame
    for box in r.boxes:  # Loop through detected bounding boxes
        cls = int(box.cls[0])  # Class ID of the detected object
        if cls == 2:  # Class 2 corresponds to "car" in COCO dataset
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = math.ceil(box.conf[0] * 100) / 100  # Confidence score (rounded)
            detected_boxes.append((x1, y1, x2, y2, conf))  # Store detection

# Sort detected boxes by confidence (highest first)
detected_boxes.sort(key=lambda x: x[4], reverse=True)

# IoU threshold for matching
iou_threshold = 0.5

# Iterate over each detection, and find the distance between camera and detected car
for detected_box in detected_boxes:
    x1, y1, x2, y2, conf = detected_box  # Unpack detected box
    y_lower = y2  # Bottom of the bounding box (used for depth estimation)
    detected_depth = None  # Initialize depth variable

    # Depth estimation using the midpoint of the lower bound
    if y_lower != cy:
        car_height = 1.65  # Real-world height of a car in meters
        detected_depth = (fx * car_height) / (y_lower - cy)  # Pinhole camera depth estimation formula
    
    # Match with ground truth boxes
    max_iou = 0
    best_gt_idx = -1
    for gt_idx, gt_box in enumerate(ground_truth_boxes):  # Loop through ground truth boxes
        iou = calculate_iou(gt_box[:4], detected_box[:4])  # Compute IoU
        if iou > max_iou:  # Keep track of best match
            max_iou = iou
            best_gt_idx = gt_idx

    # Only display if matched with a ground truth box
    if max_iou >= iou_threshold:
        gt_x1, gt_y1, gt_x2, gt_y2, gt_depth = ground_truth_boxes[best_gt_idx]

        # Draw detection bounding box and YOLO depth
        cv2.rectangle(vehicle, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for detection box
        text_yolo = f"YOLO - {detected_depth:.2f} m"  # Estimated depth from YOLO
        text_gt = f"GT - {gt_depth:.2f} m"  # Ground truth depth
        combined_text = f"{text_yolo}\n{text_gt}"  # Combine both texts
        
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
output_image_path = "home/nirbhayborikar/Documents/RWU/CV/project/output/images/006374.png"
try:
    cv2.imwrite(output_image_path, vehicle)  # Save final annotated image
    print(f"Output saved to: {output_image_path}")
except Exception as e:
    print(f"Error saving output image: {e}")

cv2.imshow("Final Output", vehicle)  # Show final output in a window
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()  # Close all OpenCV windows

