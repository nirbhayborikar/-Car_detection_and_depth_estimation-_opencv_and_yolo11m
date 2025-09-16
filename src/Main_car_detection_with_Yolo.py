from ultralytics import YOLO   # Import YOLO object detection model from ultralytics library
import cv2                     # OpenCV library for image loading, drawing, and visualization
import math                    # For mathematical operations (ceil, etc.)
import cvzone                  # CVZone provides utilities like putTextRect for text with background

# Load the vehicle image
vehicle = cv2.imread('/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/images/006374.png')  # Read image from path
ground_truth_file = "/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/labels/006374.txt"  # Path to ground truth label file

# Check if image is loaded correctly
if vehicle is None:
    print("Error: Image file not found or could not be loaded.")
    exit()

# Load YOLO model
model = YOLO("yolo11m.pt")  # Load YOLOv8 medium model pretrained weights

# Class names (COCO dataset 80 classes)
className = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

# Load ground truth bounding boxes
ground_truth_boxes = []  # Will store GT bounding boxes
try:
    with open(ground_truth_file, 'r') as file:  # Open ground truth label file
        for line in file:  # Iterate over each line
            parts = line.strip().split()  # Split the line into tokens
            x3, y3, x4, y4 = map(float, parts[1:5])  # Extract bounding box coordinates
            ground_truth_boxes.append((int(x3), int(y3), int(x4), int(y4)))  # Save as integers
except FileNotFoundError:  # Handle case where GT file is missing
    print("Error: Ground truth file not found.")
    exit()

# IoU calculation function
def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    Input: box1, box2 in format (x1, y1, x2, y2)
    Output: IoU score (float between 0 and 1)
    """
    xl = max(box1[0], box2[0])  # Intersection left
    yl = max(box1[1], box2[1])  # Intersection top
    xr = min(box1[2], box2[2])  # Intersection right
    yr = min(box1[3], box2[3])  # Intersection bottom

    intersection_width = max(0, xr - xl)   # Width of overlap
    intersection_height = max(0, yr - yl)  # Height of overlap
    AOI = intersection_width * intersection_height  # Area of Intersection

    # Area of first box
    width_box1, height_box1 = box1[2] - box1[0], box1[3] - box1[1]
    area_box1 = width_box1 * height_box1

    # Area of second box
    width_box2, height_box2 = box2[2] - box2[0], box2[3] - box2[1]
    area_box2 = width_box2 * height_box2

    # Union area
    union_area = area_box1 + area_box2 - AOI
    return AOI / union_area if union_area != 0 else 0

# Detect cars
results = model(vehicle, stream=True)  # Run YOLO inference (stream=True yields iterable results)
detected_boxes = []  # Store detections (with confidence)
one_detected = []   # Store just bounding box coordinates

# Iterate through YOLO results
for r in results:  # Each result object corresponds to detections in image
    for box in r.boxes:  # Loop through each detected bounding box
        cls = int(box.cls[0])  # Get class ID
        currentClass = className[cls]  # Get class name from ID

        if currentClass == "car":  # Process only cars
            # Extract coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box (top-left, bottom-right)
            conf = math.ceil(box.conf[0] * 100) / 100  # Round confidence to 2 decimal places
            detected_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))  # Save detection
            one_detected.append((int(x1), int(y1), int(x2), int(y2)))  # Save coordinates only

# Sort detected boxes by confidence in descending order
detected_boxes.sort(key=lambda x: x[4], reverse=True)

# Initialize statistics
true_positive = 0
false_positive = 0
matched_gt_boxes = set()  # Keep track of already matched GT boxes

# Prepare a list for console table display
table_data = []

# Iterate over each detection
for idx, detected_box in enumerate(detected_boxes, start=1):
    x1, y1, x2, y2, conf = detected_box  # Unpack detection
    max_iou = 0
    best_gt_idx = -1

    # Compare detection with all ground truth boxes
    for gt_idx, gt_box in enumerate(ground_truth_boxes):
        if gt_idx in matched_gt_boxes:  # Skip already matched GT boxes
            continue

        iou = calculate_iou(gt_box, detected_box[:4])  # Calculate IoU with current GT box

        if iou > 0.15:  # If IoU exceeds visualization threshold
            # Draw the detected bounding box and IoU text
            x3, y3, x4, y4, conf = detected_box
            
            # Red rectangle for detection
            cv2.rectangle(vehicle, (x3, y3), (x4, y4), (0, 0, 255), 2)

            # Text position slightly above the detected box
            text_position = (max(0, x3), max(0, y3 - 20))

            # Draw IoU value on image
            cvzone.putTextRect(vehicle, 
                               f'Car IoU: {math.ceil(iou * 100) / 100}', 
                               text_position, 
                               scale=1, 
                               thickness=1, 
                               offset=1)

            # Draw GT box (green)
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            cv2.rectangle(vehicle, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)

        # Track highest IoU and corresponding GT box
        if iou > max_iou:
            max_iou = iou
            best_gt_idx = gt_idx

    # Determine True Positive (TP) or False Positive (FP) using IoU threshold
    if max_iou >= 0.75:  # Match found
        true_positive += 1
        matched_gt_boxes.add(best_gt_idx)  # Mark GT as matched
        tp = 1
        fp = 0
    else:  # No match, count as false positive
        false_positive += 1
        tp = 0
        fp = 1

    # Calculate Precision and Recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0

    # Add row to results table
    table_data.append([idx, conf, max_iou, tp, fp, precision, recall, ground_truth_boxes[best_gt_idx] if best_gt_idx != -1 else "N/A", (x1, y1, x2, y2)])

# Calculate Average Precision (AP)
def calculate_average_precision(precision_values, recall_values):
    """
    Calculate Average Precision (AP) given precision-recall curve values.
    Formula: AP = sum(p(i) * (R(i) - R(i-1)))
    """
    AP = 0
    R0 = 0  # Previous recall
    for i in range(1, len(precision_values) + 1):
        AP += precision_values[i - 1] * (recall_values[i - 1] - R0)
        R0 = recall_values[i - 1]
    return AP

# Extract precision and recall values from table
precision_values = [row[5] for row in table_data]
recall_values = [row[6] for row in table_data]

# Calculate AP
AP = calculate_average_precision(precision_values, recall_values)

# Print detection table
print("Detection Results Table:")
print(f"{'Detection':<10}{'Confidence':<15}{'IoU':<10}{'TP':<5}{'FP':<5}{'Precision':<10}{'Recall':<10}{'GT Box':<25}{'Detected Box':<20}")
print("-" * 105)

for row in table_data:
    print(f"{row[0]:<10}{row[1]:<15.2f}{row[2]:<10.2f}{row[3]:<5}{row[4]:<5}{row[5]:<10.2f}{row[6]:<10.2f}{str(row[7]):<25}{str(row[8]):<20}")

# Print final Average Precision
print(f"\nFinal Average Precision (AP): {AP:.4f}")

#-----------------------------------------------------
# printing ap in the image
image_height, image_width, _ = vehicle.shape  # Get dimensions of the image
cvzone.putTextRect(
    vehicle, 
    f'Average Precision: {AP:.4f}',  # Draw AP text on image
    (image_width // 2 - 150, image_height - 30),  # Position text at bottom-center
    scale=1, 
    thickness=1, 
    offset=1
)

# Display image with results
cv2.imshow("Vehicle Detection", vehicle)
cv2.waitKey(0)  # Wait for key press
cv2.destroyAllWindows()  # Close all windows

