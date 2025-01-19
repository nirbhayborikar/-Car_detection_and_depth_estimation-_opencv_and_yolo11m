# all okay go for depth estimation

from ultralytics import YOLO
import cv2
import math
import cvzone
import os
# Load the vehicle image
vehicle = cv2.imread('/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/images/006042.png')
ground_truth_file = "/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/labels/006042.txt"
output_path = '/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/Output images/Filtering/006042_filter_box_out.png'

if vehicle is None:
    print("Error: Image file not found or could not be loaded.")
    exit()

# Load YOLO model
model = YOLO("yolo11m.pt")

# Class names
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
ground_truth_boxes = []
try:
    with open(ground_truth_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x3, y3, x4, y4 = map(float, parts[1:5])
            ground_truth_boxes.append((int(x3), int(y3), int(x4), int(y4)))
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

# Detect cars
results = model(vehicle, stream=True)
detected_boxes = []
one_detected=[]

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        currentClass = className[cls]

        if currentClass == "car":
            # Get bounding box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil(box.conf[0] * 100) / 100
            detected_boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
            one_detected.append((int(x1),int(y1),int(x2),int(y2)))

# Sort detected boxes by confidence in descending order
detected_boxes.sort(key=lambda x: x[4], reverse=True)

# Initialize statistics
true_positive = 0
false_positive = 0
matched_gt_boxes = set()

# Prepare a list for console table display
table_data = []

# Iterate over each detection
for idx, detected_box in enumerate(detected_boxes, start=1):
    x1, y1, x2, y2, conf = detected_box
    max_iou = 0
    best_gt_idx = -1

    # Compare with ground truth boxes
    for gt_idx, gt_box in enumerate(ground_truth_boxes):
        if gt_idx in matched_gt_boxes:
            continue

        iou = calculate_iou(gt_box, detected_box[:4])

        if iou > 0.15:
            # Draw the detected box and display IoU as text
            x3, y3, x4, y4, conf = detected_box  # Detected box coordinates and confidence
            
            # Red rectangle for the detected box
            cv2.rectangle(vehicle, (x3, y3), (x4, y4), (0, 0, 255), 2)  # Red box for detected

            # Calculate IoU text position slightly above the top-left corner of the box
            text_position = (max(0, x3), max(0, y3 - 20))

            # Display the IoU value using cvzone
            '''cvzone.putTextRect(vehicle, 
                               f'Car IoU: {math.ceil(iou * 100) / 100}',  # Round and display IoU
                               text_position, 
                               scale=1, 
                               thickness=1, 
                               offset=1)'''

            # Green box for the ground truth (this will be drawn if IoU > 0.15)
            gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
            cv2.rectangle(vehicle, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 255, 0), 2)  # Green box for GT

        if iou > max_iou:
            max_iou = iou
            best_gt_idx = gt_idx

    # Determine TP or FP based on IoU threshold
    if max_iou >= 0.75:
        true_positive += 1
        matched_gt_boxes.add(best_gt_idx)
        tp = 1
        fp = 0
    else:
        false_positive += 1
        tp = 0
        fp = 1

    # Calculate Precision and Recall
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / len(ground_truth_boxes) if len(ground_truth_boxes) > 0 else 0

    # Prepare table row for console
    table_data.append([idx, conf, max_iou, tp, fp, precision, recall, ground_truth_boxes[best_gt_idx] if best_gt_idx != -1 else "N/A", (x1, y1, x2, y2)])

# Calculate Average Precision (AP) using the provided formula
def calculate_average_precision(precision_values, recall_values):
    AP = 0
    R0 = 0  # Initial recall
    for i in range(1, len(precision_values) + 1):
        AP += precision_values[i - 1] * (recall_values[i - 1] - R0)
        R0 = recall_values[i - 1]
    return AP

# Extract precision and recall values for AP calculation
precision_values = [row[5] for row in table_data]
recall_values = [row[6] for row in table_data]

# Calculate AP
AP = calculate_average_precision(precision_values, recall_values)

# Print table and average precision
print("Detection Results Table:")
print(f"{'Detection':<10}{'Confidence':<15}{'IoU':<10}{'TP':<5}{'FP':<5}{'Precision':<10}{'Recall':<10}{'GT Box':<25}{'Detected Box':<20}")
print("-" * 105)

for row in table_data:
    print(f"{row[0]:<10}{row[1]:<15.2f}{row[2]:<10.2f}{row[3]:<5}{row[4]:<5}{row[5]:<10.2f}{row[6]:<10.2f}{str(row[7]):<25}{str(row[8]):<20}")

# Print final average precision
print(f"\nFinal Average Precision (AP): {AP:.4f}")


#-----------------------------------------------------
# printing ap in the image
image_height, image_width, _ = vehicle.shape  # Get dimensions of the image
'''cvzone.putTextRect(
    vehicle, 
    f'Average Precision: {AP:.4f}',  # Display AP with four decimal places
    (image_width // 2 - 150, image_height - 30),  # Bottom-center position
    scale=1, 
    thickness=1, 
    offset=1
)'''

# Display image
cv2.imshow("Vehicle Detection", vehicle)
####



# Define the output path


# Ensure the directory exists
output_dir = os.path.dirname(output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # Create the directory if it doesn't exist

# Assuming 'vehicle' is your image (make sure it is loaded correctly)
# Example: vehicle = cv2.imread('vehicle_image.jpg')

if vehicle is None:
    print("Error: Unable to load image. Please check the image path.")
else:
    # Save the image
    success = cv2.imwrite(output_path, vehicle)

    if success:
        print(f"Image saved successfully at: {output_path}")
    else:
        print("Error: Image could not be saved.")


####
cv2.waitKey(0)
cv2.destroyAllWindows()
