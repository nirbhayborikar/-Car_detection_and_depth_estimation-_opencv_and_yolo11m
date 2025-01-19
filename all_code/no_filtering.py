# all boxes detecting with all given groundboxes rectangle also showing no filter

from ultralytics import YOLO
import cv2
import cvzone
import math
import os

# Load the vehicle image
vehicle = cv2.imread('/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/images/006374.png')
ground_truth_file = "/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/labels/006374.txt"
output_path = '/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/Output images/NO_filtering/006374_both_box_out.png'
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
#print values of ground truth

print('The ground truth_boxes',ground_truth_boxes)

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

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        currentClass = className[cls]

        if currentClass == "car":
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_boxes.append((x1, y1, x2, y2))
                    
#print all detected boxes values             
print('the detected boxes',detected_boxes)
# Draw ground truth boxes in green
for gt_box in ground_truth_boxes:
    x1, y1, x2, y2 = gt_box
    cv2.rectangle(vehicle, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Calculate IoUs and annotate detected boxes
iou_results = []
for detected_box in detected_boxes:
    max_iou = 0
    best_gt_box = None
    current_ious = []

    for gt_box in ground_truth_boxes:
        iou = calculate_iou(gt_box, detected_box)
        current_ious.append((gt_box, iou))
        if iou > max_iou:
            max_iou = iou
            best_gt_box = gt_box

    iou_results.append({
        "detected_box": detected_box,
        "ious": current_ious,
        "max_iou": max_iou,
        "best_gt_box": best_gt_box
    })

    # Annotate detected box with IoU
    conf = math.ceil(box.conf[0] * 100) / 100
    x1, y1, x2, y2 = detected_box
    cv2.rectangle(vehicle, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #cvzone.putTextRect(vehicle, f'IoU: {max_iou:.2f}', 
     #                  (max(0, x1), max(0, y1 - 20)), scale=1, thickness=1, offset=1)

# Print IoU results in the terminal
for result in iou_results:
    print(f"Detected Box: {result['detected_box']}")
    print("IoU with Ground Truth Boxes:")
    for gt_box, iou in result['ious']:
        print(f"  Ground Truth Box: {gt_box}, IoU: {iou:.4f}")
    print(f"Max IoU: {result['max_iou']:.4f} with Ground Truth Box: {result['best_gt_box']}")
    print("-" * 50)

# Display image
print('number_of_groundboxes',len(ground_truth_boxes))
print('number of detected boxes', len(detected_boxes))
cv2.imshow("Vehicle Detection", vehicle)



#####

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
