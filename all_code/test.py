# List of image IDs

from ultralytics import YOLO
import os
import cv2
import cvzone
import math




img_ids = ['006037', '006042', '006048', '006054', '006059', '006067', '006097', '006098', '006121', '006130',
           '006206', '006211', '006227', '006253', '006291', '006310', '006312', '006315', '006329', '006374']

# Loop through each image ID
for img_id in img_ids:
    # Define the paths using the current img_id
    vehicle_path = f'/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/images/{img_id}.png'
    ground_truth_file = f'/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/labels/{img_id}.txt'
    output_path = f'/home/nirbhayborikar/Documents/RWU/CV/project/KITTI_Selection/Output images/test/{img_id}_yolo1.png'


model=YOLO("yolo11m.pt")

className = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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
vehicle = cv2.imread(vehicle_path)

results = model(vehicle, stream = True)
for r in results:
    boxes = r.boxes
    for box in boxes:
       
        # class names
       
        cls =int(box.cls[0]) # its a floating value convert it in to integer currently giving ids but we need name
        #cvzone.putTextRect(car,f'{className[cls]} conf:{conf}',(max(0,x1),max(0,y1-20)),scale=0.5,thickness=1)
        
        #if we want to detect only cars
        currentClass = className[cls]

        if currentClass == "car":


                        #for bounding box corner detection and rectangle creation
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            print('top_left_coordinate',x1,y1,'bottom_right_coordinate',x2,y2)
            
            
            cv2.rectangle(vehicle,(x1,y1),(x2,y2),(0,0,255),3) #make rectangle red by using 2 diagonal point
            # cvzone.cornerRect(car,(x1,y1,w,h), =9) #9 is thickness of corner
            # calculating confidence by using conf and ceil for rounding off
            
            #conf = box.conf[0]
            conf = math.ceil((box.conf[0]*100))/100#  here we rounding off it to 2 decimal place by using ceil function confidence value
            #cvzone.putTextRect(car,f'conf:{conf}',(max(0,x1),max(0,y1-20)))
            print('confidence',conf) #printing the confidence values
            
            #cvzone.putTextRect(vehicle,f'{currentClass} conf:{conf}',(max(0,x1),max(0,y1-20)),scale=1,thickness=1,offset = 1)
            print('classnames',cls)






    print("Number of detections:", len(boxes)) # as boxes contains number of bounding box results
cv2.imshow("camera",vehicle)

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


