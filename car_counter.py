from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

#cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)
cap = cv2.VideoCapture("./cars.mp4")


model = YOLO("./yoloweights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
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

mask = cv2.imread("mask.png")
# tracking

tracker  = Sort(max_age = 20, min_hits = 3 ,iou_threshold= 0.3)
limits = [410, 550 , 1050 , 550 ]
totalCount = 0
ids = []

while True:
    success, img = cap.read()

    img = cv2.resize(img, (1920,1080))
    imgRegion = cv2.bitwise_and(img, mask)

    results = model(imgRegion, stream = True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            # #bounding box
            # x1,y1,x2,y2 = box.xyxy[0]
            # x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            # #cv2.rectangle(img, (x1, y1) , (x2,y2) ,(255,0, 255), 3 )
            # w,h = x2-x1, y2-y1
            #
            # cvzone.cornerRect(img, (x1,y1,w,h), l = 9)
            #
            conf = math.ceil((box.conf[0]*100))/100

            cls = box.cls[0]

            currentClass = classNames[int(cls)]

            if currentClass in ["car","truck" , "bus", "motorbike"] and conf > 0.3:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1) , (x2,y2) ,(255,0, 255), 3 )
                w, h = x2 - x1, y2 - y1

                #cvzone.cornerRect(img, (x1, y1, w, h), l=9 , rt = 5)
                currentArray = np.array([x1,y1,x2,y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    cv2.line(img, (limits[0] , limits[1]), (limits[2], limits[3]), (0,0,255), 5 )
    for results in resultsTracker:
        x1,y1,x2,y2,Id = results
        print(results)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w,h = x2 - x1 , y2-y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9 , rt = 2, colorR = (255,0,255))
        cvzone.putTextRect(img, f"{int(Id)}", (max(0, x1), max(35, y1)), scale=2, thickness=3,
                           offset=10)

        cx,cy = x1+w//2, y1+h//2
        cv2.circle(img, (cx, cy), 5 , (255,0,255),cv2.FILLED)

        if limits[0]< cx <limits[2] and limits[1]-20 < cy < limits[1]+20 and int(Id) not in ids:
            totalCount += 1
            ids.append(int(Id))
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0,255,0), 5)

    cvzone.putTextRect(img, f"Count : {totalCount}", (50, 50))

    cv2.imshow("image", img)
    #cv2.imshow("imageregion", imgRegion )
    cv2.waitKey(1)
