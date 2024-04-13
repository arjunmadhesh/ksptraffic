from flask import Flask, render_template
import cv2
import numpy as np
import math
from ultralytics import YOLO
from sort import Sort

app = Flask(__name__)

@app.route('/')
def index():
    cap = cv2.VideoCapture("mad.mp4")  # For Video
    model = YOLO("../Yolo-Weights/yolov8l.pt")

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

    mask = cv2.imread("lane3.png")

    # Tracking
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    lanes = {
        "Lane 1": [850, 600, 475, 600],
        "Lane 2": [830, 600, 1100, 600],  # Define the limits for Lane 2
        "Lane 3": [1100, 600, 1300, 600],  # Define the limits for Lane 3
        "Lane 4": [1300, 600, 1600, 600]   # Define the limits for Lane 4
    }

    lane_counts = {lane: 0 for lane in lanes}

    while True:
        success, img = cap.read()
        if not success:
            break
        imgRegion = cv2.bitwise_and(img, mask)

        results = model(imgRegion, stream=True)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                for lane, limits in lanes.items():
                    if limits[0] < x1 < limits[2] and limits[1] < y1 < limits[3]:
                        if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                            lane_counts[lane] += 1

        resultsTracker = tracker.update(detections)

        for result in resultsTracker:
            x1, y1, x2, y2, id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            for lane, limits in lanes.items():
                if limits[0] < x1 < limits[2] and limits[1] < y1 < limits[3]:
                    cv2.putText(img, f'Count: {lane_counts[lane]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break

        ret, jpeg = cv2.imencode('.jpg', img)
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(index(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
