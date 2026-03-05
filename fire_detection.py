import cv2
from ultralytics import YOLO
import math

# Load custom YOLO models
fire_model = YOLO("fire_yolov8s.pt")    # Stove fire + bonfire trained model
child_model = YOLO("yolov8s.pt")  # Only children class

SAFE_DISTANCE_STOVE = 80      # smaller safe distance for stove
SAFE_DISTANCE_BONFIRE = 200   # bigger safe distance for bonfire

cap = cv2.VideoCapture("child_fire.mp4")

def calculate_distance(box1, box2):
    x1, y1, x2, y2 = box1
    cx1 = (x1 + x2) / 2
    cy1 = (y1 + y2) / 2
    x1f, y1f, x2f, y2f = box2
    cx2 = (x1f + x2f) / 2
    cy2 = (y1f + y2f) / 2
    return math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fire_results = fire_model.predict(frame, conf=0.3)[0]
    child_results = child_model.predict(frame, conf=0.4)[0]

    fire_boxes = []
    child_boxes = []

    # Extract fire boxes
    for box, cls in zip(fire_results.boxes.xyxy, fire_results.boxes.cls):
        fire_boxes.append((box.tolist(), int(cls)))

    # Extract child boxes
    for box in child_results.boxes.xyxy:
        child_boxes.append(box.tolist())

    alert_status = "SAFE"

    for c_box in child_boxes:
        for f_box, f_cls in fire_boxes:
            distance = calculate_distance(c_box, f_box)

            # Adjust safe distance based on fire type
            safe_dist = SAFE_DISTANCE_STOVE if f_cls == 0 else SAFE_DISTANCE_BONFIRE

            if distance < safe_dist / 2:
                alert_status = "EMERGENCY"
            elif distance < safe_dist:
                if alert_status != "EMERGENCY":
                    alert_status = "DANGER"

            # Draw fire box
            x1, y1, x2, y2 = [int(i) for i in f_box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, "FIRE", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Draw child box
        x1, y1, x2, y2 = [int(i) for i in c_box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255,0,0), 2)
        cv2.putText(frame, "CHILD", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

    # Display alert
    color = (0,255,0) if alert_status=="SAFE" else (0,165,255) if alert_status=="DANGER" else (0,0,255)
    cv2.putText(frame, f"STATUS: {alert_status}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Child Near Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()