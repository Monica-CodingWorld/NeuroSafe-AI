import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO Pose
model = YOLO("yolov8n-pose.pt")

VIDEO_PATH = "OldMan_fall.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

prev_hip_y = None
fall_detected = False
fall_time = None

decision = "SAFE"
reason = "Normal activity"

print("System started...")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    results = model(frame, conf=0.4)[0]

    if results.keypoints is not None:

        for i, person in enumerate(results.boxes):

            x1,y1,x2,y2 = person.xyxy[0].cpu().numpy().astype(int)

            keypoints = results.keypoints.xy[i].cpu().numpy()

            head = keypoints[0]
            l_sh = keypoints[5]
            r_sh = keypoints[6]
            l_hip = keypoints[11]
            r_hip = keypoints[12]

            if head[1] == 0:
                continue

            hip_y = (l_hip[1] + r_hip[1]) / 2
            shoulder_y = (l_sh[1] + r_sh[1]) / 2

            # --------------------------
            # BODY ORIENTATION
            # --------------------------

            body_height = abs(hip_y - shoulder_y)
            body_width = abs(l_sh[0] - r_sh[0])

            horizontal = body_width > body_height

            # --------------------------
            # FALL SPEED
            # --------------------------

            fall_speed = 0

            if prev_hip_y is not None:
                fall_speed = hip_y - prev_hip_y

            prev_hip_y = hip_y

            # --------------------------
            # FALL DETECTION
            # --------------------------

            if horizontal and fall_speed > 25 and not fall_detected:

                fall_detected = True
                fall_time = time.time()

                decision = "EMERGENCY"
                reason = "Elderly person fallen"

            # --------------------------
            # POST FALL STATE
            # --------------------------

            if fall_detected:

                elapsed = time.time() - fall_time

                if horizontal:
                    decision = "EMERGENCY"
                    reason = "Elderly person fallen"

                else:
                    decision = "DANGER"
                    reason = "Movement detected"

            else:
                decision = "SAFE"
                reason = "Normal activity"

            # --------------------------
            # DRAW BOX
            # --------------------------

            color = (0,255,0)

            if decision == "DANGER":
                color = (0,165,255)

            if decision == "EMERGENCY":
                color = (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    cv2.putText(frame,
                f"{decision} : {reason}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3)

    cv2.imshow("Fall Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()