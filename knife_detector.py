from ultralytics import YOLO
import cv2
import time

class KnifeDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")
        self.last_emergency_time = 0
    
    def detect(self, frame):

        results = self.model(frame)

        knife_box = None
        person_box = None

        for r in results:

            for box in r.boxes:

                cls = int(box.cls[0])
                label = self.model.names[cls]

                # Treat remote as knife
                if label == "remote":
                    label = "knife"

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw box
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

                if label == "knife":
                    knife_box = (x1,y1,x2,y2)

                if label == "person":
                    person_box = (x1,y1,x2,y2)

        # ---- EMERGENCY MEMORY CHECK FIRST ----
        if time.time() - self.last_emergency_time < 3:
            return "EMERGENCY"

        # ---- LOGIC ----
        if knife_box is None:
            return "SAFE"

        if person_box is None:
            return "DANGER"

        kx1,ky1,kx2,ky2 = knife_box
        px1,py1,px2,py2 = person_box

        overlap = (
            kx1 < px2 and kx2 > px1 and
            ky1 < py2 and ky2 > py1
        )

        if overlap:
            self.last_emergency_time = time.time()
            return "EMERGENCY"

        return "DANGER"


    