import cv2
import numpy as np
import time
import requests
import sqlite3
from ultralytics import YOLO

fire_model = YOLO("fire_yolov8s.pt")
print("Fire model loaded...")


# -------------------------------------------------
# CONFIG
# -------------------------------------------------

VIDEO_PATH = "child_playingScissors.mp4"     # recorded video
ENVIRONMENT = "home"          # "house" or "hospital"

DASHBOARD_URL = "http://localhost:5000/alert"

ALERT_COOLDOWN = 15   # seconds

# -------------------------------------------------
# Database (local evidence log)
# -------------------------------------------------

conn = sqlite3.connect("events.db", check_same_thread=False)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS events(
time TEXT,
environment TEXT,
decision TEXT,
reason TEXT
)
""")
conn.commit()

def store_event(payload):
    cur.execute(
        "INSERT INTO events VALUES (?,?,?,?)",
        (payload["time"], payload["environment"],
         payload["decision"], payload["reason"])
    )
    conn.commit()

# -------------------------------------------------
# Fake SMS (for now)
# -------------------------------------------------

def send_fake_sms(msg):
    print(" SMS SENT:", msg)

# -------------------------------------------------
# Load model
# -------------------------------------------------

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Video could not be opened:", VIDEO_PATH)
    exit()

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def is_near_fire_box(person_box, fire_boxes, dist_thresh=140):

    x1,y1,x2,y2 = person_box
    px = (x1+x2)//2
    py = (y1+y2)//2

    for fb in fire_boxes:
        fx1,fy1,fx2,fy2 = fb
        fx = (fx1+fx2)//2
        fy = (fy1+fy2)//2

        d = np.sqrt((px-fx)**2 + (py-fy)**2)

        if d < dist_thresh:
            return True

    return False

def is_child(box, frame_h):
    x1,y1,x2,y2 = box
    return (y2-y1) < 0.55 * frame_h

def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    inter = iw * ih
    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)

    union = area_a + area_b - inter
    if union == 0:
        return 0

    return inter / union

def is_fallen(box):
    x1,y1,x2,y2 = box
    w = x2-x1
    h = y2-y1
    return w > 1.2*h

# -------------------------------------------------
# Tracking state
# -------------------------------------------------

track = {}
last_alert_time = 0

print("NeuroSafe AI started...")

# -------------------------------------------------
# Main loop
# -------------------------------------------------

while True:

    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
     print("Empty frame skipped")
     continue

    h,w,_ = frame.shape
    now = time.time()

    result = model(frame, conf=0.25, verbose=False)[0]
    fire_result = fire_model(frame, conf=0.4, verbose=False)[0]
    fire_detected = len(fire_result.boxes) > 0
    fire_boxes = []

    for fb in fire_result.boxes:
     fx1,fy1,fx2,fy2 = fb.xyxy[0].cpu().numpy().astype(int)
     fire_boxes.append((fx1,fy1,fx2,fy2))

    persons = []
    scissors  = []

    for b in result.boxes:
        cls = int(b.cls[0])
        name = model.names[cls]
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)

        if name=="person":
            persons.append((x1,y1,x2,y2))
        if name in ["scissors"]:
            scissors.append((x1,y1,x2,y2))

    decision = "SAFE"
    reason   = ""

    # hospital exit zone (right 15%)
    exit_zone_x = int(0.85*w)

    # -------------------------------------------------
    # Reasoning per person
    # -------------------------------------------------

    for p in persons:

        x1,y1,x2,y2 = p
        pid = str(p)

        if pid not in track:
            track[pid] = {"fall_start":None}

        state = track[pid]

        child  = is_child(p,h)
        fallen = is_fallen(p)
        near_fire = fire_detected and is_near_fire_box(p, fire_boxes)

        # -------- child holding scissors
        child_with_scissors = False

        if child:
           for k in scissors:
             if box_iou(p, k) > 0.02:
              child_with_scissors = True
              break

        # -------- fall timing
        if fallen:
            if state["fall_start"] is None:
                state["fall_start"] = now
        else:
            state["fall_start"] = None

        fall_time = 0
        if state["fall_start"]:
            fall_time = now - state["fall_start"]

        # -------------------------------------------------
        # RULES
        # -------------------------------------------------

        # R1 – Child near fire
        if child and near_fire:
            decision = "DANGER"
            reason   = "Child near fire"

        # R2 – Child holding scissors
        elif child_with_scissors:
            decision = "DANGER"
            reason   = "Child holding sharp object"

        # R3 – Patient wandering near exit (hospital only)
        elif ENVIRONMENT=="hospital" and ((x1+x2)//2 > exit_zone_x):
            decision = "DANGER"
            reason   = "Patient wandering near exit"

        # R4 – Fall emergency (elderly / patient)
        elif fallen and fall_time > (10 if ENVIRONMENT=="hospital" else 20):
            decision = "EMERGENCY"
            reason   = "Person fallen and not moving"

        # R5 – Short fall / yoga / exercise
        elif fallen:
            decision = "SAFE"
            reason   = "Temporary posture change"

        # draw
        color=(0,255,0)
        if decision=="DANGER": color=(0,165,255)
        if decision=="EMERGENCY": color=(0,0,255)

        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    # -------------------------------------------------
    # Alert pipeline
    # -------------------------------------------------

    if decision in ["DANGER","EMERGENCY"] and now-last_alert_time>ALERT_COOLDOWN:

        payload={
            "time":time.ctime(),
            "environment":ENVIRONMENT,
            "decision":decision,
            "reason":reason
        }

        try:
            requests.post(DASHBOARD_URL,json=payload,timeout=1)
        except:
            pass

        store_event(payload)

        if decision=="EMERGENCY":
            send_fake_sms(f"{decision} : {reason}")

        last_alert_time=now

    label = f"{decision} - {reason}"
    cv2.putText(frame,label,(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("NeuroSafe AI",frame)

    if cv2.waitKey(25)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()