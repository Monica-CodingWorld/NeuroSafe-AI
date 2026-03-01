import cv2
import numpy as np
import time
import requests
import sqlite3
from ultralytics import YOLO

# -------------------------------------------------
# CONFIG
# -------------------------------------------------

VIDEO_PATH = "child_holdingKnife.mp4"     # recorded video
ENVIRONMENT = "hospital"          # "house" or "hospital"

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

# -------------------------------------------------
# Utilities
# -------------------------------------------------

def detect_fire_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([10,100,100])
    upper = np.array([35,255,255])
    return cv2.inRange(hsv, lower, upper)

def is_child(box, frame_h):
    x1,y1,x2,y2 = box
    return (y2-y1) < 0.45 * frame_h

def is_near_fire(box, fire_mask, dist_thresh=140):
    x1,y1,x2,y2 = box

    px = (x1+x2)//2
    py = (y1+y2)//2

    ys, xs = np.where(fire_mask>0)
    if len(xs)==0:
        return False

    fx = int(xs.mean())
    fy = int(ys.mean())

    d = np.sqrt((px-fx)**2 + (py-fy)**2)
    return d < dist_thresh

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
    if not ret:
        break

    h,w,_ = frame.shape
    now = time.time()

    fire_mask = detect_fire_mask(frame)
    result = model(frame, conf=0.4, verbose=False)[0]

    persons = []
    knives  = []

    for b in result.boxes:
        cls = int(b.cls[0])
        name = model.names[cls]
        x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)

        if name=="person":
            persons.append((x1,y1,x2,y2))
        if name in ["knife","scissors"]:
            knives.append((x1,y1,x2,y2))

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
        near_fire = is_near_fire(p, fire_mask)

        # -------- child holding knife
        child_with_knife = False
        if child:
            for k in knives:
                kx1,ky1,kx2,ky2 = k
                if kx1>x1 and kx2<x2 and ky1>y1 and ky2<y2:
                    child_with_knife = True

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

        # R2 – Child holding knife
        elif child_with_knife:
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