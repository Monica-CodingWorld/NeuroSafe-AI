import cv2

from neural.knife_detector import KnifeDetector
from symbolic.kidknife_reasoning import KidKnifeReasoning


detector = KnifeDetector()
reasoner = KidKnifeReasoning()

# Load video file
cap = cv2.VideoCapture("videos/kid_with_knife.mp4")

print("Video Started...")

while True:

    ret, frame = cap.read()

    if not ret:
        break
    
    state = detector.detect(frame)

    alert, message = reasoner.analyze(state)

    # knife, person = detector.detect(frame)

    # alert, message = reasoner.analyze(knife, person)

    if alert:
        color = (0,0,255)
    else:
        color = (0,255,0)

    cv2.putText(frame,
                message,
                (30,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                3)

    cv2.imshow("NeuroSafe AI - Kid Knife Detection",frame)

    # Slow video a bit for demo
    if cv2.waitKey(30) == 27:
        break


cap.release()
cv2.destroyAllWindows()