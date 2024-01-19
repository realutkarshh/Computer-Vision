import cv2
from ultralytics import YOLO
# loading model
model = YOLO('yolov8n.pt')

# loading video
path = 'VID-20240119-WA0011.mp4'

# read frames
cap = cv2.VideoCapture(path)
ret = True
while ret:
    ret, frame = cap.read()
    if ret:
        # detecting objects from frame
        # track object
        result = model.track(frame, persist=True)

        # plot results
        frame_ = result[0].plot()

        # visualize
        cv2.imshow('frame', frame_)
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
