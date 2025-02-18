import cv2

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)  # Force AVFoundation backend

if not cap.isOpened():
    print("Unable to access iPhone camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("iPhone Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
