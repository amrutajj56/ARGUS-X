
import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Camera opened:", camera.isOpened())
print("Click the webcam window and press S or Q")

while camera.isOpened():
    success, frame = camera.read()

    if not success:
        print("Cannot read camera")
        break

    cv2.imshow("KEYBOARD TEST", frame)

    key = cv2.waitKey(30)

    if key != -1:
        print("KEY DETECTED:", key, flush=True)

    if key == ord("s") or key == ord("S"):
        print("S KEY WORKS!")

    if key == ord("q") or key == ord("Q"):
        print("Q KEY WORKS!")
        break

camera.release()
cv2.destroyAllWindows()
