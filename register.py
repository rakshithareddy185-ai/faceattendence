import cv2
import os

student_id = input("Enter Student ID: ")
student_name = input("Enter Student Name: ")

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

cam = cv2.VideoCapture(0)

face_detector = cv2.CascadeClassifier(
     "haarcascade_frontalface_default.xml"
)

count = 0

print("Look at the camera...")
print("Collecting face images...")

while True:

    ret, img = cam.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        count += 1

        filename = f"{dataset_path}/User.{student_id}.{count}.jpg"

        cv2.imwrite(
            filename,
            gray[y:y+h, x:x+w]
        )

        cv2.rectangle(
            img,
            (x, y),
            (x+w, y+h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            img,
            f"Samples: {count}/30",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Register Face", img)

    key = cv2.waitKey(100) & 0xff

    if key == 27:
        break

    if count >= 30:
        break

print("Face registration completed.")

cam.release()
cv2.destroyAllWindows()