import cv2
import csv
import os
from datetime import datetime

# Load trained face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

# Load Haar Cascade
face_detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

# Load student names
names = {}

with open("students.txt", "r") as file:
    for line in file:
        line = line.strip()

        if line:
            student_id, student_name = line.split(",", 1)
            names[int(student_id)] = student_name


# Attendance file
attendance_file = "attendance.csv"

# Create CSV if it does not exist
if not os.path.exists(attendance_file):

    with open(attendance_file, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Date",
            "Time",
            "Student ID",
            "Name",
            "Status"
        ])


# Today's date
today = datetime.now().strftime("%Y-%m-%d")

# Find students already marked today
marked_today = set()

with open(attendance_file, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        if row["Date"] == today:
            marked_today.add(int(row["Student ID"]))


# Open webcam
cam = cv2.VideoCapture(0)

print("Face Attendance System Started")
print("Press Q to exit.")

while True:

    ret, img = cam.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        student_id, confidence = recognizer.predict(
            gray[y:y+h, x:x+w]
        )

        # Lower LBPH confidence value = better match
        if confidence < 80:

            name = names.get(
                student_id,
                "Unknown"
            )

            recognition_score = int(100 - confidence)

            status = "Present"

            # Mark attendance only once per day
            if student_id not in marked_today:

                current_time = datetime.now().strftime(
                    "%H:%M:%S"
                )

                with open(
                    attendance_file,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        today,
                        current_time,
                        student_id,
                        name,
                        "Present"
                    ])

                marked_today.add(student_id)

                print(
                    f"Attendance marked: {name}"
                )

        else:

            name = "Unknown"
            recognition_score = 0
            status = "Not Registered"

        # Draw face rectangle
        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            img,
            "Name: " + name,
            (x, y - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Display status
        cv2.putText(
            img,
            "Status: " + status,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        # Display recognition score
        cv2.putText(
            img,
            "Score: " + str(recognition_score) + "%",
            (x, y + h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    # Display title
    cv2.putText(
        img,
        "FACE ATTENDANCE SYSTEM",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Face Recognition Attendance",
        img
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cam.release()
cv2.destroyAllWindows()

print("Attendance system stopped.")