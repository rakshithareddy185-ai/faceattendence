import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

detector = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

dataset_path = "dataset"


def get_images_and_labels(path):

    image_paths = [
        os.path.join(path, file)
        for file in os.listdir(path)
        if file.lower().endswith(".jpg")
    ]

    face_samples = []
    ids = []

    for image_path in image_paths:

        pil_image = Image.open(image_path).convert("L")

        image_numpy = np.array(pil_image, "uint8")

        student_id = int(
            os.path.split(image_path)[-1].split(".")[1]
        )

        faces = detector.detectMultiScale(image_numpy)

        for (x, y, w, h) in faces:

            face_samples.append(
                image_numpy[y:y+h, x:x+w]
            )

            ids.append(student_id)

    return face_samples, ids


print("Training faces...")

faces, ids = get_images_and_labels(dataset_path)

recognizer.train(faces, np.array(ids))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.write("trainer/trainer.yml")

print("Training completed successfully!")
print("Total faces trained:", len(faces))