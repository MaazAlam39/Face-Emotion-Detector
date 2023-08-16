from fastapi import FastAPI, UploadFile, File
import cv2
from keras.models import model_from_json
import numpy as np

app = FastAPI()

# Load the model and other necessary code (similar to what you provided)
# ...

@app.post("/predict/")
async def predict_emotion(file: UploadFile):
    image = np.fromstring(file.file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            face = gray[q:q + s, p:p + r]
            face = cv2.resize(face, (48, 48))
            img = extract_features(face)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(image, prediction_label, (p - 10, q - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            cv2.rectangle(image, (p, q), (p + r, q + s), (255, 0, 0), 2)
    except cv2.error:
        pass
    _, im_buf_arr = cv2.imencode(".jpg", image)
    return im_buf_arr.tobytes()
