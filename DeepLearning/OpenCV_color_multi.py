from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
print("import complete.")

face_classifier = cv2.CascadeClassifier(r'G:/Min enhet/test/haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'G:/Min enhet/test/model_2_1.h5')
multitask_model = load_model(r'G:/Min enhet/test/model_multitask_1.h5')

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad','surprise']
gender_labels = ['male', 'female']
race_labels = ['white', 'black', 'asian', 'indian', 'others']

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    labels = []
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_classifier.detectMultiScale(rgb_frame)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(128,128,0),2)
        roi_color = rgb_frame[y:y+h, x:x+w]
        roi_color = cv2.resize(roi_color,(48,48),interpolation=cv2.INTER_AREA)
      
    
        if np.sum(roi_color)!= 0:
            roi = roi_color.astype('float32')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            # Emotion prediction    
            emotion_prediction = emotion_classifier.predict(roi)[0]
            emotion_label = emotion_labels[emotion_prediction.argmax()]
            # Age, gender, and race prediction
            multitask_prediction = multitask_model.predict(roi)
            age = int(multitask_prediction[0][0])
            gender = gender_labels[int(multitask_prediction[1][0] > 0.5)]
            race = race_labels[multitask_prediction[2].argmax()]
            
            label_position = (x, y-10)

            cv2.putText(frame, f"Emotion: {emotion_label}", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 0), 2)
            cv2.putText(frame, f"Age: {age}", (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 0), 2)
            cv2.putText(frame, f"Gender: {gender}", (x, y-50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 0), 2)
            cv2.putText(frame, f"Race: {race}", (x, y-70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 0), 2)
        else:
            cv2.putText(frame, "No faces", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 0), 2)
    cv2.imshow('Emotion, Age, Gender, and Race Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()