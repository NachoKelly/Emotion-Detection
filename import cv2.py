import cv2
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

My_image = cv2.imread(r'C:\pythonn\New folder\kelly.jpg') 

gray = cv2.cvtColor(My_image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

emotion_labels = ['Happy', 'Sad', 'Neutral', 'Surprised', 'Disgust', 'Fear']

def classify_emotion(face):
    return emotion_labels[0]  
for (x, y, w, h) in faces:
    face_region = gray[y:y + h, x:x + w]

    emotion = classify_emotion(face_region)

    cv2.rectangle(My_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(My_image, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('Emotion Detection', My_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
