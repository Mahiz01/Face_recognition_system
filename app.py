from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
from keras.models import load_model

app = Flask(__name__)

# Path to the folder containing training images
path = 'Training_images'
if not os.path.exists(path):
    os.makedirs(path)  # Create the directory if it doesn't exist

# Load emotion model
emotion_model_path = 'models/fer-master/fer-master/src/fer/data/emotion_model.hdf5'  # Update this path
emotion_model = load_model(emotion_model_path)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')  # Update path if necessary

def markAttendance(name, emotion):
    with open('Attendance.csv', 'a') as f:  # Open in append mode
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.writelines(f'{name},{dtString},{emotion}\n')  # Write name, time, and emotion to file

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_name = request.form['name']
        register_user(user_name)
        return redirect(url_for('index'))
    return render_template('register.html')

def register_user(name):
    cap = cv2.VideoCapture(0)  # Start video capture
    while True:
        success, img = cap.read()  # Read frame from webcam
        if success:
            cv2.putText(img, f"Registering: {name}", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Register User', img)  # Show the registration window

            if cv2.waitKey(1) & 0xFF == ord('c'):  # Capture on 'c'
                img_name = f"{name}.jpg"
                cv2.imwrite(os.path.join(path, img_name), img)  # Save the captured image
                break  # Exit the loop

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close the registration window

@app.route('/capture')
def capture():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames():
    images, classNames = load_images()
    encodeListKnown = findEncodings(images)
    cap = cv2.VideoCapture(0)  # Start video capture

    while True:
        success, img = cap.read()  # Read frame from webcam
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize for faster processing
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        facesCurFrame = face_recognition.face_locations(imgS)  # Find faces in current frame
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)  # Get encodings for found faces

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)  # Compare with known encodings
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)  # Get distances
            matchIndex = np.argmin(faceDis)  # Get the index of the closest match

            if matches[matchIndex]:  # If there's a match
                name = classNames[matchIndex].upper()  # Get the name
                y1, x2, y2, x1 = faceLoc  # Get face location
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scale back up
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw rectangle around face
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)  # Draw filled rectangle for name
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)  # Display name

                # Extract the face region for emotion detection
                face_region = img[y1:y2, x1:x2]
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                emotion_index = detect_emotion(gray_face)  # Detect emotion

                # Map the emotion index to emotion label (adjust this based on your model)
                emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
                emotion_label = emotions[emotion_index]

                markAttendance(name, emotion_label)  # Mark attendance with emotion

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()  # Release the webcam

def load_images():
    images = []
    classNames = []
    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])  # Get the file name without extension

    return images, classNames

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        encode = face_recognition.face_encodings(img)  # Get face encodings
        if encode:  # Check if encoding is found
            encodeList.append(encode[0])  # Append the first encoding
    return encodeList

def detect_emotion(face):
    face = cv2.resize(face, (64, 64))  # Resize to the expected size of the model
    face = face.astype('float32') / 255  # Normalize the pixel values
    face = np.expand_dims(face, axis=-1)  # Add an extra dimension for the channel (grayscale)
    face = np.expand_dims(face, axis=0)  # Expand dimensions to match input shape
    predictions = emotion_model.predict(face)  # Predict the emotion
    return np.argmax(predictions)  # Return the index of the highest prediction

if __name__ == '__main__':
    app.run(debug=True)
