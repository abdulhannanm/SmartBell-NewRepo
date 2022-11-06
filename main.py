from flask import Flask, render_template, request, url_for, redirect, session, Response
from flask_mysqldb import MySQL
import cv2
import numpy as np
import sys
from flask_ngrok import run_with_ngrok
import smtplib
import ssl
from email.message import EmailMessage    
import cv2
import imutils
import struct
import pickle
import datetime
from threading import Thread
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
from twilio.rest import Client
import config
import face_recognition
import os
import time
from flask_ngrok import run_with_ngrok
# import pymysql

app = Flask(__name__)

# pymysql.install_as_MySQLdb()

movement_count = 0

run_with_ngrok(app)

phone_index = []

app.config["MYSQL_HOST"] = 'localhost'
app.config["MYSQL_USER"] = "root"
app.config['MYSQL_PASSWORD'] = "ffrn1234"
app.config['MYSQL_DB'] = "smartbell_info"
app.config['MYSQL_CURSORCLASS'] = "DictCursor"

mysql = MySQL(app)
app.secret_key = "smartbell"


#normal video stream

def norm_capture():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            frame1 = buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n')


#motion detection stream

def motion_detection():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame1 = cap.read()
        success, frame2 = cap.read()
        
        if not success:
            break
        else:
            global movement_count
            global phone_number 
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)
                if cv2.contourArea(contour) < 20000:
                    continue
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                movement_count = movement_count + 1
                print(movement_count)
                if movement_count == 20:
                    movement_count = 0
                    # alert(phone_index[0])
                    continue
                else:
                    break
            ret, buffer = cv2.imencode(".jpg", frame1)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#weapon detection stream

def weapon_detection():
    net = cv2.dnn.readNet("yolov3_training_2000.weights", "yolov3_testing.cfg")
    classes = ["Weapon"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    movement_count = 0

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break
        else:
            height, width, channels = img.shape
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop = False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes =[]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3]*height)
                        x = int(center_y - w/2)
                        y = int(center_x - h/2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            print(indexes)
            if indexes == 0: print("weapon detected")
            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            ret, buffer = cv2.imencode(".jpg", img)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def face_detection():
    video_capture = cv2.VideoCapture(0)

    # Load a sample picture and learn how to recognize it.
    a = list(os.walk("rayyan/"))

    print(a[0][2])
    # for i in range(len(list(os.walk("rayyan/")))):
    known_face_encodings = [

    ]
    known_face_names = [
        "Rayyan", "Aadrij"
    ]
    #     print(i)
    for i in a[0][2]:
        print(i)
        image = face_recognition.load_image_file("rayyan/" + i)
        face_encoding = face_recognition.face_encodings(image)[0]

        known_face_encodings.append(face_encoding)
    # rayyan_image = face_recognition.load_image_file("rayyan/rayyan_1.png")

    # rayyan_face_encoding = face_recognition.face_encodings(rayyan_image)[0]


    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    clock = time.perf_counter()

    while True:
        # Grab a single frame of video
        a = clock
        success, frame1 = video_capture.read()
        if not success:
            break
        else:
        # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame1, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame1, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame1, (left, bottom - 35),
                            (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame1, name, (left + 6, bottom - 6),
                            font, 1.0, (255, 255, 255), 1)
            ret, buffer = cv2.imencode('.jpg', frame1)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# alert function
def alert1(number):
    account_sid = config.account_sid
    auth_token = config.auth_token
    twilio_number = config.twilio_number
    my_phone_number = number
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body = "Movement was detected, Check Your Stream",
        from_= twilio_number,
        to = my_phone_number
    )
    print(message.body)

def alert2(number):
    account_sid = config.account_sid
    auth_token = config.auth_token
    twilio_number = config.twilio_number
    my_phone_number = number
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body = "Movement was detected, Check Your Stream",
        from_= twilio_number,
        to = my_phone_number
    )
    print(message.body)

def alert3(number):
    account_sid = config.account_sid
    auth_token = config.auth_token
    twilio_number = config.twilio_number
    my_phone_number = number
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body = "Movement was detected, Check Your Stream",
        from_= twilio_number,
        to = my_phone_number
    )
    print(message.body)



    #home page
@app.route('/', methods=["GET", 'POST'])
def home():
    if "loggedin" in session:
        return render_template("loggedin.html", name = session["username"])
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "loggedin" in session:
        cursor = mysql.connection.cursor()
        form = UploadForm()
        result = ""
        if form.validate_on_submit():
            print("working")
            filename = photos.save(form.photo.data)

            file_url = url_for('get_file', filename=filename)
            result = "Files uploaded"

        else:
            file_url = None
            print("not working")
        return render_template("view.html", username=session["username"], form=form, file_url=file_url, result=result)

    else:
        return render_template("index.html")
    return render_template("index.html")


@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory("uploads", filename)



#register page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", False)
        password = request.form.get("password", False)
        number = request.form.get("number", False)
        email = request.form.get("email", False)
        cursor = mysql.connection.cursor()
        cursor.execute(''' INSERT INTO userlist(user_name, gmail, password, phone) VALUES(%s, %s, %s, %s) ''',
                       (name, email, password, number))
        mysql.connection.commit()
        return redirect(url_for('home'))
    return render_template("signup.html")


#login page
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", False)
        password = request.form.get("password", False)
        cursor = mysql.connection.cursor()
        cursor.execute(
            "SELECT * FROM userlist WHERE gmail=%s AND password=%s", (email, password,))
        record = cursor.fetchone()
        try:
            rec_list = list(record.keys())
            val_list = list(record.values())
            ind = rec_list.index("user_name")
            user = val_list[ind]
            print(user)
            rec_list1 = list(record.keys())
            val_list1 = list(record.values())
            ind1 = rec_list.index("phone")
            number = val_list[ind1]
        except:
            print("retry")
        if record:
            print(record)
            session["loggedin"] = True
            session["username"] = user
            session['phone'] = number
            phone_index.append(number)
            print(phone_index[0])
            return redirect(url_for("home"))
        return redirect(url_for("home"))
    return render_template("login.html")

@app.route("/view")
def view():
    if "loggedin" in session:
        return render_template("view.html")
    return redirect(url_for("home"))

@app.route("/weapon_feed")
def weapon_feed():
    return Response(weapon_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/user_feed")
def user_feed():
    return Response(norm_capture(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed')
def motion_feed():
    return Response(motion_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/face_feed")
def face_feed():
    return Response(face_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/logout")
def logout():
    session.pop("username")
    session.pop("loggedin")
    session.pop("phone")
    phone_index = []
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run()