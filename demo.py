import cv2
import argparse
import numpy as np
import tkinter
import  PySimpleGUI as sg
import time
import imutils

layout = [[sg.Image(filename='', key='image')]
,[sg.Button("Object Detection Yolo", button_color=("black", "silver"), size=(12, 2))
,sg.Button("Face Detection HAAR Cascade", button_color=("black", "silver"), size=(12, 2))
,sg.Button("Face Detection Neural Net", button_color=("black", "silver"), size=(12, 2))]
]

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def Yolo(image):
    try:
        #print(image)
        #print(image.shape)  
        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    except Exception as e:
        print('Failed dnn: '+ str(e))
    
    return image

def HAAR(img):
    
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = net.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
    except Exception as e:
        print("failed HAAR" + str(e))
    return img


def FaceNN(frame):
    frame = imutils.resize(frame, width=600, height=800)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (103.93, 116.77, 123.68))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < .5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    
    return frame

classes = None
net = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')


with open("yolov3.txt", 'r') as f:
     classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
window, cap = sg.Window('AI Insider Lab', layout, location=(0, 0), grab_anywhere=True), cv2.VideoCapture(0)

main_event = "HAAR"

while True:
    ret, frame = cap.read()
    #print(ret)
    if cap is None or not cap.isOpened():
        print("sleep")
        continue

    #print("awake")
    event, values = window.read(timeout=20)
     
    if(event == "Object Detection Yolo"):
        main_event = "Yolo"
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    if(event == "Face Detection HAAR Cascade"):
        main_event = "HAAR"
        cap.set(cv2.CAP_PROP_BUFFERSIZE,10)
    if(event == "Face Detection Neural Net"):
        main_event = "FaceNN"
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        net = cv2.dnn.readNetFromCaffe("deploy.prototxt.txt", "res10_300x300_ssd_iter_140000.caffemodel")

    if main_event == "Yolo":
        image = Yolo(frame)
    if main_event == "HAAR":
        image = HAAR(frame)
    if main_event == "FaceNN":
        image = FaceNN(frame)
    
    try:
        imgbytes = cv2.imencode('.png', image)[1].tobytes()  # ditto
        window['image'].update(data=imgbytes)    

    except Exception as e:
        print("nope" + str(e))

    key=cv2.waitKey(1)
    if key == ord('q'):
      break
cv2.waitKey()
    
cv2.destroyAllWindows()
