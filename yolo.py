import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image

st.write("""
# Object Detection with YOLO v3
""")

image = st.file_uploader('Image for Object Detection', type=['jpg', 'png'])

if image is not None:
  st.image(image)

  image = Image.open(image)

  st.write("### Object Detection Result with YOLO")

  net = cv2.dnn.readNet("yolov3.cfg", "yolov3.weights")
  classes = []

  with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
  layer_names = net.getLayerNames()
  output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

  image = np.uint8(image)
  image = cv2.resize(image, None, fx=1.0, fy=1.0)
  height, width, channels = image.shape

  blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

  net.setInput(blob)
  outs = net.forward(output_layers)


  class_ids = []
  confidences = []
  boxes = []
  for out in outs:
    for detection in out:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > 0.5:
        center_x = int(detection[0] * width)
        center_y = int(detection[1] * height)
        w = int(detection[2] * width)
        h = int(detection[3] * height)

        x = int(center_x - w/2)
        y = int(center_y - h/2)


        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)


  indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
  # number_of_objects_detected = len(boxes)
  font = cv2.FONT_HERSHEY_PLAIN

  results = dict()
  for i in range(len(boxes)):
    if i in indexes:
      x, y, w, h = boxes[i]
      label = classes[class_ids[i]]

      if label in results:
        results[label] += 1
      else:
        results[label] = 1

      cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
      cv2.putText(image, label, (x, y + 40), font, 2, (255, 0, 0), 2)

  st.image(image)
  
  st.write("### Object Detected")
  for x, y in results.items():
    st.write(x.capitalize(), " :", y)

