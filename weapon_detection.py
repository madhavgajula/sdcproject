import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="Weapon Detection", layout="wide")

st.title("🔫 Weapon Detection System")

# Load YOLO model (you must provide these files)
@st.cache_resource
def load_model():
    net = cv2.dnn.readNet("model/yolov3.weights", "model/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

try:
    net, output_layers = load_model()
    model_loaded = True
except:
    st.warning("⚠️ YOLO model not found. Please add model files.")
    model_loaded = False

classes = ["Weapon"]

# Upload Image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Weapon") and model_loaded:
        height, width, _ = img.shape

        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416),
                                     (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes = []
        confidences = []
        class_ids = []

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

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected = False

        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 0, 255)

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                detected = True

        if detected:
            st.success("✅ Weapon Detected")
        else:
            st.info("❌ No Weapon Detected")

        st.image(img, caption="Detection Result", use_container_width=True)

# Sidebar info
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Upload an image  
2. Click 'Detect Weapon'  
3. View results  
""")
