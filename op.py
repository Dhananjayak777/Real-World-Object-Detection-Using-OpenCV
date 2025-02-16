import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLO
print("[INFO] Loading YOLO model...")
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("C:\\Users\\Dhana\\Desktop\\new\\yolov3.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers().flatten()

# Retrieve the output layer names
output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

# Set a specific color for the bounding box and text (e.g., red color)
box_color = (0, 0, 255)  # Red color in BGR
text_color = (0, 0, 255)  # Red color in BGR

# Start video stream
print("[INFO] Starting video stream...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open video device")
    exit()

spoken_labels = set()  # Keep track of spoken labels

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    height, width, channels = frame.shape

    # Create a 4D blob from a frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    print("[INFO] Blob shape:", blob.shape)
    
    net.setInput(blob)
    outs = net.forward(output_layers)
    print("[INFO] Forward pass completed")

    # Initialize lists for detected bounding boxes, confidences, and class IDs
    class_ids = []
    confidences = []
    boxes = []

    # Iterate over each of the detections
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak predictions by ensuring the `confidence` is greater than the minimum confidence
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

    # Perform non-maxima suppression to eliminate redundant overlapping boxes with lower confidences
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    current_labels = set()

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            current_labels.add(label)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

    # Speak the detected labels only once
    new_labels = current_labels - spoken_labels
    if new_labels:
        detected_text = ", ".join(new_labels)
        engine.say(detected_text)
        engine.runAndWait()
        spoken_labels.update(new_labels)

    # Display the resulting frame with detections
    cv2.imshow("Image", frame)

    # Break the loop when 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the video capture object and close display window
cap.release()
cv2.destroyAllWindows()
