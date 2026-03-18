import cv2
import numpy as np
import sys
import os

def main():
    weights_path = "yolov3-tiny.weights"
    cfg_path = "yolov3-tiny.cfg"
    names_path = "coco.names"

    if not all(os.path.exists(p) for p in [weights_path, cfg_path, names_path]):
        print("Model files missing. Please run setup.sh first.")
        sys.exit(1)

    print("Loading YOLOv3-tiny model...")
    net = cv2.dnn.readNet(weights_path, cfg_path)
    
    classes = []
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
        
    layer_names = net.getLayerNames()
    # Handle OpenCV version differences for getUnconnectedOutLayers
    try:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except TypeError:
        # Some versions return a flat array
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Initialize video capture (0 for default webcam, or provide a video file path)
    # E.g., cap = cv2.VideoCapture("sample_drone_video.mp4")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        sys.exit(1)

    print("Starting video stream. Press 'q' to quit.")
    
    # Generate random colors for each class
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            
        height, width, channels = frame.shape
        
        # Prepare the frame for detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        # Process the outputs
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Confidence threshold
                if confidence > 0.3:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Non-maximum suppression to remove duplicate overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence_label = f"{confidences[i]:.2f}"
                color = colors[class_ids[i]]
                
                # Draw the bounding box and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence_label}", (x, max(15, y - 10)), font, 0.5, color, 2)
        
        cv2.imshow("Object Detector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()