from ultralytics import YOLO
import cv2
import numpy as np
import cvzone

# Load the model
model = YOLO("best.pt")
class_names = model.names
cap = cv2.VideoCapture('pothole.mp4')

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create a VideoWriter object
out = cv2.VideoWriter('pothole_detection_output.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

count = 0
while True:
    ret, img = cap.read()
    if not ret:
        break
    
    count += 1
    if count % 3 != 0:
        continue
    
    # Resize the image
    img = cv2.resize(img, (width, height))  # Resize to original dimensions
    h, w, _ = img.shape
    
    # Run model predictions
    results = model.predict(img)

    for r in results:
        boxes = r.boxes
        masks = r.masks
        
        if masks is not None:
            masks = masks.data.cpu()
            for seg, box in zip(masks.data.cpu().numpy(), boxes):
                seg = cv2.resize(seg, (w, h))
                contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    d = int(box.cls)
                    c = class_names[d]
                    x, y, x1, y1 = cv2.boundingRect(contour)
                    cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=2)
                    cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Write the processed frame to the output video
    out.write(img)
    
    cv2.imshow('img', img)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()  # Save the video
cv2.destroyAllWindows()
