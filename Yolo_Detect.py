from ultralytics import YOLO
import cv2
import numpy as np
import time
from datetime import datetime
import os
import socket
import struct

def detect_UAV(model, img):
    # Run inference
    results = model(img, conf=0.4, device=0, iou = 0.3, augment=True)
    # Process results list
    detect = 0
    for result in results:
        boxes = result.boxes  # Probs object for classification outputs
        if len(boxes.cls) > 0:
            for box in boxes.xyxy:
                detect += 1
                x = int(box[0].item())
                y = int(box[1].item())
                x1 = int(box[2].item())
                y1 = int(box[3].item())
                cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)
                cv2.putText(img, f"UAV: {boxes.conf[0].item():.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    return img, detect

def main_():
    # Load the exported ONNX model
    onnx_model = YOLO("yolo11.yaml").load("runs//detect//train//weights//best.pt")
    onnx_model(np.ones((640,640,3), dtype=np.uint8), device=0)
    # Path to the video file
    video_path = "Images//Sources//New folder (2)//WIN_20241106_15_13_50_Pro.mp4"

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read until the video is completed
    fps = 0
    while cap.isOpened():
        # Capture frame-by-frame
        # Start the time
        start_time = time.time()
        ret, frame = cap.read()

        # If a frame was returned successfully
        if ret: 
            # Detect the UAV in the frame    
            # Get image dimensions
            height, width = frame.shape[:2]

            # Calculate the center and radius for the circular crop
            center_x, center_y = width // 2, height // 2
            radius = min(center_x, center_y, width//4)  # Radius is limited by the smallest dimension

            # Create a circular mask
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)  # White-filled circle

            # Apply the mask to get the circular cropped area
            circular_crop = cv2.bitwise_and(frame, frame, mask=mask)

            # Crop the rectangular bounding box around the circle
            x1, y1 = center_x - radius, center_y - radius
            x2, y2 = center_x + radius, center_y + radius
            frame = circular_crop[y1:y2, x1:x2]                  
            output_frame, result = detect_UAV(onnx_model, frame)
            output_frame = cv2.resize(output_frame, (640, 640))
            cv2.imshow("uav", output_frame)
            print(fps)
            # Press 'q' on the keyboard to exit the loop early
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

        # Calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Calculate FPS
        fps = 1 / elapsed_time

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

def main():
    # Load the exported ONNX model
    yolo_model = YOLO("yolo11.yaml").load("best.pt")
    yolo_model(np.ones((640,640,3), dtype=np.uint8), device=0)

    while True:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(('0.0.0.0', 9999))
        server_socket.listen(1)
        print("Server listening on port 9999...")

        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")

            # Receive image data from client
            data = b""
            try:
                len_data = int.from_bytes(client_socket.recv(4), byteorder='big')
                data = client_socket.recv(len_data)
            except:
                continue

            # Decode the image data
            np_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # Detect the UAV in the frame                      
            # Get image dimensions
            height, width = frame.shape[:2]

            # Calculate the center and radius for the circular crop
            center_x, center_y = width // 2, height // 2
            radius = min(center_x, center_y, width//4)  # Radius is limited by the smallest dimension

            # Create a circular mask
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, (center_x, center_y), radius, (255, 255, 255), -1)  # White-filled circle

            # Apply the mask to get the circular cropped area
            circular_crop = cv2.bitwise_and(frame, frame, mask=mask)

            # Crop the rectangular bounding box around the circle
            x1, y1 = center_x - radius, center_y - radius
            x2, y2 = center_x + radius, center_y + radius
            frame = circular_crop[y1:y2, x1:x2]                  
            output_frame, result = detect_UAV(yolo_model, frame)
            output_frame = cv2.resize(output_frame, (640, 640))
            # Encode the processed image
            _, encoded_image = cv2.imencode('.jpg', output_frame)
            # Send the processed image back to the client
            client_socket.sendall(struct.pack("B", result) + encoded_image.tobytes())
            print("Send all")
            client_socket.close()

if __name__ == "__main__":
    main()
