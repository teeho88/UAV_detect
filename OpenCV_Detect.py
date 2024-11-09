import cv2
import numpy as np
import time
from datetime import datetime
import os
import socket
import struct

def detect_UAV(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper range for the color red in HSV
    lower_red = np.array([0, 60, 50])
    upper_red = np.array([10, 255, 255])

    # Create a mask for red color
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # To capture the red color in another range (for better detection)
    lower_red = np.array([170, 60, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    # Combine both masks
    mask = mask1 | mask2

    # Apply a series of dilations and erosions to remove small blobs
    kernel_1 = np.ones((5, 5), np.uint8)  # You can adjust the kernel size (5, 5) for different effects
    mask = cv2.erode(mask, kernel_1, iterations=1)
    kernel_2 = np.ones((30, 60), np.uint8)  # You can adjust the kernel size (5, 5) for different effects
    mask = cv2.dilate(mask, kernel_2, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Loop over the contours
    cnt_obj = 0
    cap_err = 0
    clone_frame = frame.copy()
    result = 0
    for contour in contours:
        # Compute the area of the contour
        x, y, w, h = cv2.boundingRect(contour)
        area = w*h
        # Set a minimum area threshold to filter out small objects
        if 25000 > area > 3000:  # Adjust this threshold based on your use case
            # Calculate aspect ratio (width/height) to ensure it matches the UAV shape
            aspect_ratio = float(w) / h
            if 1.3 < aspect_ratio < 3.7:  # Adjust based on your UAV shape
                # # Save image if detection had problems
                # cnt_obj += 1
                # if cnt_obj > 1 and cap_err == 0:
                #     cap_err = 1
                #     f_name = 'err_' + datetime.now().strftime("%Y_%m_%d %H_%M_%S")
                #     path = os.path.join('Images', 'error',f'{f_name}.jpg')
                #     cv2.imwrite(path, clone_frame)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "UAV detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                result = 1
    return frame, result

def main_test():
    # Path to the video file
    video_path = "WIN_20240822_15_57_25_Pro.mp4"

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
            
            output_frame = detect_UAV(frame)
            # Display the result
            output_frame = cv2.resize(output_frame, (640, 640))
            cv2.putText(output_frame, f"fps: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("UAV Detection", output_frame)
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
            len_data = int.from_bytes(client_socket.recv(4), byteorder='big')
            data = client_socket.recv(len_data)

            # Decode the image data
            np_array = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

            # frame = cv2.imread('Images\mau 1\Image4.jpg')

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
            
            output_frame, result = detect_UAV(frame)
            # Display the result
            output_frame = cv2.resize(output_frame, (640, 640))
            # Encode the processed image
            _, encoded_image = cv2.imencode('.jpg', output_frame)
            # Send the processed image back to the client
            client_socket.sendall(struct.pack("B", result) + encoded_image.tobytes())
            print("Send all")
            client_socket.close()

if __name__ == "__main__":
    main()
