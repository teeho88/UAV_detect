import cv2
import numpy as np

def detect_UAV(frame):
    # Convert frame to grayscale (template matching works on grayscale images)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # _, gray_frame = cv2.threshold(gray_frame, 85, 255, cv2.THRESH_BINARY
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray_frame, (51, 51), 0)
    # Apply edge detection (Canny or similar method)
    edges = cv2.Canny(blurred, 5, 7)
    # Define the kernel (structuring element)
    kernel = np.ones((30, 30), np.uint8)  # You can adjust the kernel size (5, 5) for different effects
    # Apply the dilation operation
    edges = cv2.dilate(edges, kernel, iterations=1)  # 'iterations' determines how much the dilation is applied
    output_frame = frame.copy()
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w*h
        if 2000 < area < 15000:  # Adjust these thresholds based on your UAV size
            # Calculate aspect ratio (width/height) to ensure it matches the UAV shape
            aspect_ratio = float(w) / h
            if 1.3 < aspect_ratio < 4:  # Adjust based on your UAV shape
                cv2.rectangle(output_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(output_frame, "UAV Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return output_frame


def main():
    # Path to the video file
    video_path = "WIN_20240822_15_10_45_Pro.mp4"

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Read until the video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If a frame was returned successfully
        if ret:
            # Detect the UAV in the frame
            output_frame = detect_UAV(frame)
            # Display the result
            output_frame = cv2.resize(output_frame, (960, 640))
            cv2.imshow("UAV Detection", output_frame)
            # Press 'q' on the keyboard to exit the loop early
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # Release the video capture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()