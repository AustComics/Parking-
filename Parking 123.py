import cv2

# Load the video for car parking surveillance
video = cv2.VideoCapture('C:/Users/Zablon/Downloads/2406465-uhd_3840_2160_24fps.mp4')

# Define the background subtraction method
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = video.read()
    
    if not ret:
        break
    
    # Apply background subtraction
    fgmask = fgbg.apply(frame)
    
    # Find contours of the detected objects
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        # Set minimum area threshold for contour detection
        if cv2.contourArea(contour) > 500:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Parking Lot Occupancy Detection', frame)
    
    if cv2.waitKey(30) & 0xFF == 27:  # Press Esc key to exit
        break

# Release video capture and close windows
video.release()
cv2.destroyAllWindows()
