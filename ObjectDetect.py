import cv2, time

#Create an object. Zero for external camera
video = cv2.VideoCapture(-1)
print(video)

#Create a frame object
check, frame = video.read()

print(check)
print(frame)

#Show the frame
cv2.imshow("Capturing", frame)

# PRess any key to out (msec)
cv2.waitKey(0)

# Shutdown the camera
video.release()