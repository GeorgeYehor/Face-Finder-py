import cv2

# Load the image
image = cv2.imread('image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Use Haar classifier to detect faces
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 1.4, 5)

# Iterate over all detected faces
for (x,y,w,h) in faces:
   # Crop the region of image with coordinates (x, y, w, h)
   face_image = image[y:y + h, x:x + w]

   # Save the new image
resized_face_image = cv2.resize(face_image, (640, 480))

# Display the image using cv2.imshow
cv2.imshow('Face', resized_face_image)
cv2.waitKey(0)

