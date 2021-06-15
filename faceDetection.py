import cv2

# open cv is a open source computer vision library.

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# in we are going to detect face.
image = cv2.imread("image3.jpg")

# convert in the gray scale
grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect face.
face_coord = trained_face_data.detectMultiScale(grayscaled_image)

for coord in face_coord:
    x, y, width, height = coord
    cv2.rectangle(image, (x, y), (x+width, y+height), (0, 255, 0), 2)

# showing image on the screen.
cv2.imshow("Image", image)
cv2.waitKey()

# For Video

# deleted the video so it will not work. for video.

#  Loading Video.
video = cv2.VideoCapture("video.mp4")

while True:
    read_or_not , frame = video.read()

    # converting into gray scale.
    grayscaled_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coord = trained_face_data.detectMultiScale(grayscaled_image)
    for coord in face_coord:
        x, y, width, height = coord
        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 255, 0), 1)

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    if key == 113 or key==81:
        break

video.release()
