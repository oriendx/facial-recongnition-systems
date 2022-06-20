import cv2
import numpy as np

from ml_utils import mark_face
from ml_utils import rec_face

# Load the model (harr cascade detector...)
harr = cv2.CascadeClassifier("hararcascade_frontalface_default.xml")

# Turn on video/camera
cap = cv2.VideoCapture(". /love.flv")

while True:
    # Read a frame of image
    ret, img = cap.read()

    # If the read is successful
    if ret:
        img1 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        # Face detection (detects if there is a human face in the image)
        faces = harr.detectMultiScale(image=img1, scaleFactor=1.3, minNeighbors=5)

        if isinstance(faces, np.ndarray):
            # face marker (box, landmark)
            mark_face(img=img, faces=faces)
            # Face recognition (identification)
            rec_face(img=img, faces=faces)

        # Display image
        cv2.imshow(winname="love", mat=img)
        cv2.waitKey(delay=1)

    else:
        # Failed to read, exit loop
        break

# Release the resource
cap.release()
cv2.destroyAllWindows()
