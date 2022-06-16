import cv2
from mtcnn import MTCNN
from dl_utils import mark_face
from dl_utils import rec_face

# Load model (MTCNN)
mtcnn = MTCNN()

# Turn on video/camera
cap = cv2.VideoCapture(". /love.flv")

while True:
    # Read a frame of image
    ret, img = cap.read()

    # If the read is successful
    if ret:
        # Convert the image to RGB format
        img1 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

        # Face detection (detects if there is a human face in the image)
        faces = mtcnn.detect_faces(img=img1)
        if faces:
            # Face annotation (box, landmark)
            mark_face(img=img, faces=faces)

            # Face recognition (identify)
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
