from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
from res_facenet.models import model_921

# loading model（FaceNet）
model921 = model_921()


def reg_faces(root="../faces"):
    """
        import faces using facenet
    """
    # for faces
    faces = {}
    # precoess
    preprocess = [transforms.Resize(224),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
    trans = transforms.Compose(preprocess)

    
    for file in os.listdir(root):
        if file.endswith(".jpg"):
           
            file_path = os.path.join(root, file)
         
            img = trans(Image.open(file_path)).unsqueeze(0)
            # Using the FaceNet model to turn a face into a 128-dimensional vector
            embed = model921(img)
            
            faces[file.split(".")[0]] = embed.detach().numpy()[0]
    return faces


def embed_face(img=None):
    """
        Turning a face image into a vector
    """
    preprocess = [transforms.Resize(224),
                  transforms.CenterCrop(224),
                  transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
    trans = transforms.Compose(preprocess)
    img = trans(Image.fromarray(obj=img)).unsqueeze(0)
    embed = model921(img).detach().numpy()[0]
    return embed


def get_dist(face, faces):
    """
        calculate distance
    """
    result = []
    for n, f in faces.items():
        result.append((n, np.sqrt(((f - face) ** 2).sum())))
    result.sort(key=lambda ele: ele[1])
    return result


face_db = reg_faces()


def mark_face(img=None, faces=None):
    for face in faces:
        x, y, w, h = face["box"]
        confidence = face["confidence"]
        keypoints = face["keypoints"]
        if confidence > 0.9:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 200), thickness=2)
            # lefteye
            cv2.circle(img=img, center=keypoints["left_eye"], radius=2, color=(200, 0, 0), thickness=2)
            # righteye
            cv2.circle(img=img, center=keypoints["right_eye"], radius=2, color=(200, 0, 0), thickness=2)
            # nose
            cv2.circle(img=img, center=keypoints["nose"], radius=2, color=(200, 0, 0), thickness=2)
            # leftlips
            cv2.circle(img=img, center=keypoints["mouth_left"], radius=2, color=(200, 0, 0), thickness=2)
            # rightlips
            cv2.circle(img=img, center=keypoints["mouth_right"], radius=2, color=(200, 0, 0), thickness=2)


def rec_face(img, faces):
    
    img1 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2RGB)

    for face in faces:
        x, y, w, h = face["box"]
        confidence = face["confidence"]
        # Filter some faces by confidence level
        if confidence > 0.9:
            # Face interception
            data = img1[y:y + h, x:x + w, :]
            # Embedded vectors
            vec = embed_face(img=data)
            # distance
            result = get_dist(vec, face_db)
            # shortest distance
            name, distance = result[0]
            print(distance)
            # if beyond the distance threshold, considered as a stranger
            if distance > 1.5:
                name = "Stranger"
            # add the name to the image
            cv2.putText(img=img, text=name, org=(x, y + h + 30), color=(0, 200, 0), thickness=2,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1)
