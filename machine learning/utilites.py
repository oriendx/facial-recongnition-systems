import numpy as np
import os
import cv2


def reg_faces(root="../faces"):
    """
        使用FaceNet将录入人脸
    """

    names = {}
    images = []
    labels = []
    for idx, file_name in enumerate(os.listdir(root)):
        names[idx] = file_name.split(".")[0]
        images.append(cv2.resize(src=cv2.imread(filename=os.path.join(root, file_name), flags=cv2.IMREAD_GRAYSCALE),
                                 dsize=(214, 269)))
        labels.append(idx)

    # [0, - ] < 50   > 80
    # recognizer = cv2.face.LBPHFaceRecognizer_create()

    # [0, 20000], 5000
    recognizer = cv2.face.EigenFaceRecognizer_create()

    # [0, 20000], 5000
    # recognizer = cv2.face.FisherFaceRecognizer_create()

    recognizer.train(images, np.array(labels))
    return recognizer, names, labels


# 获取人脸库
recognizer, names, labels = reg_faces()


def mark_face(img=None, faces=None):
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 200), thickness=2)


def rec_face(img, faces):
    """
        人脸识别
    """
    # 将图像转为 gray 模式
    img1 = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)

    for face in faces:
        x, y, w, h = face

        # 截取人脸
        data = img1[y:y + h, x:x + w]
        label, confidence = recognizer.predict(cv2.resize(src=data, dsize=(214, 269)))
        print(label, names[label], confidence)
        # 超过距离的阈值，则认为是陌生人
        if confidence > 8000:
            name = "Stranger"
        else:
            name = names[label]
        # 将名字打印到图像中
        cv2.putText(img=img, text=name, org=(x, y + h + 30), color=(0, 200, 0), thickness=2,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1)
