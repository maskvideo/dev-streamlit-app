import os
from PIL import Image
from retinaface import RetinaFace
import cv2
import numpy as np
import PIL


MASKED_FRAMES_DIR = "masked_frames_dir"


def all_faces_locations(unmasked_img):
    faces_locations = []
    resp = RetinaFace.detect_faces(np.array(unmasked_img))
    try:
        count_faces = len(resp.keys())
        for key in resp.keys():
            identity = resp[key]
            facial_area = identity["facial_area"]
            x1, y1, x2, y2 = facial_area
            faces_locations.append((x1, y1, x2, y2))
    except AttributeError:
        x1, y1, x2, y2 = (0, 0, 0, 0)
        faces_locations.append(None)

    return faces_locations


def update_parameters(unmasked_frame, kernel_size, epsilon, faces_locations):
    img = np.array(unmasked_frame)
    img = img[:, :, ::-1].copy()

    for face_loc in faces_locations:
        if not face_loc:
            return unmasked_frame

        x1, y1, x2, y2 = face_loc
        x1 -= epsilon
        y1 -= epsilon
        x2 += epsilon
        y2 += epsilon
        face_img = img[y1:y2, x1:x2]
        if kernel_size[0] % 2 == 0 or kernel_size[0] < 1:
            kernel_dim = max(3, kernel_size[0] + 1)
            kernel_size = (kernel_dim, kernel_dim)
        blurred_face = cv2.GaussianBlur(face_img, kernel_size, epsilon)
        img[y1:y2, x1:x2] = blurred_face
    return Image.fromarray(img)



