import cv2
import dlib
import os
import keyboard
import numpy as np

while True:
    name = input("이름과 학번을 입력해주세요 : ")
    
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        
        while True:
            ret,frame = cap.read()
        
            if ret:
                cv2.imshow('camera', frame) 

                if cv2.waitKey(1) != -1:
                    cv2.imwrite('img2.jpg', frame)
                    break
            
    cap.release()
    cv2.destroyAllWindows()

    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    facerec = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

    #load images
    img1 = dlib.load_rgb_image("img1.jpg")
    img2 = dlib.load_rgb_image("img2.jpg")

    #detection
    img1_detection = detector(img1, 1)
    img2_detection = detector(img2, 1)

    img1_shape = sp(img1, img1_detection[0])
    img2_shape = sp(img2, img2_detection[0])

    #alignment
    img1_aligned = dlib.get_face_chip(img1, img1_shape)
    img2_aligned = dlib.get_face_chip(img2, img2_shape)

    img1_representation = facerec.compute_face_descriptor(img1_aligned)
    img2_representation = facerec.compute_face_descriptor(img2_aligned)

    img1_representation = np.array(img1_representation)
    img2_representation = np.array(img2_representation)

    def findEuclideanDistance(source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    distance = findEuclideanDistance(img1_representation, img2_representation)

    text1 = "{}님은 김학택 교장 선생님과 {}% 만큼 닮았습니다.".format(name, (1-distance)*100)
    
    file = open("index.txt", "a")
    file.write(text1)
    file.close()

    print("##########################################")
    print("")
    print(text1)
    print("")
    print("##########################################")
    print("")

    os.remove("img2.jpg")
    
    print("다시 시작하기 위해 스페이스 바를 누르시오.")
    keyboard.wait("space")
    os.system('cls')

