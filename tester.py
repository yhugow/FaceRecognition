import cv2
import os
import numpy as np
import faceRecognition as fr
import urllib

from flask import Flask
from flask_restful import Api, Resource, reqparse

from flask_cors import CORS

app = Flask(__name__)   
api = Api(app)
CORS(app)

users = [
    {
        "name": "Nicholas",
        "age": 42,
        "occupation": "Network Engineer"
    },
    {
        "name": "Elvin",
        "age": 32,
        "occupation": "Doctor"
    },
    {
        "name": "Jass",
        "age": 22,
        "occupation": "Web Developer"
    }
]

class User(Resource):
    def get(self):

        #This module takes images  stored in diskand performs face recognition
        test_img=cv2.imread('/Users/itsupport/Documents/IT/Herbal App/app/fr/FaceRecognition/TestImages/img2.jpg')#test_img path
        faces_detected,gray_img=fr.faceDetection(test_img)
        print("faces_detected:",faces_detected) 


        #Uncomment belows lines when running this program first time.Since it svaes training.yml file in directory
        faces,faceID=fr.labels_for_training_data('/Users/itsupport/Documents/IT/Herbal App/app/fr/FaceRecognition/trainingImages')
        face_recognizer=fr.train_classifier(faces,faceID)
        # face_recognizer.save('trainingData.yml')
        # face_recognizer=cv2.face.LBPHFaceRecognizer_create()
        # face_recognizer.read('trainingData.yml')#use this to load training data for subsequent runs

        name={
            0:"Old Man",
            1:"Girl",
            2:"Leonardo"
        } #creating dictionary containing names for each label

        resname =''

        for face in faces_detected:
            (x,y,w,h)=face
            roi_gray=gray_img[y:y+h,x:x+h]
            label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
            print(label,confidence)
            resname=label
            fr.draw_rect(test_img,face)
            predicted_name=name[label]
            if(confidence>37):#If confidence less than 37 then don't print predicted face text on screen
                continue
            # fr.put_text(test_img,predicted_name,x,y)

        # resized_img=cv2.resize(test_img,(500,250))
        # cv2.imshow("face dtecetion tutorial",resized_img)
        # cv2.waitKey(0)#Waits indefinitely until a key is pressed
        # cv2.destroyAllWindows
        print(predicted_name)
        print(label)
        print(confidence)
        return predicted_name, 200
      
api.add_resource(User, "/user")

app.run(debug=True)




