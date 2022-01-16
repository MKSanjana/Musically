from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pygame
import os
import random
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_player import st_player
import requests

st.set_page_config(page_title = 'MusicPal - Music Buddy')

dirn =r'C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\music\Happy'
temp,temp1,gray,cap = None,None,None,None
path_list = [r'C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\music\Angry',
             r'C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\music\Disgust_or_fear',
             r'C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\music\Happy',
             #r'C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\music\m_Neutral',
             r'C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\music\Sad',
             r'C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\music\Surprise',
                   ]

def main0():
    global temp,temp1,gray,cap

    face_classifier = cv2.CascadeClassifier(r"C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\haarcascade_frontalface_default.xml")
    model = load_model(r"C:\Users\Sanjana\Desktop\MusicPal-master\MusicPal-master\Music-Recommendation-using-Facial-Expressions-master\Emotion_little_vgg.h5")

    class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    FRAME_WINDOW = st.image([])

    c = 0
    while c == 0:

        # Grab a single frame of video
        print(cap.read())
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        FRAME_WINDOW.image(frame)
        faces = face_classifier.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)


            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)

            # make a prediction on the ROI, then lookup the class

                preds = model.predict(roi)[0]
                #preds = preds[:5]
                #print(preds)
                print(preds.argmax())
                label=class_labels[preds.argmax()]
                label_position = (x,y)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
            else:
                cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        cv2.imshow('Emotion Detector',frame)
        c = 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    temp = preds.argmax()
    temp1 = class_labels[temp]
    return temp

# with st.form("my_form"):
#   st.write("Inside the form")
#   slider_val = st.slider("Form slider")

#   # Every form must have a submit button.
#   submitted = st.form_submit_button("Submit")
#   if submitted:
#     st.write("slider", slider_val, "checkbox", checkbox_val)

st.title("MusicPal")
st.header("Your own Music Buddy")   
st.write('MusicPal is a Mood Based Music Recommendation System ! \n To get started Click on the button below. We start by capturing an expression of yours to detect your mood. So please make sure to have your webcam enabled for the process.')  

col1,col2,col3 = st.beta_columns(3)
with col2:
    if st.button('Start - Capture Expression'):
        result = main0()
        st.write('Result: %s' % temp1)
 
# Showing frame captured from Camera
    

       

if temp == 0:
    print("You are Angry")
    dirn = path_list[0]
    f = open("test_cap_text.txt","w")
    f.write(dirn)
    f.close()    
elif temp == 1 or temp == 2:
    print("Calm down")
    dirn = path_list[1]
    f = open("test_cap_text.txt","w")
    f.write(dirn)
    f.close()    
elif temp == 3:
    print("Stay Happy")
    dirn = path_list[2] 
    f = open("test_cap_text.txt","w")
    f.write(dirn)
    f.close()    
elif temp == 4:
    print("Hi Human!")
    dirn = random.choice(path_list)
    f = open("test_cap_text.txt","w")
    f.write(dirn)
    f.close()    
elif temp == 5:
    print("Don't be sad, I'm here for you !")
    dirn = path_list[3] 
    f = open("test_cap_text.txt","w")
    f.write(dirn)
    f.close()
elif temp == 6:
    print("Look surprised !!!")
    dirn = path_list[4]
    f = open("test_cap_text.txt","w")
    f.write(dirn)
    f.close()

def file_selector():
    dirn = open("test_cap_text.txt","r").read()
    filenames = os.listdir(dirn)
    selected_filename = st.selectbox('Select a music you want to play', filenames)

    return os.path.join(dirn, selected_filename)

filename = file_selector()
st.write('You selected:', filename)    

audio_file = open(filename, 'rb')
audio_bytes = audio_file.read()
st.audio(audio_bytes, format='audio/ogg')

