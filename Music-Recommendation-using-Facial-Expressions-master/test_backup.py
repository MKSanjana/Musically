from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pygame
import tkinter as tkr
from tkinter.filedialog import askdirectory
import os
import random

music_player = tkr.Tk() 
music_player.title("MUSICPAL") 
music_player.geometry('450x350')

dirlist = ''
path_list = ['D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\music\Angry',
             'D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\music\Disgust_or_fear',
             'D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\music\Happy',
             #'D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\music\m_Neutral',
             'D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\music\Sad',
             'D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\music\Surprise',
                ]

pygame.init()
pygame.mixer.init()

def main0():
    face_classifier = cv2.CascadeClassifier(r"D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\haarcascade_frontalface_default.xml")
    model = load_model(r"D:\College Projects\SEM6_proj\Music-Recommendation-using-Facial-Expressions-master\Emotion_little_vgg.h5")

    class_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    c = 0
    while c == 0:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
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
    
    if temp == 0:
        print("You are Angry")
        dirlist = path_list[0]
    elif temp == 1 or temp == 2:
        print("Calm down")
        dirlist = path_list[1]
    elif temp == 3:
        print("Stay Happy")
        dirlist = path_list[2]
    elif temp == 4:
        print("Hi Human!")
        dirlist = random.choice(path_list)
    elif temp == 5:
        print("Don't be sad, I'm here for you !")
        dirlist = path_list[3]
    elif temp == 6:
        print("Look surprised !!!")
        dirlist = path_list[4]        

    #music_player.title("Music Player") 
    #music_player.geometry('450x350')    

    # directory = askdirectory()
    os.chdir(dirlist)
    song_list = os.listdir() 

    play_list = tkr.Listbox(music_player, font='Helvetica 12 bold', bg='yellow', selectmode=tkr.SINGLE)
    for item in song_list:
        pos = 0
        play_list.insert(pos, item)
        pos += 1

    var = tkr.StringVar() 
    song_title = tkr.Label(music_player, font='Helvetica 12 bold', textvariable=var)

    def play():
        pygame.mixer.music.load(play_list.get(tkr.ACTIVE))
        var.set(play_list.get(tkr.ACTIVE))
        pygame.mixer.music.play()
    def stop():
        pygame.mixer.music.stop()
    def pause():
        pygame.mixer.music.pause()
    def unpause():
        pygame.mixer.music.unpause()

    Button1 = tkr.Button(music_player, width=4, height=2, font='Helvetica 12 bold', text='PLAY', command=play, bg='blue', fg='white')
    Button2 = tkr.Button(music_player, width=4, height=2, font='Helvetica 12 bold', text='STOP', command=stop, bg='red', fg='white')
    Button3 = tkr.Button(music_player, width=4, height=2, font='Helvetica 12 bold', text='PAUSE', command=pause, bg='purple', fg='white')
    Button4 = tkr.Button(music_player, width=4, height=2, font='Helvetica 12 bold', text='UNPAUSE', command=unpause, bg='orange', fg='white')

    song_title.pack()

    Button1.pack(fill='x')
    Button2.pack(fill='x')
    Button3.pack(fill='x')
    Button4.pack(fill='x')
    play_list.pack(fill='both', expand='yes')

Button0 = tkr.Button(music_player, width=4, height=2, font='Helvetica 12 bold', text='Start', command=main0, bg='blue', fg='white')
Button0.pack(fill='x')    

music_player['bg'] = 'black'
music_player.mainloop()