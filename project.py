import streamlit as st
from PIL import Image
import cv2
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.models import load_model

from gtts import gTTS



def getImage(x):
    
    test_img_path = x

    test_img = cv2.imread(test_img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    test_img = cv2.resize(test_img, (224,224))

    test_img = np.reshape(test_img, (1,224,224,3))
    
    return test_img


def load_image(image_file):
    img = Image.open(image_file)
    return img




from pathlib import Path

HERE = Path(__file__).parent

new_dict = pickle.load(open(HERE / "new_dict.pickle","rb"))
inv_dict = pickle.load(open(HERE / "inv_dict.pickle","rb"))

# from keras.applications.resnet import ResNet50

# incept_model = ResNet50(include_top=True)
# from keras.models import Model
# last = incept_model.layers[-2].output
# modele = Model(inputs = incept_model.input,outputs = last)

def get_caption(file_path):
    modele = load_model(r"C:\Users\MYPC\Desktop\computer vision\cookie-monster\feature_model.h5")
    model = load_model(r"C:\Users\MYPC\Desktop\computer vision\cookie-monster\caption_model.h5")

    MAX_LEN=36
    
    # img = cv2.imread(file_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (224,224))
    
    # img = img.reshape(1,224,224,3)
    

    test_feature = modele.predict(getImage(file_path)).reshape(1,2048)
    
    test_img_path = file_path
    test_img = cv2.imread(test_img_path)
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)


    text_inp = ['startofseq']

    count = 0
    caption = ''
    while count < 25:
        count += 1

        encoded = []
        for i in text_inp:
            encoded.append(new_dict[i])

        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)


        prediction = np.argmax(model.predict([test_feature, encoded]))

        sampled_word = inv_dict[prediction]

        caption = caption + ' ' + sampled_word
            
        if sampled_word == 'endofseq':
            break

        text_inp.append(sampled_word)
    return caption

st.title("Image Caption Generator")
st.subheader("Upload Image to generate caption")
image_file = st.file_uploader("Upload Images",type=["png","jpeg","jpg"])
if image_file is not None:
    st.write(type(image_file))
    file_details = {"filename":image_file.name,"filetype":image_file.type,"filesize":   image_file.size}
    st.write(file_details)
    st.image(load_image(image_file),width=400)
    file_path = os.path.abspath(image_file.name)
    processed_image = getImage(file_path)
    model = load_model("caption_model.h5")
    caption = get_caption(file_path).strip(".endofseq")

    myobj = gTTS(text=caption,lang="en",slow=False)
    myobj.save("caption.mp3")

    audio_file = open("caption.mp3","rb")
    audio_bytes = audio_file.read()

    st.subheader("Caption Generated: {}".format(caption))

    import time
    with st.spinner("Genrating audio!!!"):
        time.sleep(5)
        st.header("Click here to listen audio caption")
        st.audio(audio_bytes,format='audio/ogg',start_time=0)



