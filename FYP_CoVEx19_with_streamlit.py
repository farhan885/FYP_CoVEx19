# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 23:06:59 2021

@author: Farhan
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow import keras
import cv2

st.title("CoVEx-19")

st.title('An Artificial Intelligence based Covid-19 Expert System')
st.subheader("Stay Home, Stay Safe!")
st.sidebar.title('CoVEx-19')

select = st.sidebar.selectbox('Treatment type',['Symptoms based Treatment', 'X-Rays based Treatment'],key='1')

# Load Models
with open('models/symptoms_model.pkl', 'rb') as f:
    clf2 = pickle.load(f)
covid_xray_model = keras.models.load_model('models/covid_xray_model.h5')


if select == 'Symptoms based Treatment':
    st.write("# Symptoms Based Treatment")
    with st.form(key='my_form'):
        sex = st.selectbox('Gender', ['Male', 'Female'])
        if sex=='Male':
            sex = 1
        else:
            sex = 0
            
        age = st.text_input("Enter Age: ")
        age = age
        
        st.subheader("Select Symptoms")
        smoke = st.checkbox("Smoking")
        if smoke:
            smoke = 1
        else:
            smoke = 0
            
        fever = st.checkbox("Fever")
        if fever:
            fever = 1
        else:
            fever = 0
            
        cough = st.checkbox("Dry Cough")
        if cough:
            cough = 1
        else:
            cough = 0
            
        pneumonia = st.checkbox("Pneumonia")
        if pneumonia:
            pneumonia = 1
        else:
            pneumonia = 0
            
        diabetes = st.checkbox("Diabetes")
        if diabetes:
            diabetes = 1
        else:
            diabetes = 0
            
        asthma = st.checkbox("Asthma")
        if asthma:
            asthma = 1
        else:
            asthma = 0
            
        diarrhea = st.checkbox("Diarrhea")
        if diarrhea:
            diarrhea = 1
        else:
            diarrhea = 0
            
        sthorat = st.checkbox("Sore Thorat")
        if sthorat:
            sthorat = 1
        else:
            sthorat = 0
            
        headache = st.checkbox("Headache")
        if headache:
            headache = 1
        else:
            headache = 0
            
        mpain = st.checkbox("Muscle Pain")
        if mpain:
            mpain = 1
        else:
            mpain = 0
            
        sbreath = st.checkbox("Shortness of Breath")
        if sbreath:
            sbreath = 1
        else:
            sbreath = 0
            
        rnose = st.checkbox("Runy Nose")
        if rnose:
            rnose = 1
        else:
            rnose = 0
            
        smell_taste = st.checkbox("Abnormalities in Smell and Taste")
        if smell_taste:
            smell_taste = 1
        else:
            smell_taste = 0
            
        x_test_data = {'Sex': sex, 'Age':age, 'smoking':smoke, 'Fever':fever, 'Cough':cough, 'Pneumonia':pneumonia, 'DIABETES':diabetes, 'Asthma':asthma,
       'Diarrhea':diarrhea, 'Sore_thorat':sthorat, 'Headache':headache,
       'Muscle_pain':mpain, 'Shortness_of_breath':sbreath, 'Runy Nose':rnose,
       'Abnormalities_in_smell_and_taste':smell_taste}
        x_test_data = pd.DataFrame(x_test_data, index=[0])
        
            
        submit_button = st.form_submit_button(label='Submit parameters')
        if submit_button:
            st.write("Selected Symptoms: ",x_test_data)
            st.write("Results")
            pred_value = clf2.predict(x_test_data)
            if pred_value == 0:
                st.write("Covid Not Detected")
            else:
                st.write("Covid Detected")
            
        
        
else:
    st.write("# Chest X-Ray based Treatment")
    st.write("### Upload your Chest X-Ray here for treatment")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.write(" [  ' 0 : Covid '  ,   ' 1 : Normal '  ] ")
        st.write("Classifying...")
        #image = Image.open(uploaded_file)
        #image = cv2.imread(np.array(uploaded_file)) # read file 
        image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) # arrange format as per keras
        image = cv2.resize(image,(150,150))
        image = np.array(image) / 255
        image = np.expand_dims(image, axis=0)
        
        model_pred = covid_xray_model.predict(image)
        st.write("## Results")
        st.write('Predictions: ')
        st.write(model_pred)
        probability = model_pred[0]
        print("Model Predictions:")
        if probability[0] > 0.5:
            chest_pred = str('%.2f' % (probability[0]*100) + '% NonCOVID') 
            st.write(chest_pred)
        else:
            chest_pred = str('%.2f' % ((1-probability[0])*100) + '% COVID')
            st.write(chest_pred)
        
        st.image(image, caption='Uploaded Image.', use_column_width=True)
            