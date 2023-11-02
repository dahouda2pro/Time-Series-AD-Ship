import numpy as np
import pickle
import pandas as pd
from PIL import Image

import streamlit as st

# Loading the trained model
loaded_model = pickle.load(open('Data/trained_model.sav', 'rb'))

# Creating The Function Model

def anomaly_prediction(input_Test_Data):

    # Changing the input Data to numpy array
    input_Test_Data_As_Numpy_Array = np.asarray(input_Test_Data)

    # Reshape the array as we are predicting for one Instance
    input_Test_Data_Reshaped = input_Test_Data_As_Numpy_Array.reshape(1, -1)

    prediction = loaded_model.predict(input_Test_Data_Reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return "Normal Data"
    else :
       return "Anomaly Detected"

def main():
   # Giving The Title
   st.header("Anomaly Detection in Generator Engine")
   st.sidebar.title("Anomaly Detection")
   st.sidebar.markdown("Let's start Generator Engine")

   #@st.cache_data(persist=True)
   def load():
        data = pd.read_csv('Data\Anomalies_Data.csv')
        return data
   df = load()

   if st.sidebar.checkbox("Display data", False):
        st.subheader("Show The GE Data")
        st.write(df)

   if st.sidebar.checkbox("Display Training Accuracy", False):
         st.subheader("Training Accuracy")
         image = Image.open('Figure_Acc.jpeg')
         st.image(image, caption='Training Accuracy')

   if st.sidebar.checkbox("Display Training Loss", False):
         st.subheader("Training Loss")
         image = Image.open('Figure_Loss.jpeg')
         st.image(image, caption='Training Loss')
   # Input Data From The User
   #ExhGasInletTempA , ExhGasOutletTemp, LOInletPress, LOOutletTemp, RPM	
   ExhGasInletTempA = st.text_input('ExhGasInletTempA')
   ExhGasOutletTemp = st.text_input('ExhGasOutletTemp')
   LOInletPress = st.text_input('LOInletPress')
   LOOutletTemp = st.text_input('LOOutletTemp')
   RPM = st.text_input('RPM')

   # Code for Anomaly Predicition
   Anomaly_Prediction_Result = ''

   # Creating a buttun for Anomaly Prediction on User Input Data
   if st.button('Check Anomaly'):
      Anomaly_Prediction_Result = anomaly_prediction([ExhGasInletTempA, ExhGasOutletTemp, LOInletPress, LOOutletTemp, RPM])
    
   st.success(Anomaly_Prediction_Result)


if __name__ == '__main__':
   main()