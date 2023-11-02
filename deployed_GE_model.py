import numpy as np
import pickle
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import IsolationForest

# Load the pre-trained anomaly detection model
model = keras.models.load_model('Generator_Engine-LSTM_AE_model.h5')


# Define the Streamlit app
st.title('Anomaly Detection App')


def main():
   # Giving The Title
   st.header("Anomaly Detection in Generator Engine (GE)")
   st.sidebar.title("Anomaly Detection")
   st.sidebar.markdown("Let's start Generator Engine")
   st.sidebar.markdown("LSTM AE model has been Loaded...")
   

   #@st.cache_data(persist=True)
   def load():
       # Create a file uploader widget
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:

            # Use Pandas to read the uploaded CSV file
            df = pd.read_csv(uploaded_file)

            # Display the contents of the CSV file as a data table
            st.write('**CSV File Contents:**')
            st.dataframe(df)

            # Set the title and description of your Streamlit app
            st.title('Time Series Data Visualization')
            st.write('Visualizing a sample time series.')

            # Create a line chart
            #st.line_chart(df.set_index('datetime'))
            st.line_chart(df.set_index('time'))

          # Create a scatter plot using Matplotlib
            fig, ax = plt.subplots()
            ax.scatter(df.iloc[1:10,:]['time'], df.iloc[1:10,:]['RPM'])
            st.pyplot(fig)

            # Set the title and description of your Streamlit app
            st.title('Scatter Plot of Two Variables')
            st.write('Visualizing one random variables.')

            # Create a Plotly figure
            fig = go.Figure(df=go.Scatter(x=df['X'], y=df['Y'], mode='markers'))

            # Display the figure using st.plotly_chart
            st.plotly_chart(fig)
            # Perfrom Anomaly Detection
        
            
   if st.sidebar.checkbox("Display Training Accuracy of The Model", False):
         st.subheader("Training Accuracy of The Model")
         image = Image.open('Figure_Acc_LSTM_AE.jpeg')
         st.image(image, caption='Training Accuracy')

   if st.sidebar.checkbox("Display Training Loss of The Model", False):
         st.subheader("Training Loss of The Model")
         image = Image.open('Figure_Loss_LSTM_AE.jpeg')
         st.image(image, caption='Training Loss')
    
   if st.sidebar.checkbox("Load New Data for Anomaly Detection", False):
        st.subheader("Load The GE New Test Data")
        df = load()
   
   # Perfrom Anomaly Detection
   #ExhGasInletTempA , ExhGasOutletTemp, LOInletPress, LOOutletTemp, RPM	
   

   # Code for Anomaly Predicition
   Anomaly_Prediction_Result = ''

   # Creating a buttun for Anomaly Prediction on User Input Data



if __name__ == '__main__':
   main()