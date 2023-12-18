import numpy as np
import pickle
from datetime import datetime
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# Load the pre-trained anomaly detection model
model_loaded = keras.models.load_model('CyExGas_CyPmax_Model.h5')


# Define the Streamlit app
#st.title('INS 통합 테스트 시뮬레이션 SW 계발')
st.title('Anomaly Detection in DE2 Thing HMD8310')


def main():
   # Giving The Title
   st.header("Anomaly Detection in Diesel Engine (DE)")
   # Display an image from a local file
   #st.image("ship.png", width=10,  caption="Ship", use_column_width=True)
   st.sidebar.title("Anomaly Detection in DE2")
   html_temp = """
             <div style="background-color:tomato;padding:10px>
             <h1 style="color:white;text-align:center;font--size:23px">Anomaly Detection in Diesel Engine (DE) </h1>
             </div>
               """

   st.markdown(html_temp, unsafe_allow_html=True)
   
   # Function to convert ISO 8601 time to datetime
   def ISO_8601_To_Datetime(s):
        return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ')
   
   def preprocess_steps():
        st.write('  ')
        st.write("##### 1. Selecting Features") 
        st.write("##### 2. Checking Missing Values") 
        st.write('##### 3. Dropping Missing Values')
        st.write('##### 4. Deleting rows with 0 values in the Data')
        st.write('##### 5. Feature Scaling')

   #@st.cache_data(persist=True)
   def load():
       # Create a file uploader widget
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:

            # Use Pandas to read the uploaded CSV file
            df = pd.read_csv(uploaded_file, index_col=0, parse_dates=[1], date_parser=ISO_8601_To_Datetime)
            df_2 = df.copy()
            #st.write(df.columns)

            # Display the contents of the CSV file as a data table
            st.write('**DE2 Text Data:**')
            st.dataframe(df)
            st.write("Data Shape", df.shape)

            # Set the title and description of your Streamlit app
            st.title('Time Series Data Visualization')
            st.write('Visualizing a sample time series.')
            
            # Convert Date column to datetime
            #=df['time'] = pd.to_datetime(df['time'])

            # Set Date as the index (optional, for better plotting)
            #df.set_index('time', inplace=True)

            # Create a Matplotlib figure
            plt.figure(figsize=(18, 8))

            # Plot columns from DataFrame
            plt.plot(df.index, df['Cy1ExhGasOutletTemp'], label='Cy1 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy2ExhGasOutletTemp'], label='Cy2 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy3ExhGasOutletTemp'], label='Cy3 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy4ExhGasOutletTemp'], label='Cy4 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy5ExhGasOutletTemp'], label='Cy5 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy6ExhGasOutletTemp'], label='Cy6 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy7ExhGasOutletTemp'], label='Cy7 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy8ExhGasOutletTemp'], label='Cy8 ExhGasOutletTemp')
            plt.plot(df.index, df['Cy9ExhGasOutletTemp'], label='Cy9 ExhGasOutletTemp')

            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Values')
            plt.title('Values over Time of CyExhGasOutlet Temperature 1 to 9')
            plt.legend()

            # Display Matplotlib plot in Streamlit
            st.pyplot(plt)

            plt.figure(figsize=(18, 8))

            # Plot columns from DataFrame
            plt.plot(df.index, df['Cyl1_Pmax'], label=' Cyl1 Pmax')
            plt.plot(df.index, df['Cyl2_Pmax'], label=' Cyl2 Pmax')
            plt.plot(df.index, df['Cyl3_Pmax'], label=' Cyl3 Pmax')
            plt.plot(df.index, df['Cyl4_Pmax'], label=' Cyl4 Pmax')
            plt.plot(df.index, df['Cyl5_Pmax'], label=' Cyl5 Pmax')
            plt.plot(df.index, df['Cyl6_Pmax'], label=' Cyl6 Pmax')
            plt.plot(df.index, df['Cyl7_Pmax'], label=' Cyl7 Pmax')
            plt.plot(df.index, df['Cyl8_Pmax'], label=' Cyl8 Pmax')
            plt.plot(df.index, df['Cyl9_Pmax'], label=' Cyl9 Pmax')

            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel('Values')
            plt.title('Values over Time of Cyl Max Pression 1 - 9')
            plt.legend()

            # Display Matplotlib plot in Streamlit
            st.pyplot(plt)

            # Set the title and description of your Streamlit app
            st.title('Data Preprocessing and')
            st.title('Anomaly Detection in DE2 Test Data')
            st.markdown(
                  """
                  <style>
                  .stButton>button {
                        background-color: orange; /* Change this color code */
                        color: black;
                        text:black;
                        }
                  </style>
                  """,
                  unsafe_allow_html=True
                  )
            preprocess_button = st.button("Click here for Preprocessing the Test Data", key='custom_button')
            if preprocess_button:
                 st.write('#### 1. Selecting 21 Features.')
                  #  # 1. Selecting 21 Features
                  #st.write(df.shape)
                  #st.write(df_2.columns)
                 df_21 = df.loc[:, ['time','Cy1ExhGasOutletTemp',
                                                'Cy2ExhGasOutletTemp',
                                                'Cy3ExhGasOutletTemp',
                                                'Cy4ExhGasOutletTemp',
                                                'Cy5ExhGasOutletTemp',
                                                'Cy6ExhGasOutletTemp',
                                                'Cy7ExhGasOutletTemp',
                                                'Cy8ExhGasOutletTemp',
                                                'Cy9ExhGasOutletTemp',
                                                'Cyl1_Pmax', 
                                                'Cyl2_Pmax', 
                                                'Cyl3_Pmax', 
                                                'Cyl4_Pmax', 
                                                'Cyl5_Pmax', 
                                                'Cyl6_Pmax', 
                                                'Cyl7_Pmax', 
                                                'Cyl8_Pmax', 
                                                'Cyl9_Pmax',
                                                'Load',
                                                'Power' 
                                                ]]
                 st.write(df_21.shape)
                 st.dataframe(df_21)
                 #st.dataframe(df_21.columns)
                  # 2. Checking Missing Values
                 st.write('#### 2. Checking Missing Values')
                 st.write(df_21.isnull().sum())
                 st.write(df_21.shape)

                  # 3. Dropping Missing Values
                 st.write('#### 3. Dropping Missing Values')
                 df_21_Without_NaN = df_21.dropna()
                 numberRowsBefore = df_21_Without_NaN.shape[0]
                 st.write(df_21.dropna())
                 st.write(df_21_Without_NaN.shape)
                 
                 # 4. Delete rows with 0 values in a pandas DataFrame
                 st.write('#### 4. Deleting rows with 0 values in the Data')
                 # Use boolean indexing to filter rows with 0 values
                 st.write("Number of Rows before deleting Zero Values:", df_21_Without_NaN.shape[0])
                 df_21_Without_NaN = df_21_Without_NaN[~(df_21_Without_NaN == 0).any(axis=1)]

                   # Reset the index (optional)
                 df_21_Without_NaN = df_21_Without_NaN.reset_index(drop=True)

                  # Display the modified DataFrame
                 st.write("Number of Rows containing 0:",  numberRowsBefore - df_21_Without_NaN.shape[0])
                 st.write("Number of Rows After deleting Zero Values:", df_21_Without_NaN.shape[0])
                 st.dataframe(df_21_Without_NaN)
                 st.write(df_21_Without_NaN.shape)

                 # 5. Feature Scaling
                 st.write('#### 5. Feature Scaling')
                 df_21_Without_NaN = df_21_Without_NaN.loc[:, :]
                 df_timestamp = df_21_Without_NaN.iloc[:, 0]

                 df_ = df_21_Without_NaN.iloc[:, 1:]

                 train_prp = .005
                 train = df_.loc[:df_.shape[0] * train_prp]
                 test = df_.loc[df_.shape[0] * train_prp:]

                 #st.dataframe(train)
                 #st.write("Shape of Data:", train.shape)
                 #st.dataframe(test)
                 #st.write("Shape of Text Data:", test.shape)

                 # Standardize The Data
                 scaler = StandardScaler()
                 X_train = scaler.fit_transform(train)
                 X_test = scaler.transform(test)

                 #st.write("X train Shape:", X_train.shape)
                 st.write("Shape of The Test Data:", X_test.shape)
                 st.write('Reshape the Dimension of the Test Data for LSTM - Autoencoder Model')
                  # Reshape the Dimension of the Train and Test set for LSTM Model
                 X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                 X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

                 #st.write("X train Shape:", X_train.shape)
                 st.write("Test Data Reshaped:", X_test.shape)
                 st.write("---------------------")

                 st.title('Anomaly Detection in Test Data')

                 # Load the pre-trained anomaly detection model
                 st.write("LSTM - Autoencoder Model Loading...")

                 # Display model architecture details
                 st.write("### LSTM Model Architecture:")
                 st.write("This LSTM model has the following layers:")
                 for layer in model_loaded.layers:
                     st.write(layer.name, "-", layer.output_shape)
                
                 st.success("LSTM - Autoencoder Model Loaded")

                 X_pred = model_loaded.predict(X_test)
                 st.write("### Compute loss/error between Predicted and Test values")
                 # Compute loss/error between predicted and test values columns
                 #mse_loss = (np.mean(np.square(X_pred - X_test)))/100  # MSE loss
                 #mae_loss = (np.mean(np.abs(X_pred - X_test)))/100  # Mean absolute error (MAE) loss
                 #st.write(f"Mean squared error (MSE) Loss: {mse_loss}")
                 #st.write(f"Mean absolute error (MAE) Loss: {mae_loss}")
                 st.write("Mean absolute error (MAE) Loss:", (np.mean(np.abs(X_pred - X_test))) / 100)
                 st.write("Mean squared error (MSE) Loss:", (np.mean(np.square(X_pred - X_test))) / 100)
            
                 X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
                 X_pred = scaler.inverse_transform(X_pred)
                 X_pred = pd.DataFrame(X_pred, columns=test.columns)
                 X_pred.index = test.index

                 scores_3 = pd.DataFrame()
                 scores_3 = X_pred
                 scores_3['datetime'] = df_timestamp.loc[:]
                 reconstruction_errors = np.mean(np.abs(X_pred - test), axis=1)
                 scores_3['loss_mae'] = reconstruction_errors
                 threshold = 10
                 scores_3['Threshold'] = threshold
                 scores_3['Anomaly'] = np.where(scores_3["loss_mae"] > scores_3["Threshold"], 1, 0)
                 st.dataframe(scores_3)

                 # Error distribution i Test Data
                 fig = go.Figure(data=[go.Histogram(x=scores_3['loss_mae'])])
                 fig.update_layout(title="Error distribution", xaxis=dict(title="Loss Distribution between Predicted and Test Data"), yaxis=dict(title="Data Point Counts"))
                 st.plotly_chart(fig)
                 st.write("Threshold:", threshold)
                 #st.write(scores_3['Anomaly'].value_counts())

                 fig = go.Figure()
                 fig.add_trace(go.Scatter(x=scores_3['datetime'],  y=scores_3['loss_mae'], name="Loss"))
                 fig.add_trace(go.Scatter(x=scores_3['datetime'],  y=scores_3['Threshold'], name="Threshold"))
                 fig.update_layout(title="Error Time Series and Threshold",  xaxis=dict(title="DateTime"), yaxis=dict(title="Loss"))
                 st.plotly_chart(fig)


                 # New Threshold
                 #st.text("Set Threshold:")
                 #threshold = st.slider("Set Threshold ", 1.0, 5.0, 500)


                 anomalies = scores_3[scores_3['Anomaly'] == 1][['datetime','Cy1ExhGasOutletTemp', 'Cy2ExhGasOutletTemp', 'Cy3ExhGasOutletTemp', 'Cy4ExhGasOutletTemp', 'Cy5ExhGasOutletTemp', 'Cy6ExhGasOutletTemp', 'Cy7ExhGasOutletTemp', 'Cy8ExhGasOutletTemp', 'Cy9ExhGasOutletTemp', 'Cyl1_Pmax', 'Cyl2_Pmax', 'Cyl3_Pmax', 'Cyl4_Pmax', 'Cyl5_Pmax', 'Cyl6_Pmax', 'Cyl7_Pmax', 'Cyl8_Pmax', 'Cyl9_Pmax', 'Load', 'Power']]
                 anomalies = anomalies.rename(columns={
                                    'Cy1ExhGasOutletTemp':'Cy1ExhGasOutletTemp_anomalies',
                                    'Cy2ExhGasOutletTemp':'Cy2ExhGasOutletTemp_anomalies',
                                    'Cy3ExhGasOutletTemp':'Cy3ExhGasOutletTemp_anomalies',
                                    'Cy4ExhGasOutletTemp':'Cy4ExhGasOutletTemp_anomalies',
                                    'Cy5ExhGasOutletTemp':'Cy5ExhGasOutletTemp_anomalies',
                                    'Cy6ExhGasOutletTemp':'Cy6ExhGasOutletTemp_anomalies',
                                    'Cy7ExhGasOutletTemp':'Cy7ExhGasOutletTemp_anomalies',
                                    'Cy8ExhGasOutletTemp':'Cy8ExhGasOutletTemp_anomalies',
                                    'Cy9ExhGasOutletTemp':'Cy9ExhGasOutletTemp_anomalies',
                                    'Cy1ExhGasOutletTemp':'Cy1ExhGasOutletTemp_anomalies',
                                    'Cyl1_Pmax':'Cyl1_Pmax_anomalies',
                                    'Cyl2_Pmax':'Cyl2_Pmax_anomalies',
                                    'Cyl3_Pmax':'Cyl3_Pmax_anomalies',
                                    'Cyl4_Pmax':'Cyl4_Pmax_anomalies',
                                    'Cyl5_Pmax':'Cyl5_Pmax_anomalies',
                                    'Cyl6_Pmax':'Cyl6_Pmax_anomalies',
                                    'Cyl7_Pmax':'Cyl7_Pmax_anomalies',
                                    'Cyl8_Pmax':'Cyl8_Pmax_anomalies',
                                    'Cyl9_Pmax':'Cyl9_Pmax_anomalies',
                                    'Load':'Load_anomalies',
                                    'Power':'Power_anomalies'
                                    })
               
                 scores_1a = scores_3.merge(anomalies, left_index=True, right_index=True, how='left')

                 fig = go.Figure()
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy1ExhGasOutletTemp"], mode='lines', name='Cy1 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy2ExhGasOutletTemp"], mode='lines', name='Cy2 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy3ExhGasOutletTemp"], mode='lines', name='Cy3 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy4ExhGasOutletTemp"], mode='lines', name='Cy4 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy5ExhGasOutletTemp"], mode='lines', name='Cy5 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy6ExhGasOutletTemp"], mode='lines', name='Cy6 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy7ExhGasOutletTemp"], mode='lines', name='Cy7 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy8ExhGasOutletTemp"], mode='lines', name='Cy8 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy9ExhGasOutletTemp"], mode='lines', name='Cy9 ExhGasOutletTemp'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Load"], mode='lines', name='Load'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Power"], mode='lines', name='Power'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl1_Pmax"], mode='lines', name='Cyl 1 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl2_Pmax"], mode='lines', name='Cyl 2 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl3_Pmax"], mode='lines', name='Cyl 3 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl4_Pmax"], mode='lines', name='Cyl 4 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl5_Pmax"], mode='lines', name='Cyl 5 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl6_Pmax"], mode='lines', name='Cyl 6 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl7_Pmax"], mode='lines', name='Cyl 7 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl8_Pmax"], mode='lines', name='Cyl 8 Pmax'))
                 fig.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cyl9_Pmax"], mode='lines', name='Cyl 9 Pmax'))
                 
                 fig.update_layout(title_text="Test Data", width=800, height=600)
                 st.plotly_chart(fig)
                 

                 # Anomaly Detection
                 fig_2 = go.Figure()
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy1ExhGasOutletTemp"], mode='lines', name='Cy1 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy1ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 1 ', mode='markers', marker=dict(color="red", size=11, line=dict(color="red", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy2ExhGasOutletTemp"], mode='lines', name='Cy2 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy2ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 2 ', mode='markers', marker=dict(color="blue", size=11, line=dict(color="blue", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy3ExhGasOutletTemp"], mode='lines', name='Cy3 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy3ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 3 ', mode='markers', marker=dict(color="black", size=11, line=dict(color="black", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy4ExhGasOutletTemp"], mode='lines', name='Cy4 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy4ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 4 ', mode='markers', marker=dict(color="green", size=11, line=dict(color="green", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy5ExhGasOutletTemp"], mode='lines', name='Cy5 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy5ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 5 ', mode='markers', marker=dict(color="yellow", size=11, line=dict(color="yellow", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy6ExhGasOutletTemp"], mode='lines', name='Cy6 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy6ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 6 ', mode='markers', marker=dict(color="orange", size=11, line=dict(color="orange", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy7ExhGasOutletTemp"], mode='lines', name='Cy7 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy7ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 7 ', mode='markers', marker=dict(color="purple", size=11, line=dict(color="purple", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy8ExhGasOutletTemp"], mode='lines', name='Cy8 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy8ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 8 ', mode='markers', marker=dict(color="violet", size=11, line=dict(color="violet", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy9ExhGasOutletTemp"], mode='lines', name='Cy9 ExhGasOutletTemp'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Cy9ExhGasOutletTemp_anomalies"], name='Anomaly in Cyl 9 ', mode='markers', marker=dict(color="teal", size=11, line=dict(color="teal", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Load"], mode='lines', name='Load'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Load_anomalies"], name='Anomaly in Load ', mode='markers', marker=dict(color="magenta", size=11, line=dict(color="magenta", width=2))))
                 
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Power"], mode='lines', name='Power'))
                 fig_2.add_trace(go.Scatter(x=scores_1a["datetime_x"], y=scores_1a["Power_anomalies"], name='Anomaly in Power ', mode='markers', marker=dict(color="grey", size=11, line=dict(color="grey", width=2))))
                 
                 fig_2.update_layout(title_text="Anomalies Detected in Test Data with LSTM-AE", width=800, height=600)
                 st.plotly_chart(fig_2)

                 st.success("Time of Anomaly occurrence and Values in Test Data")
                 st.dataframe(scores_3[scores_3['Anomaly'] == 1][['datetime','Cy1ExhGasOutletTemp', 'Cy2ExhGasOutletTemp', 'Cy3ExhGasOutletTemp', 'Cy4ExhGasOutletTemp', 'Cy5ExhGasOutletTemp', 'Cy6ExhGasOutletTemp', 'Cy7ExhGasOutletTemp', 'Cy8ExhGasOutletTemp', 'Cy9ExhGasOutletTemp', 'Cyl1_Pmax', 'Cyl2_Pmax', 'Cyl3_Pmax', 'Cyl4_Pmax', 'Cyl5_Pmax', 'Cyl6_Pmax', 'Cyl7_Pmax', 'Cyl8_Pmax', 'Cyl9_Pmax', 'Load', 'Power']])

            
   if st.sidebar.checkbox("Display Training Accuracy of The Model", False):
         st.subheader("Training Accuracy of The Model")
         accuracy = 97
         st.write("### Accuracy:", accuracy, "%")
         image = Image.open('Figure_Acc_LSTM_AE_Cyl_Pmax.jpeg')
         st.image(image, caption='Training Accuracy')

   if st.sidebar.checkbox("Display Training Loss of The Model", False):
         st.subheader("Training Loss of The Model")
         loss = 0.0053
         st.write("### Loss:", loss)
         image = Image.open('Figure_Loss_LSTM_AE_Cyl_Pmax.jpeg')
         st.image(image, caption='Training Loss')

   if st.sidebar.checkbox("Data Preprocessing steps", False):
        st.subheader("Data Preprocessing steps")
        preprocess_steps()
    
   if st.sidebar.checkbox("Load New Data for Anomaly Detection", False):
        st.subheader("Load The DE2 New Test Data")
        df = load()
   
   # Perfrom Anomaly Detection
   #ExhGasInletTempA , ExhGasOutletTemp, LOInletPress, LOOutletTemp, RPM	
   

   # Code for Anomaly Predicition
   Anomaly_Prediction_Result = ''

   # Creating a buttun for Anomaly Prediction on User Input Data



if __name__ == '__main__':
   main()