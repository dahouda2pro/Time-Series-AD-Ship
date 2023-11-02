# Time-Series-Anomaly Detection in Ship Data

# 1. Time Series Anomaly Detection in Generator Engine (GE):
### LSTM_AE_GE_Model file : We selected 6 features :
   ##### 1. time,
   ##### 2. ExhGasInletTempA, 
   ##### 3. ExhGasOutletTemp,
   ##### 4. LOInletPress,
   ##### 5. LOOutletTemp,
   ##### 6. RPM'
   ### from GE2TurboCharger1Thing_HMD8310 Measurement

After selecting the 6 features, We split the data into Training and Test set/
 * We build a Long Term Short Memory -  Autoencoder (LSTM-AE) model
 * We trained the model using the training set
 * We saved the trained model as 'Generator_Engine-LSTM_AE_model.h5'

 # 2. Time Series Anomaly Detection : DE2Thing_HMD8310 Measurement
 ### LSTM_AE_MTS_Ex_Gas_Temp_1_9 file : We selected 10 features:
 ##### 1. time,
 ##### 2. Cy1ExhGasOutletTemp,
 ##### 3. Cy2ExhGasOutletTemp,
 ##### 4. Cy3ExhGasOutletTemp,
 ##### 5. Cy4ExhGasOutletTemp,
 ##### 6. Cy5ExhGasOutletTemp,
 ##### 7. Cy6ExhGasOutletTemp,
 ##### 8. Cy7ExhGasOutletTemp,
 ##### 9. Cy8ExhGasOutletTemp,
 ##### 10. Cy9ExhGasOutletTemp

 * We colleced the Data From '2022-08-29T23:28:00Z' To '2023-04-26T08:00:00Z'

 !!! * Columns (38,39,55,56,57,58,59,60,61,62,63,64,65,67,68,69,70,73,74,75,76,77,78,79,80,81,89,91,103,104,105,115,122) have mixed types. 

 - Exhaust Gas Temperature 1 to 9 : Data type : Boolean
 - Cy1ExhGasOutletTemp 1 to 9 : Data type : Float
  We checked the missng values : 1. dropna(): To drop the Not a Number (NaN) and flllna() fill missing values with mean of the columns.
  We split the data into training set (80%) and Test set (20%)
  We build and train the model on the training set
  
  ** Model Accuracy : 92%
  ** Model Loss : 0.00234