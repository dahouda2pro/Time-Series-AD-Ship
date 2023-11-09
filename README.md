# Time-Series-Anomaly Detection in Ship Data

# 1. Time Series Anomaly Detection in Generator Engine (GE):
### LSTM_AE_MTS_Ex_Gas_Temp_1_9 file : We selected 21 features :
   ### time,
   ### Cy1ExhGasOutletTemp,
   ### Cy2ExhGasOutletTemp,
   ### Cy3ExhGasOutletTemp,
   ### Cy4ExhGasOutletTemp,
   ### Cy5ExhGasOutletTemp,
   ### Cy6ExhGasOutletTemp,
   ### Cy7ExhGasOutletTemp,
   ### Cy8ExhGasOutletTemp,
   ### Cy9ExhGasOutletTemp,
   ### Cyl1_Pmax, 
   ### Cyl2_Pmax,
   ### Cyl3_Pmax,
   ### Cyl4_Pmax,
   ### Cyl5_Pmax,
   ### Cyl6_Pmax,
   ### Cyl7_Pmax,
   ### Cyl8_Pmax,
   ### Cyl9_Pmax,
   ### Load,
   ### Power
   ### from DE2Thing_HMD8310 Measurement

After selecting the 6 features, We split the data into Training and Test set/
 * We build a Long Term Short Memory -  Autoencoder (LSTM-AE) model
 * We trained the model using the training set
 * We saved the trained model as 'CyExGas_CyPmax_Model.h5'

 # 2. Time Series Anomaly Detection in Diesel Engine (DE)
 ### LSTM_AE_MTS_BrgDE_Temp_1_9 file : We selected 10 features:

