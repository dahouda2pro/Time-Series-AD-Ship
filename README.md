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
