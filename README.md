# Time-Series-Anomaly Detection in Ship Data

# 1. Multivariate Time Series Anomaly Detection (MTSAD)
### LSTM_AE_MTS_Ex_Gas_Temp_Pmax_BrgDE_Test : We selected 30 features : 
   ### time,
   ### BrgDE_Temp1,
   ### BrgDE_Temp2,
   ### BrgDE_Temp3,
   ### BrgDE_Temp4,
   ### BrgDE_Temp5,
   ### BrgDE_Temp6,
   ### BrgDE_Temp7,
   ### BrgDE_Temp8,
   ### BrgDE_Temp9,
   ### BrgDE_Temp10,
   ### BrgDE_Temp11,
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
   ### from DE2Thing_HMD8310 and DG2Thing_HMD8310

After selecting the 30 features, We split the data into Training and Test set/
 * We build a Long Term Short Memory -  Autoencoder (LSTM-AE) model
 * We trained the model using the training set
 * We saved the trained model as 'Model_CyExGas_CyPmax_BrgDE.h5'

# 2. Deployment of MTSAD
### deployed_DE2_n_DG2_model file 

## To run the Streamlit Application : 
* streamlit run deployed_DE2_n_DG2_model.py

