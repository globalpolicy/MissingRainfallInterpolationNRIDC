# MissingRainfallInterpolationNRIDC
A Python script that calibrates and validates the NRIDC and IDW methods of missing daily rainfall values according to a station's immediate neighbors

This was written as part of an analysis on various missing rainfall interpolation techniques for Nepal. There are two interpolation methods included in the script: NRIDC and IDW.
# How it works:
1. Collects the daily rainfall data for all stations specified in the Excel spreadsheet pointed to by the `clustersInfoFileName` variable. The structure of the file should look like the following (*ClusteredStationsWithNameK=7.xlsx*):
   ![image](https://github.com/user-attachments/assets/e7e2cda4-e208-41ae-84fb-4357de65b3cc)
   It also expects Excel spreadsheet files containing the daily rainfall data for all the stations used for all the states mentioned in the clusters file above. These state-specific spreadsheets look like the following (*Bagmati.xlsx*):
   ![image](https://github.com/user-attachments/assets/76516989-e158-48f0-adae-0a579394c371)
   The directory structure looks like:
   ![image](https://github.com/user-attachments/assets/7dce3cd7-e409-47bf-9589-4904b1a4eed2)

2. For each station in each cluster, it generates a neighborhood table that basically looks like:
   ![image](https://github.com/user-attachments/assets/f6a9166c-2890-4c10-b5e9-7a8d995fb0fe)
   The first station column is the daily rainfall timeseries of the target station (that is being calibrated and validated) and the rest are the neighboring stations in the cluster. In generating this neighborhood table, only the common dates whose data are available for all stations in the cluster are taken (achieved using inner join on the stations' daily rainfall timeseries). Any row having any empty entry is excluded (using `dropna()`)

3. A user-defined percentage of the total observations (say 10%) is randomly marked as missing (by setting the `missing_flag` to 1) and the rest are used for calibration of the model i.e. calculating weights for the neighboring stations (for NRIDC only, since IDW is   indifferent to neighboring stations' daily rainfalls). The weights are filled in a sheet called `Weights`:
   ![image](https://github.com/user-attachments/assets/3345aa76-e129-459d-a1b2-e345ccdebad0)

   
4. Rainfall predictions are made for rainfall values for the marked rows of the target station using the calculated weights and the neighboring stations' rainfalls and filled into the `target_station_prediction` column.
5. Using the predicted and the observed values for the target station, performance metrics are calculated (currently `S-index`, `MAE` and `R`). These metrics are put into a sheet called `Metrics`:
   ![image](https://github.com/user-attachments/assets/14937a70-0453-473c-a5cb-0547bbf9f1cc)
6. The neighborhood table, weights and metrics are all written to a single Excel file named the same as the target station:
   ![image](https://github.com/user-attachments/assets/948266c9-a1f1-4d4c-9fce-d045a30d31a1)

7. This behavior - whether to produce station-specific output files - can be toggled on or off using the `STATION_WISE_DATA_OUT` variable.
8. The calibration and validation steps described above are carried out for a user-specified number of times (defined by `numberOfSamples`) for each target station in a cluster and the mean performance metrics for all samplings are averaged out for the whole cluster considering each station as the target one at a time. The result of all this - a csv file such as the following (where I have played around with the value of the `p` exponent in the NRIDC expression and obtained a number of different results):
   ![image](https://github.com/user-attachments/assets/c03fd4b5-79f4-45b7-9643-1fd255922db3)
   
  The contents of one such file:
  ![image](https://github.com/user-attachments/assets/c1fc05b5-5eed-43a6-a99e-4e39d2e8de4a)
 
