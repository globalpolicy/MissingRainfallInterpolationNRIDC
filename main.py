'''
1. Ensure that the station names are EXACT MATCHES between the cluster info file and the daily ppt file
2. Ensure that two different stations DO NOT share the same name

Inconsistency found in names of stations between cluster file and daily ppt data files:
1. Aiselukhark (Koshi)
2. Darchula (Lumbini and SudurPaschim) - same name. Rename one
3. SudurPaschim's all station names should be capitalized
4. Kanyam Tea Estate vs State
5. Okhaldhunga (trailing space) (Koshi)
6. Hetauda N.F.I.
7. Gausala (trailing space) (Madhesh)
8. Num (trailing space) (Koshi)
9. Terhathum (trailing space) (Koshi)
10. Himali Gaun (trailing space) (Koshi)
11. Chandra Gadhi (trailing space) (Koshi)

'''

import pandas as pd
import numpy as np
import os
import math
import random
import numba
import time
from numba.typed import List
from numba import prange

STATION_WISE_DATA_OUT=False
NUMBER_OF_SAMPLES=100
OLD_METHOD=False
P_EXPONENT=3.5

@numba.jit(nopython=True, parallel=True)
def performSamplingAndMetricsCalculation(targetStationTimeSeries, neighboringStationsTimeSeriesList, missingDataIndicesList, latLongList, correlationCoeffExponentP=4):
    # list of metrics calculated for each sample defined by missingDataIndicesList
    S_indexes=np.zeros(len(missingDataIndicesList))
    MAEs=np.zeros(len(missingDataIndicesList))
    Rs=np.zeros(len(missingDataIndicesList))
    
    for k in range(len(missingDataIndicesList)): # iterate over this list of list of missing data indices (essentially, iterate over given samples)
        missingIndicatorIndices=missingDataIndicesList[k]
        
        # calibration
        absoluteWeights=[] # list of weights for each neighboring station
        targetStationCalibrationData=[targetStationTimeSeries[i] for i in range(len(targetStationTimeSeries)) if i not in missingIndicatorIndices] # get the target station data for calibration
        for j in range(len(neighboringStationsTimeSeriesList)) : # iterate over this list of neighboring stations' timeseries
            neighboringStationData=neighboringStationsTimeSeriesList[j]
            neighboringStationCalibrationData=[neighboringStationData[i] for i in range(len(neighboringStationData)) if i not in missingIndicatorIndices] # get this neighboring station data for calibration
            
            d_i=haversine(latLongList[0][0],latLongList[0][1],latLongList[j+1][0],latLongList[j+1][1]) # dist. between target station and this neighboring station
            
            r_i=np.corrcoef(targetStationCalibrationData, neighboringStationCalibrationData)[0,1]
            if r_i<0:  r_i=0 # make -ve correlaction coeff. 0
            N_s=np.mean(np.array(targetStationCalibrationData))
            N_i=np.mean(np.array(neighboringStationCalibrationData))
            
            absoluteWeight=r_i**correlationCoeffExponentP*N_s/N_i*d_i**-2
            absoluteWeights.append(absoluteWeight) # add to the weight list of neighboring stations
            
        # validation
        predictedTargetStationData=np.zeros(len(missingIndicatorIndices)) # make an empty result vector the same size as the missing elements indices vector
        targetStationValidationData=np.array([targetStationTimeSeries[i] for i in missingIndicatorIndices])
        for j in range(len(neighboringStationsTimeSeriesList)): # iterate over this list of neighboring stations' timeseries
            neighboringStationData=neighboringStationsTimeSeriesList[j]
            neighboringStationValidationData=np.array([neighboringStationData[i] for i in missingIndicatorIndices])
            predictedTargetStationData=predictedTargetStationData+absoluteWeights[j]*neighboringStationValidationData # multiply each neighbor's vector with its weight and accumulate
        predictedTargetStationData=predictedTargetStationData/np.sum(np.array(absoluteWeights)) # normalize the prediction vector
        
        S_index=1-(np.sum(np.square(predictedTargetStationData-targetStationValidationData)))/(np.sum(np.square(np.abs(predictedTargetStationData-np.mean(targetStationValidationData))+np.abs(targetStationValidationData-np.mean(targetStationValidationData))) ))
        MAE=1/len(targetStationValidationData)*np.sum(np.abs(predictedTargetStationData-targetStationValidationData))
        R=np.corrcoef(targetStationValidationData, predictedTargetStationData)[0,1]
        S_indexes[k]=S_index
        MAEs[k]=MAE
        Rs[k]=R
    
    S_index_mean=np.mean(S_indexes)
    MAE_mean=np.mean(MAEs)
    R_mean=np.mean(Rs)
    
    return S_index_mean,MAE_mean,R_mean
            

def calculateWeightsForStationV2(targetStationInfo, clusterStations, testSampleProportion=0.1, correlationCoeffExponentP=4, method='NRIDC', numberOfSamples=100):
    targetStationName=targetStationInfo['Station Name']
    dataForState=masterDailyPpt[targetStationInfo['State']] # load the data for this state
    targetStationData=dataForState[['date',targetStationName]] # load the data for this targetStationInfo
    targetStationData.loc[:, 'date']=pd.to_datetime(targetStationData['date']) # convert the column to datetime type just in case

    # Ready the NEIGHBORHOOD DATA TABLE that'll hold columns: date, targetStation, neighboringStation1, neighboringStation2, ..., missingFlag via INNER JOIN on date, which is mandatory
    neighborhoodDataTable=targetStationData.copy() # start with date and the target station data
    for _,neighboringStationInfo in clusterStations.iterrows(): # iterate over all stations in this cluster, except for self
        neighboringStationName=neighboringStationInfo['Station Name']
        if(neighboringStationName==targetStationName):
            continue
        neighboringStationData=masterDailyPpt[neighboringStationInfo['State']][['date',neighboringStationName]]
        neighboringStationData.loc[:, 'date']=pd.to_datetime(neighboringStationData['date']) # convert the column to datetime type just in case
        neighborhoodDataTable=pd.merge(neighborhoodDataTable, neighboringStationData, on="date", how="inner") # perform an inner join on 'date' column. basically, tack on neighbor's data column
        neighborhoodDataTable=neighborhoodDataTable.dropna() # drop empty rows

    neighborsTimeSeriesList=List()
    allStationsLatLongList=List()
    allStationsLatLongList.append([targetStationInfo['Latitude'], targetStationInfo['Longitude']]) # start by adding target station's latlong
    for _,neighboringStationInfo in clusterStations.iterrows(): # iterate over all stations in this cluster, except for self
        neighboringStationName=neighboringStationInfo['Station Name']
        if(neighboringStationName==targetStationName):
            continue
        neighborsTimeSeriesList.append(List(neighborhoodDataTable[neighboringStationName].tolist())) # pull the timeseries data for this neighboring station from the neighborhood data table
        allStationsLatLongList.append([neighboringStationInfo['Latitude'], neighboringStationInfo['Longitude']]) # add this neighboring station's latlong to the list
    
    toValidateDataPositionsList=List()
    for i in range(numberOfSamples): # create a list of the specified number of samples, each representing the data points to be assumed missing for validation
        toValidateDataPositions=random.sample(range(len(neighborhoodDataTable)), int(testSampleProportion*len(neighborhoodDataTable)))
        toValidateDataPositionsList.append(List(toValidateDataPositions))
    
    if(len(neighborsTimeSeriesList)!=0): # for target station with no neighbor, numba gives error: "invalid operation on untyped list"
        S_index_mean,MAE_mean,R_mean = performSamplingAndMetricsCalculation(List(neighborhoodDataTable[targetStationName].tolist()), neighborsTimeSeriesList, toValidateDataPositionsList, allStationsLatLongList)
    else:
         S_index_mean,MAE_mean,R_mean=0,0,0
    return S_index_mean,MAE_mean,R_mean
    
    


def calculateWeightsForStation(targetStationInfo, clusterStations, testSampleProportion=0.1, correlationCoeffExponentP=4, method='NRIDC'):
    targetStationName=targetStationInfo['Station Name']
    dataForState=masterDailyPpt[targetStationInfo['State']] # load the data for this state
    targetStationData=dataForState[['date',targetStationName]] # load the data for this targetStationInfo
    targetStationData.loc[:, 'date']=pd.to_datetime(targetStationData['date']) # convert the column to datetime type just in case

    outputDict={}
    
    # Ready the NEIGHBORHOOD DATA TABLE that'll hold columns: date, targetStation, neighboringStation1, neighboringStation2, ..., missingFlag
    # Note: The 'missingFlag' column will be 0 or 1 and will decide whether the row will be used for calibration or validation
    neighborhoodDataTable=targetStationData.copy() # start with date and the target station data
    for _,neighboringStationInfo in clusterStations.iterrows(): # iterate over all stations in this cluster, except for self
        neighboringStationName=neighboringStationInfo['Station Name']
        if(neighboringStationName==targetStationName):
            continue
        neighboringStationData=masterDailyPpt[neighboringStationInfo['State']][['date',neighboringStationName]]
        neighboringStationData.loc[:, 'date']=pd.to_datetime(neighboringStationData['date']) # convert the column to datetime type just in case
        neighborhoodDataTable=pd.merge(neighborhoodDataTable, neighboringStationData, on="date", how="inner") # perform an inner join on 'date' column. basically, tack on neighbor's data column
        neighborhoodDataTable=neighborhoodDataTable.dropna() # drop empty rows
    neighborhoodDataTable['missing_flag']=0 # 0 == not missing, by default. not missing == to be used for calibration
    
    # Randomly mark a portion (testSampleProportion) of rows as missing i.e. set 'missing_flag' to 1
    toBeMarkedPositions=random.sample(range(len(neighborhoodDataTable)), int(testSampleProportion*len(neighborhoodDataTable)))
    neighborhoodDataTable.iloc[toBeMarkedPositions, neighborhoodDataTable.columns.get_loc('missing_flag')] = 1 # mark the rows at the positions determined above
        
    
    # Export for sanity check
    # neighborhoodDataTable.to_excel(f"{targetStationName}-PreCalibration.xlsx")
    
    # For calibration
    targetStationData=neighborhoodDataTable.loc[neighborhoodDataTable['missing_flag']==0][targetStationName] # update the target station data to contain only unmarked rows
    for _,neighboringStationInfo in clusterStations.iterrows(): # iterate over all stations in this cluster, except for self
        neighboringStationName=neighboringStationInfo['Station Name']
        if(neighboringStationName==targetStationName):
            continue
        neighboringStationData=neighborhoodDataTable.loc[neighborhoodDataTable['missing_flag']==0][neighboringStationName] # get this neighbor's column with unmarked rows
        
        numTotalDataPointsSize=len(neighboringStationData)
        numCalibrationDataPoints=int(numTotalDataPointsSize*(1-testSampleProportion))
        numValidationDataPointsSize=numTotalDataPointsSize-numCalibrationDataPoints

        d_i=haversine(neighboringStationInfo['Latitude'],neighboringStationInfo['Longitude'],targetStationInfo['Latitude'],targetStationInfo['Longitude'])
        if(method=='NRIDC'):
            r_i=np.corrcoef(targetStationData, neighboringStationData)[0,1]
            if r_i<0:  r_i=0 # make -ve correlaction coeff. 0
            N_s=targetStationData.mean()
            N_i=neighboringStationData.mean()
            
            absoluteWeight=r_i**correlationCoeffExponentP*N_s/N_i*d_i**-2
        elif(method=="IDW"):
            absoluteWeight=1/d_i**2
        
        outputDict[neighboringStationName]={
            'totNoOfDataPoints':numTotalDataPointsSize,
            'numOfCalibrationDataPoints':numCalibrationDataPoints,
            'numOfvalidationDataPoints':numValidationDataPointsSize,
            'absoluteWeight':absoluteWeight
        }
    
    '''
    outputDict's structure should be as follows:
    
    {
        'Salyan Bazar':            
            {
                'totNoOfDataPoints':numTotalDataPointsSize,
                'numOfCalibrationDataPoints':numCalibrationDataPoints,
                'numOfvalidationDataPoints':numValidationDataPointsSize,
                'absoluteWeight':absoluteWeight
            },
        'Daman':            
            {
                'totNoOfDataPoints':numTotalDataPointsSize,
                'numOfCalibrationDataPoints':numCalibrationDataPoints,
                'numOfvalidationDataPoints':numValidationDataPointsSize,
                'absoluteWeight':absoluteWeight
            },
        ............
    }

    The keys are the target station's neighboring stations' names
    
    '''
    
    # Export for sanity check
    # outputDictDf=pd.DataFrame.from_dict(outputDict)
    # outputDictDf.to_excel(f"{targetStationName}-PostCalibration.xlsx")
    
    
    # For validation
    neighborhoodDataTable['target_station_prediction']=0 # add targetStationPrediction column
    
    targetStationData=neighborhoodDataTable.loc[neighborhoodDataTable['missing_flag']==1][targetStationName] # update the target station data to contain only MARKED rows
    
    # calculate sum of absolute weights to later calculate the relative weights for each neighbor
    sumOfWeights=0
    for neighboringStationName in outputDict.keys():
        sumOfWeights=sumOfWeights+outputDict[neighboringStationName]['absoluteWeight']
        
    for _,neighboringStationInfo in clusterStations.iterrows(): # iterate over all stations in this cluster, except for self
        neighboringStationName=neighboringStationInfo['Station Name']
        if(neighboringStationName==targetStationName):
            continue
        neighboringStationData=neighborhoodDataTable.loc[neighborhoodDataTable['missing_flag']==1][neighboringStationName] # get this neighbor's column with MARKED rows
        neighborhoodDataTable['target_station_prediction']=neighborhoodDataTable['target_station_prediction']+outputDict[neighboringStationName]['absoluteWeight']*neighboringStationData # vector (column) accumulation

    neighborhoodDataTable['target_station_prediction']=neighborhoodDataTable['target_station_prediction'] / sumOfWeights # normalization
    
    # Export for sanity check
    # neighborhoodDataTable.to_excel(f"{targetStationName}-PostValidation.xlsx")
    
    # calculate accuracy metrics
    targetStationPredictedData=neighborhoodDataTable.loc[neighborhoodDataTable['missing_flag']==1]['target_station_prediction']
    S_index=1-(np.sum(np.square(targetStationPredictedData-targetStationData)))/(np.sum(np.square(np.abs(targetStationPredictedData-np.mean(targetStationData))+np.abs(targetStationData-np.mean(targetStationData))) ))
    MAE=1/len(targetStationData)*np.sum(np.abs(targetStationPredictedData-targetStationData))
    R=np.corrcoef(targetStationData, targetStationPredictedData)[0,1]
    
    # Return the calculated weights for all neighbors as a dict., the accuracy metrics and the full neighborhood table containing the sample used for calibration and validation
    return outputDict, S_index, MAE, R, neighborhoodDataTable
    
        
def loadAllDailyPptData(workingDir):
    pptDataDict=dict()
    states=['Bagmati','Gandaki','Koshi','Madhesh','Lumbini','SudurPaschim'] # must correspond with xlsx filenames in workingDir
    for state in states:
        filePath=workingDir+'\\'+state+'.xlsx'
        if os.path.isfile(filePath):
            pptDataForState=pd.read_excel(filePath,sheet_name='data') # load the state's xlsx file's 'data' sheet
            pptDataDict[state]=pptDataForState
    return pptDataDict            
    

@numba.njit
# Calculate distance between two latlong points in kilometers
def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371.0

    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in kilometers
    distance = R * c
    return distance

# print(haversine(29.84011, 80.53886, 27.68333, 87.78333))

workingDir=r"C:\Users\s0ft\Desktop\NepalHydrologyResearch\GapFillingMethodsCodeAndFiles" # NOTE: CHANGE THIS ACCORDINGLY
clustersInfoFilename="ClusteredStationsWithNameK=7.xlsx" # AND THIS

# Load all available daily ppt data to memory. (Could take a few 10s of seconds)
masterDailyPpt=loadAllDailyPptData(workingDir)
#print(masterDailyPpt)

# Read main clusters info table
clustersInfo=pd.read_excel(f'{workingDir}\\{clustersInfoFilename}', sheet_name='Clustering for K equals to 7')
clusters=np.unique((clustersInfo['Cluster'].to_numpy())) # get unique clusters
#print(clusters)

# Iterate through the clusters
clusterMetricsDf=pd.DataFrame(columns=['ClusterIndex','S-index','MAE','R']) # for output
for clusterIndex in clusters:
    clusterStations=clustersInfo[clustersInfo['Cluster']==clusterIndex] # stations belonging to this cluster
    #print(f'Cluster:{clusterIndex}\nStnNames:{clusterStations['Station Name'].values}')
    
    cluster_S_index_mean=0
    cluster_MAE_mean=0
    cluster_R_mean=0
    for _,station in clusterStations.iterrows(): # iterate over all stations in this cluster
        
        time_start=time.perf_counter()
        
        if(OLD_METHOD):
            # sampling block start
            outputDict_samples=[]
            S_index_samples=[]
            MAE_samples=[]
            R_samples=[]
            for i in range(NUMBER_OF_SAMPLES): # calibrate and validate this many times. required coz we select the data points for cali/vali at random.
                outputDict, S_index, MAE, R, neighborhoodDataTable = calculateWeightsForStation(station, clusterStations, method="NRIDC", correlationCoeffExponentP=P_EXPONENT)
                outputDict_samples.append(outputDict)
                S_index_samples.append(S_index)
                MAE_samples.append(MAE)
                R_samples.append(R)
            
            # calculate sample means of metrics
            S_index_sample_mean=np.average(S_index_samples)
            MAE_sample_mean=np.average(MAE_samples)
            R_sample_mean=np.average(R_samples)
            
            # calculate sample means of absolute weights of neighboring stations
            outputDict_sample_mean={} # new value and new structure for outputDict. tot num. of calibration and validation data points no longer makes sense here.
            for neighboringStationName in outputDict_samples[0].keys():
                absoluteWeight_sample_mean=0 # initialize the sample mean for absolute weight for this neighboring station
                for outputDict in outputDict_samples:
                    absoluteWeight_sample_mean+=outputDict[neighboringStationName]['absoluteWeight']
                absoluteWeight_sample_mean/=len(outputDict_samples)
                outputDict_sample_mean[neighboringStationName]=absoluteWeight_sample_mean
            # sampling block end
        else:
            S_index_sample_mean,MAE_sample_mean,R_sample_mean=calculateWeightsForStationV2(station, clusterStations, method="NRIDC", correlationCoeffExponentP=P_EXPONENT)
        
        time_end=time.perf_counter()
        time_duration=time_end-time_start
        print(f'Cluster #{clusterIndex} - Station "{station['Station Name']}" - {time_duration:.3f} s\n')
        
        # calculate average metrics for the whole cluster
        cluster_S_index_mean+=S_index_sample_mean/len(clusterStations)
        cluster_MAE_mean+=MAE_sample_mean/len(clusterStations)
        cluster_R_mean+=R_sample_mean/len(clusterStations)
        
        
        if STATION_WISE_DATA_OUT:
            # write output to file with the same name as the station
            stationName=station['Station Name']
            fullPathToSave=f'{workingDir}\\OUTPUT\\{stationName}.xlsx'
            os.makedirs(os.path.dirname(fullPathToSave), exist_ok=True)
            with pd.ExcelWriter(fullPathToSave) as xlwriter:
                outputDictDf=pd.DataFrame.from_dict(outputDict_sample_mean)
                outputDictDf.to_excel(xlwriter, sheet_name='Weights')
                
                metricsDict={
                    "S-index":S_index_sample_mean,
                    "MAE":MAE_sample_mean,
                    "R":R_sample_mean
                    }
                metricsDictDf=pd.DataFrame([metricsDict])
                metricsDictDf.to_excel(xlwriter,sheet_name='Metrics')
                
                neighborhoodDataTable.to_excel(xlwriter,sheet_name='NeighborhoodTable(Last Sample)')

    # update the cluster metrics dataframe with this cluster's summary
    new = pd.DataFrame(columns=clusterMetricsDf.columns, data=[[clusterIndex,cluster_S_index_mean,cluster_MAE_mean,cluster_R_mean]])
    clusterMetricsDf = pd.concat([clusterMetricsDf, new], axis=0)
    
    # print(f'Cluster: {clusterIndex}\n')
    # print('\n')
    # print(f'Mean S-index: {cluster_S_index_mean}\n')
    # print(f'Mean MAE: {cluster_MAE_mean}\n')
    # print(f'Mean R: {cluster_R_mean}\n')
    # print("="*50)

# write the cluster summary dataframe to csv
fullPathToSave=f'{workingDir}\\OUTPUT\\ClusterPerf-NRIDC-P={P_EXPONENT}.csv'
os.makedirs(os.path.dirname(fullPathToSave), exist_ok=True)
clusterMetricsDf.to_csv(fullPathToSave)