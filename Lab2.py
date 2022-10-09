import types
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler,OneHotEncoder,LabelEncoder,OrdinalEncoder
from sklearn.cluster import KMeans,DBSCAN,AffinityPropagation
from pyclustering.cluster.clarans import clarans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns


#
#AutoML (data, listOfScaler, listOfModel, listOfParam)
#
#It is a custom function that automatically executes the combination we created and returns the results to the list and visulaize the results.
#
#Objects supported by the current program, except for these objects, are ignored.
#
#   Encoder - OneHotEncoder, LabelEncoder, OrdinalEncoder
#   Scaler  - StandardScaler, MinMaxScaler, RobustScaler
#   Model   - KMeans, DBSCAN, AffinityPropagation, clarans, GaussianMixture
#
########################Parameters###########################
#
#data : DataFrame
#
#   It receives data from The California Housing Prices Dataset
#
#listOfScaler : List of scaler object
#
#   List of scaler object
#
#listOfModel : List of model object
#
#   List of model object
#
#listOfParam : List of dictionaries
# 
#   List of dictionaries of hyperparameters
#
#   Dictionary's key is the class of the model, and the value is the list of dictionaries of hyperparameters
#
#   example
#
#       listOfScaler = [StandardScaler(), 
#                       RobustScaler()]
#    
#       listOfModel = [DBSCAN(),
#                       KMeans(),
#                       GaussianMixture()]
#
#       listOfParam = {KMeans:[{'n_clusters':2},{'n_clusters':3},{'n_clusters':4},{'n_clusters':5},{'n_clusters':6},{'n_clusters':7},{'n_clusters':8}],
#                       DBSCAN:[{'eps':0.025, 'min_samples':15},{'eps':0.025, 'min_samples':17},{'eps':0.025, 'min_samples':19},{'eps':0.03, 'min_samples':18},{'eps':0.03, 'min_samples':20},{'eps':0.03, 'min_samples':22}],
#                       GaussianMixture:[{'n_components': 2},{'n_components': 3},{'n_components': 4},{'n_components': 5},{'n_components': 6},{'n_components': 7},{'n_components': 8}]}
#
#

def AutoML (data, listOfScaler, listOfModel,listOfParam):
    scaledData = []
    data_Y = data["median_house_value"]
    data_X = data.drop(labels=["median_house_value"],axis=1)
    column = list(data_X.columns)

    #Scale data using all the scalers received as parameters and insert them into the scaledData list.
    for scaler in listOfScaler:
        if scaler is not None and isinstance(scaler,(StandardScaler,MinMaxScaler,RobustScaler)):
            scaledData.append(pd.DataFrame(scaler.fit_transform(data_X),columns = column))
        else:
            scaledData.append([])
            print(str(scaler) + " is ignored")
    
    #Train all models received as parameters to all scaled datas, respectively.
    for model in listOfModel :
        if model is not None and isinstance(model,(KMeans,DBSCAN,clarans,GaussianMixture, AffinityPropagation)):
            
            #Create a subplot to match the number of scalers and parameters
            fig, ax = plt.subplots(len(listOfScaler)*2, len(listOfParam.get(type(model))),figsize=(20,10))
            fig.suptitle(str(model))
            for scalerIndex, scaler in enumerate(listOfScaler):

                #Set ylable as the scaler.
                ax[scalerIndex*2][0].set_ylabel(str(scaler))
                ax[scalerIndex*2+1][0].set_ylabel("Compare")

                if scaler is not None and isinstance(scaler,(StandardScaler,MinMaxScaler,RobustScaler)):

                    #Check and learn all the parameters and put the clustering results in the cluster column.
                    #Several if-else statements were used because the use of functions was slightly different, but the results were the same.
                    for pramIndex,param in  enumerate(listOfParam.get(type(model))):
                        if isinstance(model,clarans):
                            cluster = clarans(scaledData[scalerIndex].values.tolist(),param[0],param[1],param[2])
                            cluster.process()
                            label = cluster.get_clusters()
                            emptyList = [0 for i in range(len(scaledData[scalerIndex]))]
                            scaledData[scalerIndex]['cluster']=emptyList
                            for labelNumber,labelList in enumerate(label):
                                for labelIndex in labelList:
                                    scaledData[scalerIndex].loc[labelIndex,'cluster']=labelNumber
                        else :

                            #If the model is Affinity Propagation, add hard-coded parameters
                            #The reasons are written in the final report.
                            if isinstance(model,AffinityPropagation):
                                param.update({'damping':0.9,'max_iter':1000})
                            cluster = model.set_params(**param)
                            if isinstance(model,GaussianMixture):
                                scaledData[scalerIndex]['cluster'] = cluster.fit_predict(scaledData[scalerIndex])
                            else: 
                                scaledData[scalerIndex]['cluster'] = cluster.fit(scaledData[scalerIndex]).labels_


                        #If there is data identified as noise in the density algorithm, the data is excluded from the comparison with median_house_value
                        noiseIndex = scaledData[scalerIndex][scaledData[scalerIndex]['cluster'] == -1].index
                        noiseCleanedData = scaledData[scalerIndex].drop(noiseIndex)
                        noiseCleanedValue = data_Y.drop(noiseIndex)

                        #Sort median_house_value in ascending order for comparison with cluster
                        sorted = noiseCleanedValue.sort_values().reset_index(drop=True)
                        clusterResult = noiseCleanedData.groupby('cluster')['room_per_bedroom'].count()

                        index = 0
                        rangeToCluster = []

                        #Divide median_house_value equal to the size of each cluster
                        for clusterSize in clusterResult:
                            index = index+clusterSize
                            rangeToCluster.append(sorted[index-1])

                        dataToCompare = pd.concat([noiseCleanedData,
                                                    pd.DataFrame(np.digitize(noiseCleanedValue.values,bins=rangeToCluster,right=True),columns=['median_house_value'])],axis=1)

                        compareGroup = dataToCompare.groupby(['cluster','median_house_value'])['room_per_bedroom'].count()

                        x=[]
                        y=[]
                        #We do not know which subset of median_house_value the cluster represents. 
                        #Therefore, determine that it is closest to the subset that contains the most and calculate the hit rate by max divided cluster size
                        for clusterNumber, clusterSize in enumerate(clusterResult):
                            x.append(clusterNumber)
                            try :
                                y.append(np.max(compareGroup[clusterNumber])/clusterSize)
                            except:
                                y.append(0)

                        #Visualization of accuracy
                        ax[scalerIndex*2+1][pramIndex].bar(x,y)
                        ax[scalerIndex*2+1][pramIndex].axes.xaxis.set_ticks([])
                        ax[scalerIndex*2+1][pramIndex].axes.yaxis.set_ticks([0,0.5,1])

                        #Visualization of Cluster
                        numOfCluster = len(set(scaledData[scalerIndex]['cluster']))
                        colors = plt.cm.Spectral(np.linspace(0, 1, numOfCluster))
                        for k, col in zip(range(numOfCluster), colors):
                            my_members = (scaledData[scalerIndex]['cluster'] == k)
                            ax[scalerIndex*2][pramIndex].scatter(scaledData[scalerIndex]['median_income'][my_members], 
                                                               scaledData[scalerIndex]['room_per_bedroom'][my_members],
                                                               color = col,
                                                               alpha = 0.5)
                        ax[scalerIndex*2][pramIndex].axes.xaxis.set_ticks([])
                        ax[scalerIndex*2][pramIndex].axes.yaxis.set_ticks([])
                        ax[scalerIndex*2][pramIndex].set_title(str(param))

                        #Drop the clustering column because it can affect the calculation of the following algorithms
                        scaledData[scalerIndex].drop(labels=["cluster"],axis=1,inplace=True)

                #If there is any ignored, it is explicitly shown
                else:
                    print(str(scaler) + " is ignored")
            plt.tight_layout()
            plt.show()

        else:
            print(str(model) + " is ignored")

#Parameters for automation functions
listOfScaler = [StandardScaler(), 
                MinMaxScaler(), 
                RobustScaler()]

listOfEncoder = [OneHotEncoder(sparse = False),
                 LabelEncoder(),
                 OrdinalEncoder()]

listOfModel = [DBSCAN(),
               KMeans(),
               GaussianMixture()
               ]
'''            
               KMeans(),
               DBSCAN(),
               clarans(),
               GaussianMixture(),
               AffinityPropagation()
'''

listOfParam = {KMeans:[{'n_clusters':2},{'n_clusters':3},{'n_clusters':4},{'n_clusters':5},{'n_clusters':6},{'n_clusters':7},{'n_clusters':8}],
               DBSCAN:[{'eps':0.025, 'min_samples':15},{'eps':0.025, 'min_samples':17},{'eps':0.025, 'min_samples':19},{'eps':0.03, 'min_samples':18},{'eps':0.03, 'min_samples':20},{'eps':0.03, 'min_samples':22}],
               clarans:[[2,3,5],[3,3,5],[4,3,5],[5,3,5],[6,3,5],[7,3,5],[8,3,5]],
               GaussianMixture:[{'n_components': 2},{'n_components': 3},{'n_components': 4},{'n_components': 5},{'n_components': 6},{'n_components': 7},{'n_components': 8}],
               AffinityPropagation:[{'preference':-10},{'preference':-30},{'preference':-50},{'preference':-80}]}

#Data definition
df = pd.read_csv("housing.csv")

#Print data information
print(df.info())

#Handling Missing Values
#And since there are quite a few records, we dropped the missing values.
df.dropna(inplace=True)
df.reset_index(drop=True,inplace=True)

#Store and drop 'ocean_proximity'
label = df['ocean_proximity']
df = df.drop(columns='ocean_proximity')

#Encode data using all the encoder and insert them into the df Dataframe.
#If there is wrong encoder, print 'encoder' is ignored.
for encoder in listOfEncoder:
    if(isinstance(encoder,(OneHotEncoder,LabelEncoder,OrdinalEncoder))):
        encoded = encoder.fit_transform(label.values.reshape(-1,1))
        if(isinstance(encoder,OneHotEncoder)):
            df = pd.concat([df,pd.DataFrame(encoded,columns=encoder.categories_[0])],axis=1)
        elif(isinstance(encoder,LabelEncoder)):
            df = pd.concat([df,pd.DataFrame(encoded,columns=["labelEncoder"])],axis=1)
        elif(isinstance(encoder,OrdinalEncoder)):
            df = pd.concat([df,pd.DataFrame(encoded,columns=["ordinalEncoder"])],axis=1)
    else:
        print(str(encoder) + " is ignored")


df['room_per_bedroom'] = df['total_rooms']/ df['total_bedrooms']

#Visualization of heat map for checking correlation coefficients
corr= df.corr()
sns.heatmap(corr, annot = True, cmap = 'RdYlBu_r',vmin = -1, vmax = 1)
plt.show()

#We determined that the valid feature was a feature with a correlation coefficient of 0.35 or more.
#So the set threshold was 0.35, and all the features below it were droped.
#It then prints the information of the changed data.
index = corr[corr["median_house_value"]<0.35].index
df.drop(index,inplace=True,axis=1)
print(df.info())

#Visualization of heat map for checking correlation coefficients
corr= df.corr()
sns.heatmap(corr, annot = True, cmap = 'RdYlBu_r',vmin = -1, vmax = 1)
plt.show()

#Function call
result = AutoML(df, listOfScaler, listOfModel, listOfParam) 