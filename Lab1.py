import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

#
#finder (data, listOfScaler, listOfModel, listOfKFold, listOfParam)
#
#It is a custom function that automatically executes the combination we created and returns the results to the list.
#
########################Parameters###########################
#
#data : DataFrame
#
#   It receives data from breast-cancer-wisconsin.data
#
#listOfScaler : List of scaler object
#
#   List of scaler object
#   
#listOfModel : List of model object
#
#   List of model object
#
#listOfKFold : List of int
#
#   Specifies the number of K's in the k-fold as a list. 
#   There is no specific limit to the length of the list, but the program execution time can be very long.
#
#listOfParam : List of dictionaries
#
#   List of dictionaries of hyperparameters to enter GridSearchCV.
#   You must match the index of the model you put in the listOfModel 
#   with the index of the dictionary of the parameters you want to test on that model.
#
#   example
#
#       listOfModel = [DecisionTreeClassifier(criterion='gini'), 
#                      DecisionTreeClassifier(criterion='entropy'), 
#                      SVC()]
#
#       listOfParam = [{'max_depth':[2,3,5,7,10], 'min_samples_split':[2,3,5]}, <-- DecisionTreeClassifier(criterion='gini')
#                      {'max_depth':[2,3,5,7,10], 'min_samples_split':[2,3,5]}, <-- DecisionTreeClassifier(criterion='entropy')
#                      {'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1]}] <-- SVC()
#
#########################Returns#############################
#
#result : List of list
#
#   2-dimensional list containing the results of all combinations
#
#   [[0],[1],[2]....[n]]
#
#   result[n][0] : str        => scaler used 
#   result[n][1] : str        => model used
#   result[n][2] : int        => k used
#   result[n][3] : dictionary => best combination of parameters
#   result[n][4] : float      => score
#

def finder (data, listOfScaler, listOfModel, listOfKFold, listOfParam):
    scaledData = []
    result = []
    column = list(data.columns)
    data_X = data.drop(labels=["Class"],axis=1)
    data_Y = data["Class"].to_numpy()

    #Scale data using all the scalers received as parameters and insert them into the scaledData list.
    for scaler in listOfScaler:
        if scaler is not None:
            scaledData.append(pd.DataFrame(np.hstack((scaler.fit_transform(data_X),data_Y[:, np.newaxis])),columns = column))

    #Train all models received as parameters to all scaled datas, respectively.
    for scalerIndex, scaler in enumerate(listOfScaler):
        if scaler is not None:
            for modelIndex, model in enumerate(listOfModel):
                if model is not None:
                    #Repeat all k received as parameters to cross-validate
                    #Use gridsearchCV to perform K-fold cross-validation and hyperparameter tuning at the same time, And save the best results.
                    #In addition, since we test the classification algorithm, we used StratifiedKFold, not just KFold, for better results.
                    for k in listOfKFold:
                        kFold = StratifiedKFold(n_splits = k, shuffle=True, random_state=0)
                        gridModel = GridSearchCV(model, param_grid=listOfParam[modelIndex],cv=kFold,scoring='accuracy', refit=True)
                        gridModel.fit(scaledData[scalerIndex].drop(labels=["Class"],axis=1),scaledData[scalerIndex]["Class"])
                        result.append([str(scaler),str(model),k,gridModel.best_params_,gridModel.best_score_])

    return result

#
#print_result(result, printAll)
#
#Custom functions for output
#
#Prints the results of the finder function in a good way.
#By default, the output is in ascending order of the average score.
#
########################Parameters######################
#
#result : List
#
#   Return value of finder function
#
#printAll : Boolean
#
#   If True, outputs all results of all combinations. 
#   If False, only the five with the best average score are printed.
#

def print_result(result, printAll):
    result.sort(key=lambda x:x[4])
    printRange = range(len(result))

    #If printAll is False, only the top five, not the whole, are printed.
    if printAll == False:
        result = result[-5:]
        printRange = range(5)
        print("\n\nTop 5 Combinations :\n")
    else:
        print("\n\nAll Combinations :\n")

    for i in printRange:
        sentence = ""

        sentence += result[i][0][:-2]

        sentence += " & "

        if result[i][1][0:2] == "SV" : sentence += result[i][1][:-2]
        elif result[i][1][0:2] == "Lo" : sentence += result[i][1][:-20]
        elif result[i][1][-2:] == "()" : sentence += result[i][1][:-2] + " - gini"
        else : sentence += result[i][1][:-21] + " - entropy"

        sentence += " & "

        sentence += "K-fold :" + str(result[i][2]) +"\n"
        sentence += "Best Parameter :" + str(result[i][3])+"\n"
        sentence += "Average Score :" + str(result[i][4])+"\n"

        print(sentence)


#Data definition
data = np.loadtxt("breast-cancer-wisconsin.data", delimiter=",", dtype=str)
column = ["Sample code number",
          "Clump Thickness",
          "Uniformity of Cell Size",
          "Uniformity of Cell Shape",
          "Marginal Adhesion",
          "Single Epithelial Cell Size",
          "Bare Nuclei",
          "Bland Chromatin",
          "Normal Nucleoli",
          "Mitoses","Class"]
df = pd.DataFrame(data,columns = column)

#Print data information
print(df.info())

#Handling Missing Values
#This data shows missing values as ?. 
#And since there are quite a few records, we dropped the missing values.
df.drop(df[df["Bare Nuclei"]=='?'].index,inplace=True)
df.reset_index(drop=True,inplace=True)

#Change type for memory savings and scaling
df = df.astype('int')

#Visualization of heat map for checking correlation coefficients
corr= df.corr()
sns.heatmap(corr, annot = True, cmap = 'RdYlBu_r',vmin = -1, vmax = 1)
plt.show()

#We determined that the valid feature was a feature with a correlation coefficient of 0.7 or more.
#So the set threshold was 0.7, and all the features below it were droped.
#It then prints the information of the changed data.
index = corr[corr["Class"]<0.7].index
df.drop(index,inplace=True,axis=1)
print(df.info())

#Parameters for automation functions
listOfScaler = [StandardScaler(), 
                MinMaxScaler(), 
                RobustScaler()]

listOfModel = [DecisionTreeClassifier(criterion='gini'), 
               DecisionTreeClassifier(criterion='entropy'), 
               LogisticRegression(solver='liblinear'), 
               SVC()]

listOfKFold = [5,7,10]

listOfParam = [{'max_depth':[2,3,5,7,10], 'min_samples_split':[2,3,5]},
               {'max_depth':[2,3,5,7,10], 'min_samples_split':[2,3,5]},
               {'C': [0.001, 0.01, 0.1, 1, 10, 100],'penalty': ['l1', 'l2']},
               {'C': [0.001, 0.01, 0.1, 1, 10, 100],'gamma': [0.001, 0.01, 0.1]}]

#Function call
result = finder(df, listOfScaler, listOfModel, listOfKFold, listOfParam)
print_result(result,False)