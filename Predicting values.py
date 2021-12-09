#  We have dataset about different transactions and their parameters. Our aim is to make a model that can predict suspicious transactions that may lead to a fraud. In this dataset the target value is "Class". If column "Class" equals 1,
# the transaction is fraud. If column "Class" equals 0, the transaction is non-fraud.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob

path = r'./data'
filenames = glob.glob(path + "/*.zip")

dfs = list()
for filename in filenames:
    df = pd.read_csv(filename, compression='zip')    
    dfs.append(df)
data = pd.concat(dfs, axis=0, ignore_index=True)


##data=pd.read_csv('creditcard.csv')
print(data.head())
print(data.info())

# Visualizing the number of each type of deals.
sns.barplot(x = 'Class', y = 'Amount', data = data,
            palette = 'hls',  
            capsize = 0.05,  
            estimator = np.sum,
            errcolor = 'grey', errwidth = 2)
plt.show()


# Choosing parameters for a prediction model using a correlation matrix.
correlation_matrix = data.corr(method ='pearson').round(2)
sns.heatmap(data=correlation_matrix, annot=True, cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(15,15)
plt.show()

columns_for_model=['V17', 'V14', 'V12', 'V10', 'V7', 'V3', 'V4', 'V11', 'V5', 'V18', 'V2', 'V16']


# Assigning X and y, spliting data for train and test group.      
X_1=data.loc[:, columns_for_model]
y=data.iloc[:,30]


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import sklearn.pipeline as pl
import sklearn.preprocessing as sp
import sklearn.linear_model as lm
import sklearn.metrics as sm
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test=train_test_split(X_1,y, test_size=0.2, random_state=0)


# Creating and traing the model
from sklearn.linear_model import LogisticRegression
classifier = pl.make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs',random_state=0))
classifier.fit(X_train, y_train)
y_pred=classifier.predict(X_test)

# Evaluating the model
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
conf_matrix_baseline_1 = pd.DataFrame(confusion_matrix(y_test, y_pred), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])

print(conf_matrix_baseline_1)
print('Logistic regression recall score is', recall_score(y_test, y_pred))

print("The recall score is not so high. We are going to try to get better results by using other type of model")

  
# Creating a new model using Random Forest algorythm. Please pay attention that applying Random Forest algorytm takes time.
X_2 = data.iloc[:,:30]

X_train_2, X_test_2, y_train_2, y_test_2=train_test_split(X_2,y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier_2 = pl.make_pipeline(StandardScaler(), RandomForestClassifier())
classifier_2.fit(X_train_2, y_train_2)
y_pred_2=classifier_2.predict(X_test_2)

# Evaluating the second model
conf_matrix_baseline_2 = pd.DataFrame(confusion_matrix(y_test_2, y_pred_2), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])
print(conf_matrix_baseline_2)
print('Random forest model recall score is', recall_score(y_test_2, y_pred_2))
print("We have a lot of parametres for the second model. We can try to eliminate their number")

# Choosing the best parametres for the second model
feats = {}
for feature, importance in zip(data.columns, classifier_2.steps[1][1].feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})
sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
fig, ax = plt.subplots()
fig.set_size_inches(30,15)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Importance', fontsize=25, weight = 'bold')
plt.ylabel('Features', fontsize=25, weight = 'bold')
plt.title('Feature Importance', fontsize=25, weight = 'bold')
plt.show()


columns_for_model_2=['V12', 'V17', 'V14', 'V11', 'V10', 'V16', 'V18', 'V9', 'V4', 'V7', 'V26']
X_3=data.loc[:, columns_for_model_2]


# Creating a new model using Random Forest algorythm and limited number of parametres
X_train_3, X_test_3, y_train_3, y_test_3=train_test_split(X_3,y, test_size=0.2, random_state=0)
classifier_2.fit(X_train_3, y_train_3)
y_pred_3=classifier_2.predict(X_test_3)

# Evaluating the second model with limited parametres
conf_matrix_baseline_3 = pd.DataFrame(confusion_matrix(y_test_3, y_pred_3), index = ['actual 0', 'actual 1'], columns = ['predicted 0', 'predicted 1'])
print(conf_matrix_baseline_3)
print('Random forest model recall score is', recall_score(y_test_3, y_pred_3))

print("We can see that Random Forest model with limited number of parameters has also high performance as the model with all parameters. In comparison with Logistic regression it has also better power of detection")



