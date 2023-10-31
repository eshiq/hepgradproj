import inline as inline
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

df=pd.read_csv('/users/eshiqabdulaziz/Desktop/مشروع التخرج/HepatitisCdatacleanedd.csv')
print(df)

#print(x.info())
df.loc[614,"ALP"] = 49.7
df.loc[613,"ALP"] = 69.6
df.loc[603,"ALB"] = 44
df.loc[603,"ALP"] = 120.9
df.loc[603,"CHOL"] = 5.5
df.loc[592,"ALP"] = 66
df.loc[590,"ALP"] = 85.3
df.loc[590,"CHOL"] = 5.34
df.loc[590,"PROT"] = 71.8
df.loc[585,"ALP"] = 66
df.loc[584,"ALP"] = 43.1
df.loc[584,"CHOL"] = 4.2
df.loc[583,"ALP"] = 43.1
df.loc[582,"ALP"] = 35.7
df.loc[581,"ALP"] = 35.7
df.loc[576,"ALP"] = 39.8
df.loc[571,"ALP"] = 29.7
df.loc[570,"ALP"] = 41.8
df.loc[569,"ALP"] = 29.7
df.loc[568,"ALP"] = 43.1
df.loc[546,"ALP"] = 32.9
df.loc[545, "ALP"] = 27.3
df.loc[541,"ALP"] = 20.6
df.loc[540,"ALT"]= 10.5
df.loc[498,"CHOL"]= 5.1
df.loc[433,"CHOL"]= 5.84
df.loc[424,"CHOL"] = 4.05
df.loc[413,"CHOL"] = 7.09
df.loc[329,"CHOL"]=5.07
df.loc[319,"CHOL"]=5.57
df.loc[121,"CHOL"] = 7.51


#الكورليشن ضابط
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
print(df.corr())
#graph بعد الفيتشر ريدكشن
df.plot()
plt.show()


#Importing Libraries and Classes
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

#X is the input and Y is the Output
x =df[['ALB','ALP','ALT','AST','BIL',
      'CHE','CHOL','CREA','GGT','PROT']].values
y =df[["Category"]].values


"splitting the dataset into the training set and test set 1.0"
# Importing Libraries and Classes
# Divie the data into 70 and 30 % for traning and testing purpose
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.3 , random_state=0)

#Train the model
model.fit(x_train, y_train)
#Training accurcy
tr_ac_LR=model.score(x_train , y_train)
print(tr_ac_LR)



#Testing Accuracy
ts_ac_LR=model.score(x_test , y_test)
print(ts_ac_LR)
# y contains all the outputs and x contains all the input.
# I am testing if the model gives the expected output for the corresponding imput.
expected =y_test
predicted = model.predict(x_test)
#Importing Libraries and Class
from sklearn import metrics
print(metrics.classification_report(expected , predicted))
#out of 51, Yes 'Y' outcomes 24 were right and 27 were wrong
#Out of 134 outcomes for No, 'N', 131 were right and  3 were worng
print(metrics.confusion_matrix(expected , predicted))



#Support Vector Machine Algorithm
#Importing class and Libraries
from sklearn.svm import SVC
model_svc=SVC()
#Train the model
model_svc.fit(x_train,y_train)
#Accuracy of the model in training
tr_ac_SVC=model_svc.score(x_train,y_train)
print(tr_ac_SVC)
#Accuracy of the model in Testing
ts_sc_SVC=model_svc.score(x_test,y_test)
print(ts_sc_SVC)
#Import Libraries and Classes
from sklearn import metrics
rep_svc=print(metrics.classification_report(expected,predicted))
fin_svc=print(metrics.confusion_matrix(expected,predicted))


#Decision Tree Algorithm
#Import Libraries and Classes
from sklearn.tree import DecisionTreeClassifier
model_dt=DecisionTreeClassifier()
#Train the model
model_dt.fit(x_train,y_train)
#Train Accuracy of DT Model
tr_ac_dt=model_dt.score(x_train,y_train)
print(tr_ac_dt)
#Testing Accuracy of DT model
ts_sc_dt=model_dt.score(x_test,y_test)
print(ts_sc_dt)
expected_dt=y_test
predicted_dt=model_dt.predict(x_test)
#Generating Report
print(metrics.classification_report(expected_dt,predicted_dt))
#Output in the form of Matrix
print(metrics.confusion_matrix(expected_dt,predicted_dt))


filename = 'hep_pred_model'
pickle.dump(model, open(filename, 'wb'))
loaded_model =pickle.load(open(filename, 'rb'))
loaded_model.predict(x_test)