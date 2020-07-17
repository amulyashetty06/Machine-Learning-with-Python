'''MACHINE LEARNING MODEL ON LIVER DISEASE PREDICTION'''

#datacollection
#importlibrariesanddataframe
import pandas as pd
data=pd.read_csv('liver.csv')

#datainterpretation
data.info()
print(data.describe())
print(data['Dataset'].unique())

#datacleaning
data.drop('SrNo',inplace=True,axis=1)
data.drop('Gender',inplace=True,axis=1)
classes={'Positive':1, 'Negative':0}
data.replace({'Dataset':classes},inplace=True)
#classes1={'Male':1, 'Female':0}
#data.replace({'Gender':classes1},inplace=True)
data=data.dropna()

#createarrays
#x:all independent data
#y:Outcome(depenedent data)
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

#splituniversaldataset(train:test)
#library:sklearn
#module:model_selection
#classtrain_test_split
from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=38)

#algorithmselection
#logisticregression
#library:sklearn
#module:linear_model
#class:LogisticRegression
from sklearn.linear_model import LogisticRegression as logreg
model_logreg=logreg()

#trainthemodel
model_logreg.fit(x_train,y_train)

#Testthemodel
#predictingoutput
y_pred=model_logreg.predict(x_test)

#Checkingaccuracy
accuracy=model_logreg.score(x_test,y_test)
print('Logistic regression accuracy:',accuracy)

#confusionmatrix
#library:sklearn
#module:metrics
#class:confusion_matrix
from sklearn.metrics import confusion_matrix as conmat
cm=conmat(y_test,y_pred)

#Visualization

#pairplot
import seaborn as sb
sb.pairplot(data,hue='Dataset')

#countplot
sb.countplot(x='Dataset',data=data)

#Box-whiskerplot
data.plot(kind='box',subplots=True,layout=(3,5),figsize=(15,15))

#heatmap
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))  
sb.heatmap(data.corr(), annot=True, fmt='.2')





