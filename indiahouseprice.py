'''MACHINE LEARNING MODEL ON INDIA HOUSE PRICING'''
#datacollection
#importlibrariesanddataframe
import pandas as pd
data=pd.read_csv('indiahouse.csv')

#datainterpretation
data.info()
print(data.describe())

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
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=3)

#algorithmselection
#linearregression
#library:sklearn
#module:linear_model
#class:LinearRegression
from sklearn.linear_model import LinearRegression as linreg
model_linreg=linreg()

#trainthemodel
model_linreg.fit(x_train,y_train)

#Testthemodel
#predictingoutput
y_pred=model_linreg.predict(x_test)

#Checkingaccuracy
accuracy=model_linreg.score(x_test,y_test)
print('Linear regression accuracy:',accuracy)

#Visualzsation
#heatmap
import seaborn as sb
import matplotlib.pyplot as plt
plt.figure(figsize=(20,20))  
sb.heatmap(data.corr(), annot=True, fmt='.2')




