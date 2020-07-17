import pandas as pd
data=pd.read_csv('liver.csv')

data.drop('SrNo',inplace=True,axis=1)
data.drop('Gender',inplace=True,axis=1)
data=data.dropna()

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

classes={'Positive':1, 'Negative':0}
data.replace({'Dataset':classes},inplace=True)

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.3,random_state=21)

#Library:sklearn
#Module:neighbors
#class:KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
knnmodel=knn(n_neighbors=5)
knnmodel.fit(x_train,y_train)
accuracy=knnmodel.score(x_test,y_test)
predicted=knnmodel.predict(x_test)

k_range=range(1,96)
k_score=[]
k_best=1
acc_best=0

for k in k_range:
    knnmodel=knn(n_neighbors=k)
    
knnmodel.fit(x_train,y_train)
k_acc=knnmodel.score(x_test,y_test)
k_score.append(k_acc)
if(acc_best<k_acc):
    acc_best=k_acc
    k_best=k


