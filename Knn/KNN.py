#For running this you need to download iris.csv
#To produce graph run in jupyter notebook
#import for iris data set
from sklearn.datasets import load_iris
#KneighboursClassifier
from sklearn.neighbors import KNeighborsClassifier
#Import numpy
import numpy as np
# model for spliting the dataset
from sklearn.model_selection import train_test_split
#This is imported to plot data
import matplotlib.pyplot as plt
import pandas as pd

#loading data
iris_dataset=load_iris()

iris=pd.read_csv('iris.csv',names=['sepal_length','sepal_width','petal_length','petal_width','class'])
print(iris.head(150))

#Plot data
colors={'Setosa':'r','Versicolor':'g','Virginica':'b'}
createFigure,createAxis=plt.subplots()
for i in range(len(iris['sepal_length'])):
    createAxis.scatter(iris['sepal_length'][i],iris['sepal_width'][i],color=colors[iris['class'][i]])
createAxis.set_title('Dataset Iris')
createAxis.set_xlabel('sepal_length')
createAxis.set_ylabel('sepal_width')


#spliting the data
X_train,X_test,y_train,y_test=train_test_split(iris_dataset["data"],iris_dataset["target"],random_state=0)
#KNN where n is set to 1
kn=KNeighborsClassifier(n_neighbors=1)
#train the model
kn.fit(X_train,y_train)
#dimention of new flower
x_new=np.array([[5,2.9,4,0.2]])
#for predicting the type of flower
prediction=kn.predict(x_new)

print("Predicted target value:{}\n".format(prediction))
print("Predicted feature name:{}\n".format(iris_dataset["target_names"][prediction]))
#to test performance
print("Test score:{:.2f}".format(kn.score(X_test,y_test)))


