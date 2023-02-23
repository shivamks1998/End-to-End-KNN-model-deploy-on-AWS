import pandas as pd
from sklearn.datasets import load_iris
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import Normalizer


class datasets:
    def __init__(self,dataset,model):
        self.dataset = dataset
        self.model  = model


    def data_split(self):
        X = self.dataset.iloc[:,:4]
        y = self.dataset.iloc[:,4]
       #split the data into train and test
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=2018)
        return X_train,X_test,y_train, y_test


    def normaize_data(self,x_train,x_test):
        scaler= Normalizer().fit(x_train) # the scaler is fitted to the training set
        x_train= scaler.transform(x_train) # the scaler is applied to the training set
        x_test= scaler.transform(x_test)
        return x_train, x_test
  

    def model_train(self,y_train,x_train):
        model_fit = self.model.fit(x_train,y_train)
        with open(".\\model.pkl",'wb') as fil:
            saved_model = pickle.dump(model_fit,fil)
        return saved_model


    def inference(self,y_test,x_test):
        load = open(".\\model.pkl",'rb')
        load_model = pickle.load(load)
        prediction = load_model.predict(x_test)
        eval  = confusion_matrix(y_test,prediction)
        acc = accuracy_score(y_test,prediction)
        return acc,eval


def main():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    model = KNN(n_neighbors=3)
    trained_model = datasets(model=model,dataset=df)
    X_train,X_test,y_train, y_test = trained_model.data_split()
    x_train,x_test = trained_model.normaize_data(X_train,X_test)
    saved_model = trained_model.model_train(x_train = x_train,y_train = y_train)
    acc,matrix  = trained_model.inference(x_test=x_test,y_test=y_test)
    return acc,matrix
    
    
if __name__=="__main__":
    print(main())

        