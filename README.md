# Kaggle_titanic

import numpy as np
import pandas as pd
import random as rnd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns',1000)

train=pd.read_csv('~/downloads/titanic/train.csv')
test=pd.read_csv('~/downloads/titanic/test.csv')

#========================================clean_data()===============================================
train_Y=train['Survived']
train_X=train
train_X.drop(labels='Survived',axis=1,inplace=True)
Merged= train_X.append(test)

##Creat new feature "Title"
Merged['Title'] = Merged['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

def Group_Title():
    Title_Dictionary={
        "Capt":"Rare",
        "Col":"Rare",
        "Don":"Rare",
        "Dona":"Rare",
        "Dr":"Rare",
        "Jonkheer":"Rare",
        "Lady":"Rare",
        "Major":"Rare",
        "Master":"Master",
        "Miss":"Miss",
        "Mlle":"Rare",
        "Mme":"Rare",
        "Mr":"Mr",
        "Mrs":"Mrs",
        "Ms":"Rare",
        "Rev":"Rare",
        "Sir":"Rare",
        "the Countess":"Rare"
    }
    Merged['Title']=Merged['Title'].map(Title_Dictionary)

Group_Title()

Merged['FamilySize']=Merged['SibSp']+Merged['Parch']+1

print(Merged[Merged['Embarked'].isnull()],'\n\n\n\n')
print(Merged[['Pclass','Fare','Embarked','FamilySize']].groupby(['Embarked','Pclass','FamilySize'],as_index=True).mean(),'\n\n\n')
Merged['Embarked'].fillna('C',inplace=True)
print(Merged.info(),'\n\n\n')

print(Merged[Merged['Fare'].isnull()],'\n\n\n')
Merged['Fare'].fillna(9.6,inplace=True)
print(Merged.info(),'\n\n')

print(Merged[Merged['Age'].isnull()][['Title','Fare']].groupby(['Title']).count())
Merged[['Title','Age']].groupby(['Title'])
grid = sns.FacetGrid(Merged, col='Title', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
plt.show()

MedMaster = np.nanmedian(Merged[Merged['Title']=="Master"]['Age'])
MedMiss = np.nanmedian(Merged[Merged['Title']=="Miss"]['Age'])
MedMr = np.nanmedian(Merged[Merged['Title']=="Mr"]['Age'])
MedMrs = np.nanmedian(Merged[Merged['Title']=="Mrs"]['Age'])
MedRare = np.nanmedian(Merged[Merged['Title']=="Rare"]['Age'])


print(Merged.loc[Merged['Age'].isnull()],'\n\n') 
print(Merged.loc[(Merged['Age'].isnull())&(Merged['Title']=="Master")],'\n\n')
print(Merged.loc[(Merged['Age'].isnull())&(Merged['Title']=="Master"),'Age'])

Merged.loc[(Merged['Age'].isnull())&(Merged['Title']=='Master'),'Age']=MedMaster
Merged.loc[(Merged['Age'].isnull())&(Merged['Title']=='Miss'),'Age']=MedMiss
Merged.loc[(Merged['Age'].isnull())&(Merged['Title']=='Mr'),'Age']=MedMr
Merged.loc[(Merged['Age'].isnull())&(Merged['Title']=='Mrs'),'Age']=MedMrs
Merged.loc[(Merged['Age'].isnull())&(Merged['Title']=='Rare'),'Age']=MedRare

print(Merged.info())
print(Merged.nunique(),'\n\n')

label_encoder=LabelEncoder()

Merged.loc[:,'Sex'] = label_encoder.fit_transform(Merged.loc[:,'Sex'])
Merged.loc[:,'Title']=label_encoder.fit_transform(Merged.loc[:,'Title'])
Merged.loc[:,'Embarked']=label_encoder.fit_transform(Merged.loc[:,'Embarked'])
print(Merged.head(10))
print(Merged.isnull().sum(),'\n\n')

Merged.drop(['Name','Ticket','Cabin','Parch','SibSp'],axis=1,inplace=True)
print(Merged.info())
Train_new = Merged[0:891]
Test_new=Merged[891:Merged.shape[0]]


#========================================train_test_split()===============================================
X_train,X_test,Y_train,Y_test=train_test_split(Train_new,train_Y,test_size=0.2,random_state=0)


##Model using Logistic Regression
Model=LogisticRegression()
Model.fit(X_train,Y_train)
Y_pred=Model.predict(X_test)
print(round(Model.score(X_train, Y_train) * 100,2),round(Model.score(X_test, Y_test)*100,2))
test=test.drop("PassengerId",axis=1).copy()
prediction=Model.predict(Test_new)

print(Test_new)



submission = pd.DataFrame({
    "PassengerId": Test_new["PassengerId"],
    "Survived": prediction
})
submission.to_csv('submission.csv',index=False)


