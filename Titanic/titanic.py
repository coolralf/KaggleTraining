#%%
import pandas as pd
import numpy as np
import os
from pandas import Series,DataFrame

path = os.getcwd()
data_train = pd.read_csv(r"Titanic/train.csv")
print(data_train)
#data_info...

from sklearn.ensemble import RandomForestRegressor

def set_missing_ages(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]

    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    y = known_age[:,0]
    X = known_age[:,1:]

    rfr = RandomForestRegressor(random_state=0,n_estimators=2000,n_jobs=-1)
    rfr.fit(X,y)

    predictedAges = rfr.predict(unknown_age[:,1::])

    df.loc[(df.Age.isnull()),'Age'] = predictedAges

    return df,rfr

def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'
    return df

data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)


dummies_Cabin = pd.get_dummies(data_train["Cabin"],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'],prefix='Pclass')
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis = 1)
df.drop(['Cabin','Pclass','Sex','Name','Ticket','Embarked'],axis = 1,inplace=True)


#scaling 

import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled']= scaler.fit_transform(df['Fare'],fare_scale_param)

#modelling

from sklearn import linear_model

train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|')
train_np = train_df.as_matrix()

y = train_np[:,0]

X = train_np[:,1:]

clf = linear_model.LogisticRegression(C=1.0,penalty='l2',tol=1e-4)
clf.fit(X,y)

#processing test_data

data_test= pd.read_csv("test.csv")
data_test.loc[(data_test.Fare.isnull()),'Fare']= 0

tmp_df = data_test[['Age','Fare','Parch','SibSp','Pclass']]
null_age = tmp_df[data_test.Age.isnull()].as_matrix()

X = null_age[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test["Cabin"],prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'],prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'],prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'],prefix='Pclass')

df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex],axis = 1)
df_test.drop(['Cabin','Pclass','Sex','Name','Ticket','Embarked'],axis = 1,inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'],age_scale_param)
df_test['Fare_scaled']= scaler.fit_transform(df_test['Fare'],fare_scale_param)

test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|')
predictions = clf.predict(test)

result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survied':predictions.astype(np.int32)})
result.to_csv('result.csv')