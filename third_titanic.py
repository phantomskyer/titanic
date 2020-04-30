import pandas as pd 
import numpy as np 
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model


data_train = pd.read_csv("E:titanic_data/train.csv")
data_test = pd.read_csv("E:titanic_data/test.csv")


#数据处理函数
def class_with_name(df):
    df.loc[(df.Pclass == 1)&(df['Name'].str.contains('Mr')), 'c1Mr'] = 1
    df.loc[(df.Pclass != 1)|(df['Name'].str.contains('Mr') == 0), 'c1Mr'] = 0
    df.loc[(df.Pclass == 2)&(df['Name'].str.contains('Mr')), 'c2Mr'] = 1
    df.loc[(df.Pclass != 2)|(df['Name'].str.contains('Mr') == 0), 'c2Mr'] = 0
    df.loc[(df.Pclass == 3)&(df['Name'].str.contains('Mr')), 'c3Mr'] = 1
    df.loc[(df.Pclass != 3)|(df['Name'].str.contains('Mr') == 0), 'c3Mr'] = 0
    df.loc[(df.Pclass == 1)&(df['Name'].str.contains('Mrs')), 'c1Mrs'] = 1
    df.loc[(df.Pclass != 1)|(df['Name'].str.contains('Mrs') == 0), 'c1Mrs'] = 0
    df.loc[(df.Pclass == 2)&(df['Name'].str.contains('Mrs')), 'c2Mrs'] = 1
    df.loc[(df.Pclass != 2)|(df['Name'].str.contains('Mrs') == 0), 'c2Mrs'] = 0
    df.loc[(df.Pclass == 3)&(df['Name'].str.contains('Mrs')), 'c3Mrs'] = 1
    df.loc[(df.Pclass != 3)|(df['Name'].str.contains('Msr') == 0), 'c3Mrs'] = 0
    df.loc[(df.Pclass == 1)&(df['Name'].str.contains('Miss')), 'c1Miss'] = 1
    df.loc[(df.Pclass != 1)|(df['Name'].str.contains('Miss') == 0), 'c1Miss'] = 0
    df.loc[(df.Pclass == 2)&(df['Name'].str.contains('Miss')), 'c2Miss'] = 1
    df.loc[(df.Pclass != 2)|(df['Name'].str.contains('Miss') == 0), 'c2Miss'] = 0
    df.loc[(df.Pclass == 3)&(df['Name'].str.contains('Miss')), 'c3Miss'] = 1
    df.loc[(df.Pclass != 3)|(df['Name'].str.contains('Miss') == 0), 'c3Miss'] = 0
    return df

def sex_with_pclass(df):
    df.loc[(df['Sex'] == "male") & (df['Pclass'] == 1), 'c1man'] = 1
    df.loc[(df['Sex'] != "male") | (df['Pclass'] != 1), 'c1man'] = 0
    df.loc[(df['Sex'] == "male") & (df['Pclass'] == 2), 'c2man'] = 1
    df.loc[(df['Sex'] != "male") | (df['Pclass'] != 2), 'c2man'] = 0
    df.loc[(df['Sex'] == "male") & (df['Pclass'] == 3), 'c3man'] = 1
    df.loc[(df['Sex'] != "male") | (df['Pclass'] != 3), 'c3man'] = 0
    df.loc[(df['Sex'] == "female") & (df['Pclass'] == 1), 'c1woman'] = 1
    df.loc[(df['Sex'] != "female") | (df['Pclass'] != 1), 'c1woman'] = 0
    df.loc[(df['Sex'] == "female") & (df['Pclass'] == 2), 'c2woman'] = 1
    df.loc[(df['Sex'] != "female") | (df['Pclass'] != 2), 'c2woman'] = 0
    df.loc[(df['Sex'] == "female") & (df['Pclass'] == 3), 'c3woman'] = 1
    df.loc[(df['Sex'] != "female") | (df['Pclass'] != 3), 'c3woman'] = 0
    df.drop(['Pclass','Sex'], axis=1, inplace=True)
    return df
    
def mother_or_not(df):
    df.loc[(df['Name'].str.contains('Mrs')) & (df['Parch'] > 1), 'mother'] = 1
    df.loc[(df['Parch'] <= 1)|(df['Name'].str.contains('Mrs') == 0), 'mother'] = 0
    return df
    
def age_train(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'c1Mr','c2Mr','c3Mr','c1Mrs','c2Mrs','c3Mrs','c1Miss','c2Miss','c3Miss']]
    known_age = age_df[age_df.Age.notnull()].values
    unknown_age = age_df[age_df.Age.isnull()].values
    y = known_age[:, 0]
    X = known_age[:, 1:]
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::])
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    return df,rfr
    
def age_test(data_test,rfr):
    data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
    tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'c1Mr','c2Mr','c3Mr','c1Mrs','c2Mrs','c3Mrs','c1Miss','c2Miss','c3Miss']]
    null_age = tmp_df[data_test.Age.isnull()].values
    X = null_age[:, 1:]
    predictedAges = rfr.predict(X)
    data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
    return data_test
    
def child_or_not(df):
    df.loc[(df['Age'] <= 12), 'child'] = 1
    df.loc[(df['Age'] > 12), 'child'] = 0
    return df

def mul_cabin_or_not(df):
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    df.loc[(df['Cabin'].str.contains(' ')), 'mul_cabin'] = 1
    df.loc[(df['Cabin'].str.contains(' ') == 0), 'mul_cabin'] = 0
    return df
    
#用分组替代归一化
def group_age_fare(df):
    df.loc[ df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[ df['Age'] > 64, 'Age']
    
    df.loc[ df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare']   = 2
    df.loc[ df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

    return df
    
def scale_age_fare(df):
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(np.array(df['Age']).reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(np.array(df['Age']).reshape(-1, 1), age_scale_param)
    fare_scale_param = scaler.fit(np.array(df['Fare']).reshape(-1, 1))
    df['Fare_scaled'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1, 1), fare_scale_param)
    return df
    
def mul_ticket(data_train):
    df = data_train['Ticket'].value_counts()
    df = pd.DataFrame(df)
    df = df[df['Ticket'] > 1]
    df_ticket = df.index.values        #共享船票的票号
    tickets = data_train.Ticket.values    #所有的船票
    result = []
    for ticket in tickets:
        if ticket in df_ticket:
            ticket = 1
        else:
            ticket = 0                 #遍历所有船票，在共享船票里面的为1，否则为0
        result.append(ticket)
    results = pd.DataFrame(result)
    results.columns = ['Ticket_Count']
    data_train = pd.concat([data_train, results], axis=1)
    return data_train


def data_initialization(df):

    df = class_with_name(df)
    
    df = sex_with_pclass(df)

    df = mother_or_not(df)

    df,rfr = age_train(df)

    df = child_or_not(df)
    
    df = mul_cabin_or_not(df)
    
    df = mul_ticket(df)
   
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix= 'Embarked')
    df = pd.concat([df, dummies_Embarked], axis=1)
    df.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    
    df = group_age_fare(df)
    
    return df,rfr

def data_initialization_test(data_test,rfr):
    data_test = class_with_name(data_test)
    
    data_test = sex_with_pclass(data_test)

    data_test = mother_or_not(data_test)

    #data_test,rfr = age_train(data_test)
    
    data_test = age_test(data_test,rfr)

    data_test = child_or_not(data_test)
    
    data_test = mul_cabin_or_not(data_test)
    
    data_test = mul_ticket(data_test)
   
    dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
    data_test = pd.concat([data_test, dummies_Embarked], axis=1)
    data_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
    
    data_test = group_age_fare(data_test)
    
    return data_test


train_data,rfr = data_initialization(data_train)
train_data.to_csv("E:titanic_data/deal_data.csv", index=False)
data_test = data_initialization_test(data_test,rfr)
train_data.to_csv("E:titanic_data/deal_data_test.csv", index=False)

train_df = train_data.filter(regex='Survived|Age*|SibSp|Parch|Fare*|Cabin_.*|Ticket_Count|Embarked_.*|c1Mr|c2Mr|c3Mr|c1Mrs|c2Mrs|c3Mrs|c1Miss|c2Miss|c3Miss|c1man|c2man|c3man|c1woman|c2woman|c3woman|mother|child|mul_cabin')
train_np = train_df.values
# y即Survival结果
y = train_np[:, 0]
# X即特征属性值
X = train_np[:, 1:]
# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
clf.fit(X, y)

test = data_test.filter(regex='Survived|Age*|SibSp|Parch|Fare*|Cabin_.*|Ticket_Count|Embarked_.*|c1Mr|c2Mr|c3Mr|c1Mrs|c2Mrs|c3Mrs|c1Miss|c2Miss|c3Miss|c1man|c2man|c3man|c1woman|c2woman|c3woman|mother|child|mul_cabin')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("E:titanic_data/third_logistic_regression_predictions.csv", index=False)
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})