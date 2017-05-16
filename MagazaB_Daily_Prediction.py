import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression, BayesianRidge, OrthogonalMatchingPursuitCV, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import dateutil




#####DAİLY###############
dataset = pd.read_csv('C:/Users/User/Desktop/daily.csv', sep=';')
df = pd.DataFrame(dataset)
df['TARIH'] = df['TARIH'].apply(dateutil.parser.parse, dayfirst=True)
df['TARIH'] = pd.to_datetime(df['TARIH'])
df['month'] = df['TARIH'].dt.month
df['day'] = df['TARIH'].dt.day
df['day_of_week'] = df['TARIH'].dt.dayofweek
days = {0:'0',1:'0',2:'0',3:'0',4:'0',5:'1',6:'1'}
df['day_of_week'] = df['day_of_week'].apply(lambda x: days[x])

datasets = pd.read_csv('C:/Users/User/Desktop/dailyTest.csv', sep=';')
test = pd.DataFrame(datasets)
test['TARIH'] = test['TARIH'].apply(dateutil.parser.parse, dayfirst=True)
test['TARIH'] = pd.to_datetime(test['TARIH'])
test['month'] = test['TARIH'].dt.month
test['day'] = test['TARIH'].dt.day
test['day_of_week'] = test['TARIH'].dt.dayofweek
days = {0:'0',1:'0',2:'0',3:'0',4:'0',5:'1',6:'1'}
test['day_of_week'] = test['day_of_week'].apply(lambda x: days[x])
index = test['TARIH']


####### TRAİN SETS################
trainingSet = np.array(df['CIRO'])
y = np.transpose(np.matrix(trainingSet))
x1 = ['euro','dolar','temperature','special days','day_of_week']
trainingSetLabels =np.array(df[x1])


######## TEST SETS ################
testSet = np.array(test['CIRO'])
w = np.transpose(np.matrix(testSet))
x2 = ['euro','dolar','temperature','special days','day_of_week']
testSetLabels = np.array(test[x2])



###########################################################################33
##########################  MAPE     #######################################

def mean_absolute_percentage_error(y_pred, y_true):
    idx = y_true != 0.0
    return np.mean(np.abs((y_pred[idx] - y_true[idx]) / y_true[idx])) * 100



#################### PREDICTION ##############################################

########### LINEAR REGRESSION #################

classifier = LinearRegression()
predictionModel = classifier.fit(trainingSetLabels,y)
p = predictionModel.predict(testSetLabels)

mse = mean_squared_error(testSet,p)
mae = mean_absolute_error(testSet,p)
R2 = r2_score(testSet,p,multioutput= 'variance_weighted')
mape = mean_absolute_percentage_error(p, testSet)

print('mean squared error:\n',mse)
print('mean absolute error:\n',mae)
print('R2-Score:\n', R2)
print('MAPE:\n', mape)


index = test['TARIH']
plt.plot(index,testSet,label='Actual Values', linestyle = '-',color = 'blue')
plt.plot(index,p,label='predict Values', linestyle = '-',color = 'red')
plt.title("LinearRegression")
plt.xlabel("Date")
plt.ylabel("GIRO")
plt.show()


######### DECİSİON TREE #############
model = DecisionTreeRegressor()
decisionPredict = model.fit(trainingSetLabels ,y)
dp = decisionPredict.predict(testSetLabels)

mse = mean_squared_error(testSet,dp)
mae = mean_absolute_error(testSet,dp)
R2 = r2_score(testSet, dp,multioutput= 'variance_weighted')
mape = mean_absolute_percentage_error(dp, testSet)

print('mean squared error:\n', mse)
print('mean absolute error:\n', mae)
print('R2-Score:\n', R2)
print('MAPE:\n', mape)

index = test['TARIH']
plt.plot(index,testSet,label='Actual Values', linestyle = '-',color = 'blue')
plt.plot(index,dp,label='predict Values', linestyle = '-',color = 'red')
plt.title("DecisionTree")
plt.xlabel("Date")
plt.ylabel("GIRO")
plt.show()

############ BAYESIAN RİDGE #####################
clf = BayesianRidge(compute_score=True)
predictionModel = clf.fit(trainingSetLabels , y)
p = predictionModel.predict(testSetLabels)

mse = mean_squared_error(testSet, p)
mae = mean_absolute_error(testSet, p)
R2 = r2_score(testSet,p, sample_weight=None, multioutput= 'variance_weighted')
mape = mean_absolute_percentage_error(p, testSet)

print('mean squared error:\n', mse)
print('mean absolute error:\n', mae)
print('R2-Score:\n', R2)
print('MAPE:\n', mape)

index = test['TARIH']
plt.plot(index,testSet,label='Actual Values', linestyle = '-',color = 'blue')
plt.plot(index,p,label='predict Values', linestyle = '-',color = 'red')
plt.title("BayesianRidge")
plt.xlabel("Date")
plt.ylabel("GIRO")
plt.show()



##################3#Logistic Regression #########################
clf = LogisticRegression()
predictionModel = clf.fit(trainingSetLabels,y)
p = predictionModel.predict(testSetLabels)

mse = mean_squared_error(testSet, p)
mae = mean_absolute_error(testSet, p)
R2 = r2_score(testSet, p, multioutput= 'variance_weighted')
mape = mean_absolute_percentage_error(p, testSet)

print('mean squared error:\n', mse)
print('mean absolute error:\n', mae)
print('R2-Score:\n', R2)
print('MAPE:\n', mape)

index = test['TARIH']
plt.plot(index,testSet,label='Actual Values', linestyle = '-',color = 'blue')
plt.plot(index,p,label='predict Values', linestyle = '-',color = 'red')
plt.title("LogisticRegression")
plt.xlabel("Date")
plt.ylabel("GIRO")
plt.show()


###################3OrthogonalMatchingPursuitCV
 omp_cv = OrthogonalMatchingPursuitCV()
omp = omp_cv.fit(trainingSetLabels,y)
o = omp.predict(testSetLabels)

mse = mean_squared_error(testSet, o)
mae = mean_absolute_error(testSet, o)
R2 = r2_score(testSet, o, multioutput= 'variance_weighted')
mape = mean_absolute_percentage_error(o, testSet)

print('mean squared error:\n', mse)
print('mean absolute error:\n', mae)
print('R2-Score:\n', R2)
print('MAPE:\n', mape)

index = test['TARIH']
plt.plot(index,testSet,label='Actual Values', linestyle = '-',color = 'blue')
plt.plot(index,o,label='predict Values', linestyle = '-',color = 'red')
plt.title("OrthogonalMatchingPursuitCV")
plt.xlabel("Date")
plt.ylabel("GIRO")
plt.show()



##########################  Random Forest  ############################
regressor = RandomForestRegressor(n_estimators=150, min_samples_split=9)
regressor.fit(trainingSetLabels , y)
reg = regressor.predict(testSetLabels)

mse = mean_squared_error(testSet, reg)
mae = mean_absolute_error(testSet, reg)
R2 = r2_score(testSet, reg, multioutput= 'variance_weighted')
mape = mean_absolute_percentage_error(reg, testSet)

print('mean squared error:\n', mse)
print('mean absolute error:\n', mae)
print('R2-Score:\n', R2)
print('MAPE:\n', mape)

index = test['TARIH']
plt.plot(index,testSet,label='Actual Values', linestyle = '-',color = 'blue')
plt.plot(index,reg,label='predict Values', linestyle = '-',color = 'red')
plt.title("RandomForest")
plt.xlabel("Date")
plt.ylabel("GIRO")
plt.show()



##############################   SGD    ###############################
clf = linear_model.SGDRegressor(loss="huber")
clfReg = clf.fit(trainingSetLabels,y)
sgd = clfReg.predict(testSetLabels)

mse = mean_squared_error(testSet, sgd)
mae = mean_absolute_error(testSet, sgd)
R2 = r2_score(testSet, sgd, multioutput= 'variance_weighted')
mape = mean_absolute_percentage_error(sgd, testSet)

print('mean squared error:\n', mse)
print('mean absolute error:\n', mae)
print('R2-Score:\n', R2)
print('MAPE:\n', mape)

index = test['TARIH']
plt.plot(index,testSet,label='Actual Values', linestyle = '-',color = 'blue')
plt.plot(index,sgd,label='predict Values', linestyle = '-',color = 'red')
plt.title("Stochastic Gradient Descent (SGD) ")
plt.xlabel("Date")
plt.ylabel("GIRO")
plt.show()

######################
