import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import ensemble
import seaborn as sns
import matplotlib.pyplot as plt

#read the data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#train.drop('SalePrice', axis=1, inplace=True)


train_corr = train.corr()

sns.set(font_scale=1.25)
sns.jointplot(x=train['OverallQual'], y=train['SalePrice'])

train.drop(train[((train['OverallQual'] < 4.5) & (train['OverallQual'] > 3.7))
                         & (train['SalePrice']>200000)].index).reset_index(drop=True)

train.drop(train[(train['OverallQual'] == 10)
                         & (train['SalePrice']>650000)].index).reset_index(drop=True)

train.drop(train[(train['OverallQual'] == 10)
                         & (train['SalePrice']<200000)].index).reset_index(drop=True)

train.drop(train[(train['OverallQual'] == 8)
                         & (train['SalePrice']>500000)].index).reset_index(drop=True)

sns.jointplot(x=train['OverallQual'], y=train['SalePrice'])

sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['GrLivArea']>4000)
                         & (train['SalePrice']<200000)].index).reset_index(drop=True)

sns.jointplot(x=train['GrLivArea'], y=train['SalePrice'], kind='reg')

sns.jointplot(x=train['ExterQual'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['ExterQual'] == 3)
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

train.drop(train[(train['ExterQual'] == 4)
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

sns.jointplot(x=train['ExterQual'], y=train['SalePrice'], kind='reg')

sns.jointplot(x=train['KitchenQual'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['KitchenQual'] == 3)
                         & (train['SalePrice']>600000)].index).reset_index(drop=True)

train.drop(train[(train['KitchenQual'] == 4)
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

sns.jointplot(x=train['KitchenQual'], y=train['SalePrice'], kind='reg')

sns.jointplot(x=train['GarageCars'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['GarageCars'] == 3)
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

sns.jointplot(x=train['GarageCars'], y=train['SalePrice'], kind='reg')

sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')

train.drop(train[((train['GarageArea'] < 1000) & (train['GarageArea'] > 500))
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

sns.jointplot(x=train['GarageArea'], y=train['SalePrice'], kind='reg')

sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg')

train.drop(train[((train['TotalBsmtSF'] < 4000) & (train['TotalBsmtSF'] > 2000))
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

train.drop(train[(train['TotalBsmtSF'] > 4000)
                         & (train['SalePrice']<200000)].index).reset_index(drop=True)

sns.jointplot(x=train['TotalBsmtSF'], y=train['SalePrice'], kind='reg')


sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='reg')

train.drop(train[((train['1stFlrSF'] < 3000) & (train['1stFlrSF'] > 2000))
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

train.drop(train[(train['1stFlrSF'] > 4000)
                         & (train['SalePrice']<200000)].index).reset_index(drop=True)

sns.jointplot(x=train['1stFlrSF'], y=train['SalePrice'], kind='reg')

sns.jointplot(x=train['FullBath'], y=train['SalePrice'], kind='reg')


train.drop(train[(train['FullBath']== 0)
                         & (train['SalePrice']>300000)].index).reset_index(drop=True)

sns.jointplot(x=train['FullBath'], y=train['SalePrice'], kind='reg')


sns.jointplot(x=train['GarageFinish'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['GarageFinish']== 1)
                         & (train['SalePrice']>400000)].index).reset_index(drop=True)

train.drop(train[(train['GarageFinish']== 2)
                         & (train['SalePrice']>550000)].index).reset_index(drop=True)

train.drop(train[(train['GarageFinish']== 3)
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)


sns.jointplot(x=train['GarageFinish'], y=train['SalePrice'], kind='reg')


sns.jointplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['TotRmsAbvGrd']== 10)
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)

train.drop(train[(train['TotRmsAbvGrd']== 14)
                         & (train['SalePrice']<250000)].index).reset_index(drop=True)


sns.jointplot(x=train['TotRmsAbvGrd'], y=train['SalePrice'], kind='reg')


sns.jointplot(x=train['YearBuilt'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['YearBuilt'] < 1900)
                         & (train['SalePrice']>400000)].index).reset_index(drop=True)

train.drop(train[(train['YearBuilt'] < 2000)
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)


sns.jointplot(x=train['YearBuilt'], y=train['SalePrice'], kind='reg')

sns.jointplot(x=train['FireplaceQu'], y=train['SalePrice'], kind='reg')

train.drop(train[(train['FireplaceQu']== 3)
                         & (train['SalePrice']>600000)].index).reset_index(drop=True)

train.drop(train[(train['FireplaceQu']== 5)
                         & (train['SalePrice']>600000)].index).reset_index(drop=True)

sns.jointplot(x=train['FireplaceQu'], y=train['SalePrice'], kind='reg')


sns.jointplot(x=train['YearRemodAdd'], y=train['SalePrice'], kind='reg')

train.drop(train[((train['YearRemodAdd'] < 2000) & (train['YearRemodAdd'] > 1980))
                         & (train['SalePrice']>700000)].index).reset_index(drop=True)


sns.jointplot(x=train['YearRemodAdd'], y=train['SalePrice'], kind='reg')

plt.show()


salePrice = train['SalePrice']
train.drop('SalePrice', axis=1, inplace=True)
train.drop('Id', axis=1, inplace=True)

solution_df = pd.DataFrame(test['Id'])
test.drop('Id', axis=1, inplace=True)

#linear regression
lr = linear_model.LinearRegression().fit(train, salePrice)
solution_df['SalePrice'] = lr.predict(test)
solution_df.to_csv('solutions/lr_solution_no_outliers.csv')

solution_df.drop('SalePrice', axis=1, inplace=True)

#elastic net regression
eNet = linear_model.ElasticNet(alpha=.01).fit(train, salePrice)
solution_df['SalePrice'] = eNet.predict(test)
solution_df.to_csv('solutions/eNet_solution_no_outliers.csv')

solution_df.drop('SalePrice', axis=1, inplace=True)


#lasso regression
lasso = linear_model.Lasso(alpha=1).fit(train, salePrice)
solution_df['SalePrice'] = lasso.predict(test)
solution_df.to_csv('solutions/lasso_solution_no_outliers.csv')

solution_df.drop('SalePrice', axis=1, inplace=True)


#gradient boosting regression
gb = ensemble.GradientBoostingRegressor()
gb.fit(train, salePrice)
solution_df['SalePrice'] = gb.predict(test)
solution_df.to_csv('solutions/gb_solution_no_outliers.csv')

solution_df.drop('SalePrice', axis=1, inplace=True)


#random forest regression
rf = ensemble.RandomForestRegressor(n_estimator, bootstrap=False)
rf.fit(train, salePrice)
solution_df['SalePrice'] = rf.predict(test)
solution_df.to_csv('solutions/rf_solution_no_outliers.csv')

solution_df.drop('SalePrice', axis=1, inplace=True)


#ridge regression
ridge = linear_model.Ridge(alpha=.01)
ridge.fit(train, salePrice)
solution_df['SalePrice'] = ridge.predict(test)
solution_df.to_csv('solutions/ridge_solutions_no_outliers.csv')
