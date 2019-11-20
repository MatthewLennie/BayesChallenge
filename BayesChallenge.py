# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:54:35 2019

@author: lennie
"""

import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
dat = pd.read_csv("challenge_data.csv")
keys = ['game_time', 'winner', 'feature_1','feature_2', 'feature_3', 'feature_4']
sns.pairplot(dat[keys])

#%% Strip starting sequence 
# not sure about why but it appears that the sequence is initialized with negative zeros
# for the first 60 seconds. 
# In this analysis I have removed but I consider this an open question. 


dat = dat[dat.winner>=0]


#%% Uh oh
#this shows something wrong. 
dat.dtypes


#%% Investigate weirdness. 
how_many_weird = pd.to_numeric(dat['feature_4'], errors='coerce').isna().sum()

#ok only one weird. What is it?
weird_entry = dat['feature_4'][pd.to_numeric(dat['feature_4'], errors='coerce').isna()]

#Just remove the single entry. 
dat.drop(weird_entry.index,inplace=True)

dat.loc[:,'feature_4'] = pd.to_numeric(dat.loc[:,'feature_4'], errors='coerce')

#%% Look at the data in the simplest form. 

# Feature_4 is suspect
dat.corr()

# it has an outlier and seems to just be a copy of the game_time anyway. 
sns.pairplot(dat[keys])

# Just demonstrating the point a little further here to show the massive outlier
results = smf.ols('feature_4 ~ game_time',data=dat).fit()
sm.graphics.influence_plot(results)

dat.drop('feature_4',axis=1,inplace=True)
#%%
X_train, X_test, y_train, y_test = train_test_split(dat[['feature_1','feature_2']], dat['winner'], test_size=0.33, random_state=42)
#train scaler on the training data only, no data leakages. 
scaler = StandardScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)



#%%
clf = LinearDiscriminantAnalysis()
clf.fit(X_train_scaled, y_train)
print(clf.score(X_train_scaled,y_train))
print(clf.score(X_test_scaled,y_test))

#%%
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
print(clf.score(X_train_scaled,y_train))
print(clf.score(X_test_scaled,y_test))

##
#%%
grouped = dat.groupby('match_id')
#%%
groups =list(grouped.groups.keys())
sns.pairplot(grouped.get_group(groups[1]))




