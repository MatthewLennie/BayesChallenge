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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
import numpy as np

dat = pd.read_csv("challenge_data.csv")
keys = ['game_time', 'winner', 'feature_1','feature_2', 'feature_3', 'feature_4']
sns.pairplot(dat[keys])

#dat = pd.concat([dat, dat.shift(1).add_suffix('lag')],axis=1)

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
#dat.drop('feature_4lag',axis=1,inplace=True)

#%%
X_train, X_test, y_train, y_test = train_test_split(dat[['feature_1','feature_2','feature_3']], dat['winner'], test_size=0.33, random_state=42)
#train scaler on the training data only, no data leakages. 
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#%% Just tried straight up fitting a few basic models ignoring the time series effects. 
#%%
clf = LinearDiscriminantAnalysis()
clf.fit(X_train_scaled, y_train)
print(clf.score(X_train_scaled,y_train))
print(clf.score(X_test_scaled,y_test))

#%%
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
print(clf.score(X_train_scaled,y_train))
print(clf.score(X_test_scaled,y_test))

#%%
clf = RandomForestClassifier(n_estimators=100, max_depth=3,random_state=0)
clf.fit(X_train, y_train)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))


#%% Tried squeezing out a little more performance. 
def GroupedFeatures(group):
    group[1]['feature_1_cum'] = group[1]['feature_1'].cumsum()
    group[1]['feature_2_cum'] = group[1]['feature_2'].cumsum()
    group[1]['feature_1_var'] = group[1]['feature_1'].rolling(4).var()
    group[1]['feature_2_var'] = group[1]['feature_2'].rolling(4).var()
    group[1]['feature_1_mean'] = group[1]['feature_1'].rolling(4).mean()
    group[1]['feature_2_mean'] = group[1]['feature_2'].rolling(4).mean()
    return group
#%%
grouped = dat.groupby('match_id')
new_df =[]
groups =list(grouped.groups.keys())
variances = []
for group in grouped:

#    group[1][group[1].winner>=0]
    
    new_df.append(GroupedFeatures(group)[1])
    
df = pd.concat(new_df)

#sns.pairplot(df[['game_time', 'winner', 'feature_1','feature_2', 'feature_3','feature_1_cum','feature_2_cum']].sample(1000),hue='winner')

#%%

df = df[df.winner>=0]
df.dropna(inplace=True)
X_train, X_test, y_train, y_test = train_test_split(df[['feature_1','feature_2','feature_3','feature_1_cum','feature_2_cum','feature_1_var','feature_2_var','feature_1_mean','feature_2_mean']], df['winner'], test_size=0.33, random_state=42)
#train scaler on the training data only, no data leakages. 
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
clf1 = LinearDiscriminantAnalysis()
clf1.fit(X_train_scaled, y_train)
print(clf1.score(X_train_scaled,y_train))
print(clf1.score(X_test_scaled,y_test))

#%%
clf2 = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)
print(clf2.score(X_train_scaled,y_train))
print(clf2.score(X_test_scaled,y_test))

#%%
clf3 = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)
clf3.fit(X_train, y_train)

print(clf3.score(X_train, y_train))
print(clf3.score(X_test, y_test))

#%%

eclf2 = VotingClassifier(estimators=[
        ('lda', clf1), ('lr', clf2), ('rf', clf3)],
        voting='soft')
eclf2 = eclf2.fit(X_train, y_train)
print(eclf2.score(X_train,y_train))
print(eclf2.score(X_test,y_test))


