#!/usr/bin/env python
# coding: utf-8

# In[708]:


#### NBA Player Salary Prediction #### 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn import cluster 
from sklearn import metrics
from sklearn import datasets
from sklearn.preprocessing import scale
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark')
import warnings
warnings.filterwarnings('ignore')
from statsmodels.api import OLS
import pylab as pl
import sklearn
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering


# In[709]:


df = pd.read_csv("/Users/newuser/Downloads/NBA_Players.csv")


# In[ ]:


##### Data Clearning #####


# In[710]:


df.columns = [i.strip() for i in df.columns.tolist()]
# Change data formal.
df['AGE'] = df['AGE'].str.replace('-', '0')
df['AGE'] = df['AGE'].astype(int)
df['SALARY'] = df['SALARY'].str.replace('Not signed', '0')
df['SALARY'] = df['SALARY'].str.replace(',', '')
df['SALARY'] = df['SALARY'].astype(int)


# In[711]:


## delete Salary w = no signed =0 becuase they don't have PER score  
df_salary = df[ df['SALARY'] == 0 ]
df = df.drop(df_salary.index, axis=0)


# In[712]:


## delete Experience = 0 becuase they don't have any performance score  
df_exp = df[ df['EXPERIENCE'] == 0 
df = df.drop(df_exp.index, axis=0)


# In[713]:


## delete these player becuase they dun have PER score 
df.drop([309, 335, 484, 465, 514],axis=0, inplace = True)


# In[714]:


df.shape


# In[447]:


df


# In[210]:


df.SALARY.describe()


# In[125]:


######Top Bottom 10% approah on salary to show how competitive in NBA market


# In[126]:


SortSalaryH = df.SALARY.sort_values(ascending=False)


# In[127]:


SortSalaryH.head(38)


# In[128]:


MeanH = SortSalaryH.head(38)


# In[129]:


MeanH.mean()


# In[130]:


SortSalaryL = df.SALARY.sort_values(ascending=True)


# In[131]:


SortSalaryL.head(38)


# In[132]:


MeanL = SortSalaryL.head(38)


# In[133]:


MeanL.mean()


# In[ ]:


##### show the correlation b/w salary and other factors not including position ####


# In[745]:


pl.figure(figsize=(12,8))

corr = df.corr() 
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation between Factors')
plt.show()


# In[738]:


df.shape


# In[ ]:


#### Visualising distribution for a single variable: want to prove the career for NBA player is very short


# In[736]:


fig, ax = plt.subplots(figsize=(10, 10))

experience_group = df['EXPERIENCE'].value_counts(bins=[0, 3, 5, 8, 10, 15, 25]).values
labels = ['0-3 years', '5-8 years', '3-5 years', '8-10 years', '10-15 years', 'more than 15 years']
explodes = (0, 0, 0, 0, 0.1, 0.2)
cmap = plt.get_cmap("tab20c")
colors = cmap(np.array([1, 2, 7, 8, 9, 12]))

ax = plt.pie(experience_group, labels=labels, autopct='%1.0f%%', colors=colors, 
             pctdistance=.55, explode=explodes, labeldistance=1.05)
plt.title('NBA Players Experience Group as of 18-19 Season', fontsize=12);


# In[136]:


df.shape


# In[137]:


df3=df.copy


# In[138]:


df_position = df[ df['POSITION'] == "G"]
df3 = df.drop(df_position.index, axis=0)


# In[141]:


df3_position = df3[ df3['POSITION'] == "F"]
df4 = df3.drop(df3_position.index, axis=0)


# In[142]:


df4.shape


# In[143]:


#### Visiualising Distribution for Multiple Variables 


# In[144]:


df.groupby('POSITION').SALARY.describe()


# In[ ]:


##### delate F and G because the sample sizes are so small looks court 


# In[145]:


df4.groupby('POSITION').SALARY.describe()


# In[753]:


##### create means for Histograms####
grouped = df4.groupby(['POSITION'])['SALARY'].mean().sort_values(ascending=False)
grouped


# In[758]:


#### Histograms in means####
grouped.plot.bar(cmap='rainbow')
plt.title('Mean of Salary in Position')
plt.ylabel('Salary')


# In[762]:


grouped1 = df4.groupby(['POSITION'])['HT'].mean().sort_values(ascending=False)
grouped1


# In[764]:


grouped1.plot.bar()
plt.title('Mean of Hight in Position')
plt.ylabel('Hight')
plt.ylabel('Salary')


# In[761]:


##### create sum for Histograms
grouped = df4.groupby(['POSITION'])['SALARY'].mean().sort_values(ascending=False)
grouped


# In[749]:


#### Histograms in TTL
grouped.plot.bar()
plt.title('Sum of Salary in Position')
plt.ylabel('Salary')


# In[150]:


###### compared mean salary in position on each team####
fig, ax = plt.subplots(figsize=(12, 12))

data = df.loc[(df.POSITION!='F') & (df.POSITION!='G')]
position_salary_team = pd.pivot_table(data, values='SALARY', index='TEAM', columns=['POSITION'], aggfunc=np.mean)

sns.heatmap(position_salary_team, linewidth=.7, robust=True, annot=True, cmap='rainbow',
            annot_kws={'size':11}, fmt='.1e');
ax.set_title('Position Mean Salary in Each NBA Team 2018-2019 Season');


# In[151]:


#### Compare Game Data W/ Salary in  Scatter plots #####

pl.figure(figsize=(15,15))
fig, ax = plt.subplots()
ax.scatter(df["SALARY"], df["RPG_LAST_SEASON"], color='g', marker='^')
ax.scatter(df["SALARY"], df["PPG_LAST_SEASON"], color='y', marker = '*')
ax.scatter(df["SALARY"], df["APG_LAST_SEASON"], color='R', marker = '+')
ax.scatter(df["SALARY"], df["PER_LAST_SEASON"], color='B', marker = '1')
plt.show()


# In[715]:


df2=df.copy()


# In[716]:


df2.drop(['TEAM', 'NAME', 'EXPERIENCE', 'URL', 'POSITION', 'AGE', 'HT', 'WT',
       'COLLEGE', 'PPG_CAREER', 'APG_CAREER',
       'RGP_CAREER', 'GP', 'MPG', 'FGM_FGA', 'FGP', 'THM_THA', 'THP',
       'FTM_FTA', 'FTP', 'APG', 'BLKPG', 'STLPG', 'TOPG', 'PPG'], axis=1,inplace=True)


# In[718]:


data = df2


# In[719]:


x = df2.values[:, 1:5]
y = df2.values[:,0]
scaled_datax = scale(x)
scaled_datay = scale(y)


# In[ ]:


#### Scale Data for Clustering ####


# In[720]:


scaled_datax


# In[721]:


scaled_datay


# In[722]:


model = cluster.AgglomerativeClustering(n_clusters=3, linkage="average", affinity="cosine")


# In[723]:


model.fit(scaled_datax)


# In[724]:


#### Completeness_score ####
print(metrics.completeness_score(scaled_datay, model.labels_)) 


# In[725]:


#### Homogeneity_score ####
print(metrics.homogeneity_score(scaled_datay, model.labels_))


# In[498]:


model = linkage(df2, 'ward')
plt.figure()


# In[362]:


pl.figure(figsize=(15,10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(model, leaf_rotation=90., leaf_font_size=8.,)
plt.show()


# In[726]:


cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)


# In[727]:


plt.figure(figsize=(10, 7))


# In[765]:


####Performance vs Salary #####

plt.scatter(df2.values[:,0], df2.values[:, 1], c=cluster.labels_, cmap='rainbow')
plt.scatter(df2.values[:,0], df2.values[:, 4], c=cluster.labels_, cmap='rainbow')
plt.title('Performance Data vs Salary')
plt.xlabel('Salary')
plt.ylabel('Performance per game')
plt.show()


# In[456]:


####Logistic Regression###

SalaryTarget = df2.values[:, 0]
SalaryTarget 


# In[608]:


Performancedata = df2.values[:,1:]
Performancedata


# In[458]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(Performancedata, SalaryTarget,
test_size=0.30)


# In[165]:


lm = LinearRegression()
lm.fit(X_train, Y_train)
lm.fit(Performancedata, SalaryTarget)
print(lm.intercept_)
print(lm.coef_)


# In[451]:


##### want to change salary = 1 > mean and 0 < mean to run the  Precision etc
df2.SALARY.mean()


# In[452]:


#### 1 == > mean, 0 == < mean
new_salary = []
for line in df.SALARY:
    if line> 8562280:
        new_salary.append(1)
    else:
        new_salary.append(0)
        


# In[532]:


data2 = df2


# In[533]:


data2["SALARY"] = new_salary


# In[534]:


data2.head()


# In[632]:


lm = LogisticRegression()


# In[692]:


X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(Performancedata, SalaryTarget, test_size=0.30)


# In[694]:


lm.fit(X_train, Y_train)
lm.predict_proba(X_test)
print(lm.intercept_)


# In[695]:


print(lm.coef_)


# In[696]:


predicted = lm.predict(X_test)
print(metrics.classification_report(Y_test, predicted))


# In[697]:


print(metrics.confusion_matrix(Y_test, predicted))


# In[698]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[699]:


cnf_matrix = metrics.confusion_matrix(Y_test, predicted)


# In[700]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[701]:


####Accuracy####
print("Accuracy:",metrics.accuracy_score(Y_test, predicted))


# In[702]:


####Precision###
print("Precision:",metrics.precision_score(Y_test, predicted))


# In[703]:


####Recall####
print("Recall:",metrics.recall_score(Y_test, predicted))

