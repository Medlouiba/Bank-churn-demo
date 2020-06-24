#!/usr/bin/env python
# coding: utf-8

# In[45]:


import pandas as p
import numpy as np
import plotly as py
import sklearn as sk
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,validation_curve,StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_curve,precision_score,precision_recall_curve,auc,recall_score,plot_precision_recall_curve
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import chart_studio.tools as tls

df=p.read_csv('bank_churning_dataset.csv',index_col=0)

df.info()
print('\nshape of df =>',df.shape)
print('df has null values ? =>',df.isnull().values.any())


# In[46]:


df.head()


# In[47]:


data=df.corr()
fig= px.imshow(data,x=data.columns,y=data.columns)
fig.show()


# In[48]:


labels = ['Exits','Remaining']
values = [df['Exit'].sum(),1000-df['Exit'].sum()]

fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0,0.1])])
fig.show()


# In[49]:


fig = px.violin(df, y=df["Age"], x="Exit",color='IsActive',box=True, points='all',
          hover_data=df.columns)
fig.show()


# In[50]:


df=df.replace({'Male':1,'Female':0}) #turn Gender str  values to numeric
X=df.drop(['Exit'],axis=1) #Assign feature space to X
Y=df['Exit']               #Assign target values to Y

#Train_test split, perform feature scaling using standardization
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=17,shuffle=True)
scaler=StandardScaler().fit(x_train)
x_train_normalized=scaler.transform(x_train)
x_test_normalized=scaler.transform(x_test)

#Create an svc object, assign basic parameters
scv=StratifiedKFold(n_splits=5)
svc=svm.SVC(kernel='rbf',random_state=17).fit(x_train_normalized,y_train)

#Using a GridsearchCV, search SVC parameters for the best "f1" score,
params={'C':[17,15,16,10,20,25,23],'gamma':[0.02,0.03,0.1,0.2]} 
search=GridSearchCV(svc,param_grid= params,cv=scv,scoring='f1')
search.fit(x_train_normalized,y_train)

#Get the estimator from gridsearchCV                                                                                               ghest precision_micro scoring  
print(search.best_params_)
svc=search.best_estimator_

#plot the cross validation curve 
p_range=np.logspace(-3,0,10)
train_curve,cross_val_curv=validation_curve(svc,x_train_normalized,
                                            svc.predict(x_train_normalized),
                                            param_name='gamma',
                                            param_range=p_range,
                                            cv=scv,scoring='precision_micro')
train_curve=[x.mean() for x in train_curve]
cross_val_curv=[x.mean() for x in cross_val_curv]

fig = px.line(x=p_range, y=[train_curve,cross_val_curv],log_x=True,title='Cross validation curve')
fig.update_xaxes(title_text='Gamma')
fig.update_yaxes(title_text='Micro Precision  ')

fig.update_layout(showlegend=False)




# In[56]:



#plot the ROC_curve and precision recall curve
y_pred=svc.decision_function(x_test_normalized)
fp,tp,_=roc_curve(y_test,y_pred)

fig = px.line(x=fp,y=tp)
fig.update_xaxes(title_text='False positives')
fig.update_yaxes(title_text='True positives')
fig.show()


# In[58]:


precision,recall,thresholds=precision_recall_curve(y_test,y_pred)

fig = px.line(x=recall,y=precision)
fig.update_xaxes(title_text='Recall')
fig.update_yaxes(title_text='Precision ')
fig.show()


# In[59]:


#Visualize the confusion matrix
matrix=p.DataFrame(confusion_matrix(y_test,svc.predict(x_test_normalized)))
matrix


# In[60]:


precision_score(y_test,svc.predict(x_test_normalized))


# In[62]:


(21+199)/(199+21+29+1)#ACCURACY


# In[ ]:





# In[ ]:




