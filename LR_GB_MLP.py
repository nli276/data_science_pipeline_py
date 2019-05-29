#!/usr/bin/env python
# coding: utf-8

# # Summary

# **To get results**
# - Simply run all cells to refresh.
# - This notebook will use internet to download dataset and will write files to local disk.
# - Assuming Anaconda python 3 with sklearn, tensorflow, keras, xgboost installed.
# - I worked on this notebook on Saturday afternoon, Saturday evening and Sunday afternoon.
# 
# **Key learnings**
# - Multiple methods are giving similar poor signal (AUC~0.6). Perhaps need to revisit feature engineering in future iterations. Fine tuning of the model training parameters is probably not yet necessary till later stage. 
# - The poor signal is probably either due to weak signal from data (some things are just hard to predict), or feature engineered inproperly (could try more granular binning at the price of more parameters).
# - I think none of the pilot models is good enough for deployment in real world at the current stage; not sure about loan industry risk model benchmark though.
# - If I have to pick, it would be GB. GB and LR are easier to explain than MLP, and easier to convert to rule-based policy design for lenders (unless the lender only consumes risk score - in which case intelligibility of model doesn't matter). 
# - Between GB and LR, in this particular use case GB is more convenient with feature importance function. So my preferrance is GB > LR(with CV) > MLP for this dataset. 
# - This dataset is too small to benefit from deep learning methods.
# - Out-of-sample log loss (~3.8) and weighted avg f1 score (~0.85) are very similar between GB and LR.
# - CV is not always necessary if holdout data (X_test, y_test) is giving similar performance as in-sample. This is actually the case for GB. But it is helpful for LR regularization.
# - Using GB feature importance function, the top 3 features learned are: Annual income of the loan applicant - log scale, Loan applicant’s percentage utilization of their revolving credit facility, and Total number of accounts for the loan applicant. 
# - This notebook is written in more of a "data science" style than software engineering, which would be more modular. Once data processing and experimenting with pilots are done, it would be relatively straightforward to format the pilot models to pipeline, especially with off-the-shelf tools like sklearn or keras.

# # load tools

# In[1]:


import os
from six.moves import urllib


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 100)


# In[4]:


from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


# In[6]:


fig_size = plt.rcParams["figure.figsize"]

fig_size[0] = 5
fig_size[1] = 5

plt.rcParams["figure.figsize"] = fig_size

print("Current size:", fig_size)


# # download data and explore

# In[7]:


#Data exploration is needed when getting new data. 
#Production pipeline of working pilot model may not need this step.


# In[8]:


DOWNLOAD_ROOT = "https://s3.amazonaws.com/"
LOCAL_PATH = os.path.join("datasets")
FILENAME = "DR_Demo_Lending_Club.csv"
DOWNLOAD_URL = DOWNLOAD_ROOT + FILENAME
#since this is data downloaded from public website, I am not taking any precautions as I normally would for ePHI data.


# In[9]:


def fetch_data(url=DOWNLOAD_URL, local_path=LOCAL_PATH, filename=FILENAME):
    if not os.path.isdir(local_path):
        os.makedirs(local_path)
    csv_path = os.path.join(local_path, filename)
    urllib.request.urlretrieve(url, csv_path)


# In[10]:


fetch_data()


# In[11]:


df = pd.read_csv(os.path.join(LOCAL_PATH,FILENAME))
#in general we should be careful with read dtype, when not sure use dtype=object first; although here it seems unnecessary


# In[12]:


df.Id.nunique()#Id unique, use as index


# In[13]:


df.set_index('Id',inplace=True)


# In[14]:


df.head(50)


# In[15]:


df.info()


# In[16]:


df.is_bad.value_counts(dropna=False)
# prevalence of endpoint event is fairly high, that's good. Otherwise for rare event we might need stratified sampling.


# In[17]:


df.emp_title.value_counts(dropna=False)
#too many categories, can onehot if needed; 
#but I doubt if there's much signal from employers (domain expert might think differently), 
#but rather the signal is from whether the borrower have employer or not ... 
#so simply create a binary indicator for NaN (assuming NaN here is unemployed - that might be wrong though)


# In[18]:


df.emp_length.value_counts(dropna=False)
# bin this something like (nan,1,2-3,4-6,7-9,10 and above) and pivot
# why there's a spike at 10 years? (and very different from 9 or 11). 
#I guess after working for 10 years (after college) is about the right time for wedding and home? check their loan purpose next


# In[19]:


df.purpose_cat[df.emp_length=='10'].value_counts(dropna=False)
#ok, the top reasons are not wedding or home
#I wonder if this data of employment length is reliable...maybe it's a biased sample 


# In[20]:


df.home_ownership.value_counts()
#pivot, or just get_dummies in this simple case as each row is for one Id.
#pivoting multi-level categorical variables would help with the parameter penalty 


# In[21]:


df.annual_inc.describe()


# In[22]:


np.log(df.annual_inc+1).hist()
#use log dollar for modeling; +1 to avoid log(0)


# In[23]:


df[df.annual_inc.isna()]
#there is one person has NaN for annual income and employer, along with lots of other fields being NaN
#yet is_bad = 0...this data point would favor low debt_to_income (=1)


# In[24]:


df.verification_status.value_counts(dropna=False)
#what's the difference between the two verified values? I think should combine those two


# In[25]:


df.pymnt_plan.value_counts(dropna=False)
#


# In[26]:


df[df.pymnt_plan=='y']
#is_bad two for two... but probably too few data points to make it in the model


# In[27]:


df.Notes[:10]


# In[28]:


df.purpose[:10]


# In[29]:


df.purpose_cat[:10]


# In[30]:


df.purpose_cat.value_counts()
# note > purpose > purpose_cat, the information gets more concise from notes to purpose_cat
# There might be some useful signal from extracting Notes and Purpose but I doubt it...
# but this is not NLP focused task... so dropping notes and purpose and onehot purpose_cat


# In[31]:


df.zip_code.head()
#there some interesting socio-economic data available online (such as County Health Ranking and Area Deprivation Index) that map to zip.
#but that's probably beyond the scope of this task, dropping zip code


# In[32]:


df.addr_state.value_counts(dropna=False)


# In[33]:


df[df.is_bad==1]['addr_state'].value_counts(dropna=False)


# In[34]:


df.addr_state.value_counts(dropna=False)/df[df.is_bad==1]['addr_state'].value_counts(dropna=False)
#I'm afraid the variance in this ratio isn't enough to justify the use of state in model...dropping it.
#should avoid geo-discriminating algorithms?


# In[35]:


df.debt_to_income.hist(by=[df.is_bad])


# In[36]:


df.groupby(['is_bad'])['debt_to_income'].mean()


# In[37]:


df.delinq_2yrs.value_counts(dropna=False)
#bin and treat as category,0,1,2 and above
#the more the worse obviously


# In[38]:


df.delinq_2yrs.hist(by=[df.is_bad])


# In[39]:


df.groupby(['is_bad'])['delinq_2yrs'].mean()


# In[40]:


df.earliest_cr_line[:20]
#create a feature being length of credit history, only year needed


# In[41]:


#df['earliest_cr_line_date'] = pd.to_datetime(df.earliest_cr_line,format='%m/%d/%Y',errors='coerce')


# In[42]:


#df.earliest_cr_line_date[:10]


# In[43]:


int('12/01/1996'[-4:])


# In[44]:


df['earliest_cr_year'] = df.earliest_cr_line.apply(lambda x: float(x[-4:]) if pd.notnull(x) else np.nan)


# In[45]:


df['earliest_cr_year'].max()
df['earliest_cr_year'].min()


# In[46]:


df.earliest_cr_year.value_counts(dropna=False)


# In[47]:


df['cr_line_history_yr'] = df['earliest_cr_year'].max() - df['earliest_cr_year']


# In[48]:


df[df.cr_line_history_yr.isnull()].shape
#few nan, can consider imputation...but imputation is generally a bad idea


# In[49]:


df.cr_line_history_yr.hist()


# In[50]:


df.groupby(['is_bad'])['cr_line_history_yr'].mean()


# In[51]:


df.cr_line_history_yr.hist(by=[df.is_bad])
#binning is the way to go it seems


# In[52]:


df.inq_last_6mths.hist()
df.inq_last_6mths.value_counts(dropna=False)
#bin it


# In[53]:


df.mths_since_last_delinq.hist()
#bin: nan, 0-36, 


# In[54]:


df.mths_since_last_delinq.value_counts(dropna=False)


# In[55]:


df.groupby(['is_bad'])['mths_since_last_delinq'].mean()
df.groupby(['is_bad'])['mths_since_last_delinq'].median()


# In[56]:


df.mths_since_last_record.hist()


# In[57]:


df.mths_since_last_record.value_counts(dropna=False)


# In[58]:


df.groupby(['is_bad'])['mths_since_last_record'].mean()
df.groupby(['is_bad'])['mths_since_last_record'].median()
#bin: nan, 0,20-80,80 and above


# In[59]:


df.open_acc.hist()#bin or impute


# In[60]:


df.pub_rec.hist()
df.pub_rec.value_counts(dropna=False)#categorize


# In[61]:


df.revol_bal.hist()#very skewed, log($+1) 


# In[62]:


np.log(df.revol_bal+1).hist()


# In[63]:


df.revol_util.hist()


# In[64]:


df.total_acc.hist()


# In[65]:


df.initial_list_status.value_counts(dropna=False)


# In[66]:


df.collections_12_mths_ex_med.describe()


# In[67]:


df.collections_12_mths_ex_med.value_counts(dropna=False)
#this one is weird


# In[68]:


df.groupby(['is_bad'])['collections_12_mths_ex_med'].apply(lambda x: x.isna().sum())
#this one is not as useful as it sounds


# In[69]:


df.mths_since_last_major_derog.hist()


# In[70]:


df.mths_since_last_major_derog.value_counts()


# In[71]:


df.policy_code.value_counts()
#can use but cannot explain without code mapping/description.


# In[72]:


df.groupby(['policy_code'])['is_bad'].sum()
#probably not much signal from policy 


# In[ ]:





# # data processing

# In[73]:


#adding prefix "input_cat/input_num/output" for model ready engineered features: 
#categorial and numerical (continuous) input, and output (target)


# In[74]:


# we'll have to repeat to pd.cut and pd.get_dummies a couple times, 
# as it's hard to automate since each variable would need different cut


# In[75]:


df = pd.read_csv(os.path.join(LOCAL_PATH,FILENAME))


# In[76]:


df.set_index('Id',inplace=True)


# In[77]:


df.shape


# In[78]:


df.head()


# In[ ]:





# In[79]:


df['output_is_bad'] = df.is_bad 


# In[80]:


df['input_cat_employerNA'] = df.emp_title.apply(lambda x: 1 if pd.isnull(x) else 0)


# In[81]:


df.input_cat_employerNA.value_counts()


# In[82]:


df["emp_length"] = df.emp_length.apply(lambda x: float(x) if x!='na' else 0)


# In[83]:


df.emp_length.value_counts()


# In[84]:


df["input_cat_emp_length"] = pd.cut(df["emp_length"],
                               bins=[-1,0.,1.,3.,6.,9.,np.inf],
                               labels=['NA','1yr','3yr','6yr','9yr','10plusyr']
                                   )


# In[85]:


df.input_cat_emp_length.value_counts(dropna=False)


# In[86]:


df.shape


# In[87]:


df = pd.concat([df,pd.get_dummies(df['input_cat_emp_length'], prefix='input_cat_emp_length')],axis=1)
df.drop(['input_cat_emp_length'],axis=1, inplace=True)


# In[88]:


df.shape


# In[89]:


df.input_cat_emp_length_1yr.value_counts()


# In[90]:


df = pd.concat([df,pd.get_dummies(df['home_ownership'], prefix='input_cat_home_ownership')],axis=1)
df.drop(['home_ownership'],axis=1, inplace=True)


# In[91]:


# need to be careful about the highly correlated variables generated from onehot of categorical variable (for example, male, female)
# here we are not using drop_first=True as we would like to see the effect of each category level.


# In[92]:


df.input_cat_home_ownership_RENT.value_counts()


# In[93]:


df.annual_inc.median()


# In[94]:


df.annual_inc.fillna(df.annual_inc.median(),inplace=True)
#impute this NA as median income


# In[95]:


df.annual_inc.describe()


# In[96]:


df['input_num_log_annual_inc'] = np.log(df.annual_inc+1)


# In[97]:


df.input_num_log_annual_inc.hist(bins=50)


# In[98]:


df['input_cat_verified'] = df.verification_status.apply(lambda x: 0 if x=='not verified' else 1)


# In[99]:


df.input_cat_verified.value_counts()


# In[100]:


df['input_cat_pymnt_plan'] = df.pymnt_plan.apply(lambda x: 1 if x=='y' else 0)


# In[101]:


df.input_cat_pymnt_plan.value_counts()


# In[102]:


df.shape


# In[103]:


df = pd.concat([df,pd.get_dummies(df['purpose_cat'], prefix='input_cat_purpose_cat')],axis=1)
df.drop(['purpose_cat'],axis=1, inplace=True)


# In[104]:


df.columns = df.columns.str.replace(' ', '_')


# In[105]:


df.input_cat_purpose_cat_major_purchase_small_business.value_counts()


# In[106]:


df['input_num_debt_to_income'] = df.debt_to_income
#good as is


# In[107]:


df["input_cat_delinq_2yrs"] = pd.cut(df["delinq_2yrs"],
                               bins=[-1,0.,1.,2.,np.inf],
                               labels=['0','1','2','gt2']
                                   )


# In[108]:


df.input_cat_delinq_2yrs.value_counts(dropna=False)


# In[109]:


df = pd.concat([df,pd.get_dummies(df['input_cat_delinq_2yrs'], prefix='input_cat_delinq_2yrs',dummy_na=True)],axis=1)
df.drop(['input_cat_delinq_2yrs'],axis=1, inplace=True)


# In[110]:


df['earliest_cr_year'] = df.earliest_cr_line.apply(lambda x: float(x[-4:]) if pd.notnull(x) else np.nan)


# In[111]:


df['cr_line_history_yr'] = df['earliest_cr_year'].max() - df['earliest_cr_year']
#don't know when is application year so assuming the max, it's relative so signal won't be affected


# In[112]:


df["input_cat_cr_line_history_yr"] = pd.cut(df["cr_line_history_yr"],
                               bins=[-1,5.,10.,15.,20.,np.inf],
                               labels=['5yr','10yr','15yr','20yr','gt20yr']
                                   )


# In[113]:


df.input_cat_cr_line_history_yr.value_counts(dropna=False)


# In[114]:


df = pd.concat([df,pd.get_dummies(df['input_cat_cr_line_history_yr'], prefix='input_cat_cr_line_history_yr',dummy_na=True)],axis=1)
df.drop(['input_cat_cr_line_history_yr'],axis=1, inplace=True)


# In[115]:


df.shape


# In[116]:


df["input_cat_inq_last_6mths"] = pd.cut(df["inq_last_6mths"],
                               bins=[-1,0.,1.,2.,np.inf],
                               labels=['0','1','2','gt2']
                                   )


# In[117]:


df.input_cat_inq_last_6mths.value_counts(dropna=False)


# In[118]:


df = pd.concat([df,pd.get_dummies(df['input_cat_inq_last_6mths'], prefix='input_cat_inq_last_6mths',dummy_na=True)],axis=1)
df.drop(['input_cat_inq_last_6mths'],axis=1, inplace=True)


# In[119]:


df["input_cat_mths_since_last_delinq"] = pd.cut(df["mths_since_last_delinq"],
                               bins=[-1,24.,48.,np.inf],
                               labels=['2yr','4yr','gt4yr']
                                   )


# In[120]:


df.input_cat_mths_since_last_delinq.value_counts(dropna=False)


# In[121]:


df = pd.concat([df,pd.get_dummies(df['input_cat_mths_since_last_delinq'], prefix='input_cat_mths_since_last_delinq',dummy_na=True)],axis=1)
df.drop(['input_cat_mths_since_last_delinq'],axis=1, inplace=True)


# In[122]:


df["input_cat_mths_since_last_record"] = pd.cut(df["mths_since_last_record"],
                               bins=[-1,0.,84.,np.inf],
                               labels=['0','7yr','gt7yr']
                                   )
#7-yr ARM? 


# In[123]:


df.input_cat_mths_since_last_record.value_counts(dropna=False)


# In[124]:


df = pd.concat([df,pd.get_dummies(df['input_cat_mths_since_last_record'], prefix='input_cat_mths_since_last_record',dummy_na=True)],axis=1)
df.drop(['input_cat_mths_since_last_record'],axis=1, inplace=True)


# In[125]:


df['input_num_open_acc'] = df.open_acc.fillna(df.open_acc.median())
#impute with median (only 5 NaN)


# In[126]:


df["input_cat_pub_rec"] = pd.cut(df["pub_rec"],
                               bins=[-1,0.,1.,np.inf],
                               labels=['0','1','gt1']
                                   )


# In[127]:


df.input_cat_pub_rec.value_counts(dropna=False)


# In[128]:


df = pd.concat([df,pd.get_dummies(df['input_cat_pub_rec'], prefix='input_cat_pub_rec',dummy_na=True)],axis=1)
df.drop(['input_cat_pub_rec'],axis=1, inplace=True)


# In[129]:


df['input_num_log_revol_bal'] = np.log(df.revol_bal+1)


# In[130]:


df['input_num_revol_util'] = df.revol_util.fillna(df.revol_util.median())


# In[131]:


df['input_num_total_acc'] = df.total_acc.fillna(df.total_acc.median())


# In[132]:


df['input_cat_initial_list_status_m'] = df.initial_list_status.apply(lambda x: 1 if x=='m' else 0)


# In[133]:


df['input_cat_collections_12_mths_ex_med_NA'] = df.collections_12_mths_ex_med.apply(lambda x: 0 if x==0 else 1)
#it's either 0 or NA, get_dummies for both levels would be wrong


# In[134]:


df['input_cat_mths_since_last_major_derog'] = df.mths_since_last_major_derog.astype(str)

df = pd.concat([df,pd.get_dummies(df['input_cat_mths_since_last_major_derog'], prefix='input_cat_mths_since_last_major_derog',dummy_na=False)],axis=1)
df.drop(['input_cat_mths_since_last_major_derog'],axis=1, inplace=True)


# In[135]:


df = pd.concat([df,pd.get_dummies(df['policy_code'], prefix='input_cat_policy_code',dummy_na=False)],axis=1)


# In[136]:


df.input_cat_policy_code_PC4.value_counts()


# In[137]:


df.filter(regex='input|output').shape
df.filter(regex='input|output').columns
df.filter(regex='input|output').dtypes


# In[138]:


#save modeling data frame
#could round the float64 to reduce the size of csv...but not really necessary for small data
df.filter(regex='input|output').to_csv(os.path.join(LOCAL_PATH,'df.csv'))


# # pilot models

# In[139]:


df = pd.read_csv(os.path.join(LOCAL_PATH,'df.csv'))


# In[140]:


#df = pd.read_csv(os.path.join(LOCAL_PATH,FILENAME))


# In[141]:


df.set_index('Id',inplace=True)


# In[142]:


df.info()


# In[143]:


df.head()


# In[144]:


df.output_is_bad.mean()


# In[145]:


from sklearn.model_selection import train_test_split


# In[146]:


from sklearn import metrics


# In[147]:


y = df.output_is_bad.values

X = df.filter(regex='input').values


# In[148]:


X.max()
X.min()
y.max()
y.min()


# In[149]:


X.shape
y.shape


# ## MLP

# In[150]:


#probably not much benefit to use neural network for such small dataset


# In[151]:


#Here scaling is used (for practical reason) for faster approaching to global minima at error surface


# In[152]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[153]:


X_scale = scaler.fit_transform(X)


# In[154]:


X_scale.shape
X_scale.max()


# In[155]:


X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state = 42)


# In[156]:


print(len(y_train),'train samples')
print(len(y_test),'test samples')


# In[157]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

#rmsprop seems to overfit when many inputs to my previous experience (and sgd is normally better)
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid',name='preds'))

model.summary()
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    validation_data=(X_test,y_test),
                    epochs=20,
                    batch_size=1000)
score = model.evaluate(X_test, y_test, batch_size=1000)


model.summary()


# In[158]:


from sklearn import metrics


# In[159]:


y_score = model.predict(X_test)

y_score = y_score.reshape(y_test.shape[0],)

y_score.mean()
y_test.mean()


# In[160]:


plt.hist(y_score);#weird distribution


# In[161]:


fpr, tpr, _ = metrics.roc_curve(y_test, y_score)
plt.plot(fpr,tpr);


# In[162]:


metrics.auc(fpr,tpr)


# In[163]:


auc=metrics.roc_auc_score(y_test, y_score)

print('AUC: {0:0.2f}'.format(auc))


# In[164]:


y_score = model.predict(X_train)
y_score = y_score.reshape(y_train.shape[0],)
y_score.mean()
y_train.mean()


# In[165]:


fpr, tpr, _ = metrics.roc_curve(y_train, y_score)

plt.plot(fpr,tpr);

metrics.auc(fpr,tpr)

auc=metrics.roc_auc_score(y_train, y_score)

print('AUC: {0:0.2f}'.format(auc))


# In[ ]:





# ## LR

# In[166]:


from sklearn.linear_model import LogisticRegressionCV#so no need to manually regularize LR with CV
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[167]:


y = df.output_is_bad.values
X = df.filter(regex='input').values
X_scale = scaler.fit_transform(X)


# In[168]:


#for LR scaling helps a bit


# In[169]:


X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.3, random_state=0)
logreg = LogisticRegressionCV(cv=5,solver='liblinear')
#tried a few solver but little difference in result
logreg.fit(X_train, y_train)


# In[170]:


y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


# In[171]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


# In[172]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[173]:


from sklearn.metrics import roc_auc_score,roc_curve,log_loss,average_precision_score,precision_recall_curve


# In[174]:


plt.rcParams["figure.figsize"] = [5,5]


# In[175]:


logloss = log_loss(y_train, logreg.predict(X_train))
print('log loss: ',logloss)
logit_roc_auc = roc_auc_score(y_train, logreg.predict(X_train))
fpr, tpr, thresholds = roc_curve(y_train, logreg.predict_proba(X_train)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show();


# In[176]:


logloss = log_loss(y_test, logreg.predict(X_test))
print('log loss: ',logloss)
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show();


# In[177]:


y_pred = logreg.predict_proba(X_test)[:,1]


# In[178]:


plt.hist(y_pred)


# In[179]:


average_precision = average_precision_score(y_test, y_pred)

print('Average precision-recall score: {0:0.5f}'.format(average_precision))


# In[180]:


precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
auprc = metrics.auc(recall,precision)
auprc


# In[181]:


plt.step(recall, precision, color='b', alpha=0.2,
         where='post');
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b');

plt.xlabel('Recall');
plt.ylabel('Precision');
plt.ylim([0.0, 1.0]);
plt.xlim([0.0, 1.0]);
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision));


# In[182]:


# the shape is weird


# ## K-fold CV (using SVM as example)

# In[183]:


df = pd.read_csv(os.path.join(LOCAL_PATH,'df.csv'))
df.set_index('Id',inplace=True)


# In[184]:


from scipy import interp
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()


# In[185]:


y = df.output_is_bad.values
X = df.filter(regex='input').values
X_scale = scaler.fit_transform(X)


# In[186]:


n_samples, n_features = X_scale.shape

# Run classifier with cross-validation and plot ROC curves
cv = StratifiedKFold(n_splits=5)
classifier = svm.SVC(kernel='linear',gamma='auto',probability=True)
#use linear kernel, or it's very slow

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

i = 0
for train, test in cv.split(X_scale, y):
    probas_ = classifier.fit(X_scale[train], y[train]).predict_proba(X_scale[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show();


# In[187]:


from sklearn.metrics import roc_auc_score,log_loss


# In[188]:


logloss = log_loss(y_test, classifier.predict(X_test))
print('log loss: ',logloss)
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show();


# In[189]:


#difference in AUC indicates overfitting


# ## GB (with feature importance)

# In[190]:


#XGBoost model automatically calculates feature importance. 


# In[191]:


#There is no free lunch, but I find these are usually good starting places for quick classifier trials (besides LR): 
#XGBClassifier (from xgboost, faster than the one from sklearn), RandomForestClassifier (from sklearn), MLP (keras with tensorflow backend)


# In[192]:


# plot feature importance using built-in function
#from numpy import loadtxt
from xgboost import XGBClassifier, plot_importance
from matplotlib import pyplot


# In[193]:


#no need for scaling here


# In[194]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)


# In[ ]:





# In[195]:


plt.rcParams["figure.figsize"] = [15,10]
# fit model with training data
model = XGBClassifier()#using the scikit-learn API in xgboost
model.fit(X_train, y_train)
model.get_booster().feature_names = df.filter(regex='input').columns.tolist()
# plot feature importance
plot_importance(model.get_booster())
pyplot.show();


# In[196]:


#so the top 3 features are 
#1) Annual income of the loan applicant - log scale
#2) Loan applicant’s percentage utilization of their revolving credit facility 
#3) Total number of accounts for the loan applicant
#Makes sense, can consult with domain expert / literature to confirm.


# In[ ]:





# In[197]:


plt.rcParams["figure.figsize"] = [5,5]
xgb = model.fit(X_train, y_train)
roc_auc = roc_auc_score(y_train, xgb.predict(X_train))
fpr, tpr, thresholds = roc_curve(y_train, xgb.predict_proba(X_train)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='xgboost (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show();


# In[198]:


roc_auc = roc_auc_score(y_test, xgb.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, xgb.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='xgboost (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0]);
plt.ylim([0.0, 1.05]);
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show();


# In[199]:


from sklearn.metrics import classification_report

print(classification_report(y_test, xgb.predict(X_test)))


# In[200]:


plt.hist(xgb.predict_proba(X_test)[:,1]);


# In[201]:


log_loss(y_test, xgb.predict(X_test))


# In[202]:


#performance is similar to LR.


# In[ ]:




