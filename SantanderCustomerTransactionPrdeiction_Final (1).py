#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from scipy import stats
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


train_df1 = pd.read_csv("G://Edwisor Projects//Santander Customer Transaction//train.csv")


# In[3]:


test_df1=pd.read_csv("G://Edwisor Projects//Santander Customer Transaction//test.csv")


# In[4]:


print(train_df1.columns)


# In[5]:


print(train_df1.head())


# In[6]:


print(np.unique(train_df1['target']))


# In[7]:


#Checking the shape of data
print(train_df1.shape)


# In[8]:


print(test_df1.head())


# In[9]:


print(test_df1.shape)


# In[10]:


#As Target column has two classsed 0 and 1 so its a classification problem

#Missing Value Analysis
print(train_df1.isnull().sum().any())


# In[11]:


print(test_df1.isnull().sum().any())


# In[12]:


#We can see# there are no missing values in both train and test data.
#Exploratoroy Data Analysis
#Checking for Stats of the data
print(train_df1.describe())


# In[13]:


print(test_df1.describe())


# In[14]:


#The standard deviation is large for both train and test data.
#Mean Values are also spread largely across both train and test data


# In[15]:


#Checking for distribution of target variable 
plt.hist(train_df1["target"])


# In[16]:


#As it is seen train data is highly biased towards 0 class.Its is Target Class Imbalance problem


# In[17]:


#Dist  Plot of Features  in train and test data
def distplot_features(df_1, df_2,name_1,name_2,features):
    i = 0
    sns.set_style('whitegrid')
    plt.figure()
    fig, ax = plt.subplots(10,10,figsize=(18,22))
    for feature in features:
        i += 1
        plt.subplot(10,10,i)
        sns.distplot(df_1[feature], hist=False,label=name_1)
        sns.distplot(df_2[feature], hist=False,label=name_2)
        plt.xlabel(feature, fontsize=10)
        locs, labels = plt.xticks()
        plt.tick_params(axis='x', which='major', labelsize=6, pad=-6)
        plt.tick_params(axis='y', which='major', labelsize=6)
    plt.show();


# In[18]:


features = train_df1.columns.values[2:102]
distplot_features(train_df1, test_df1, 'Train', 'Test', features)


# In[19]:


#Now the Last 100 Feature Variables distplot

features = train_df1.columns.values[102:202]
distplot_features(train_df1, test_df1, 'Train', 'Test', features)


# In[20]:


#Each Univariant displot is looks balanced.


# In[21]:


#Looking at the Mean and STandard Deviation of each features in train and test data how they variate
def errorbarplot_train_df(feature,train_df,label):
     
        std_tr=np.std(train_df[feature])
        mean_tr=np.mean(train_df[feature])
        x1=[feature]
        error_tr=[std_tr]
        plt.errorbar(x1,mean_tr,yerr=error_tr,fmt="o",alpha=0.5,ecolor="green",capsize=10,elinewidth=10,label= label)
        plt.ylabel("Mean")
        plt.legend()
        plt.show()
                   
def errorbarplot_test_df(feature ,test_df,label): 
     
        std_test=np.std(test_df[feature])
        mean_test=np.mean(test_df[feature])
        x2=[feature]
        error_test=[std_test]
        plt.errorbar(x2,mean_test,yerr=error_test,fmt="o",alpha=0.5,ecolor="red",capsize=10,elinewidth=10,label= label)
        plt.ylabel("Mean")
        plt.legend()
        plt.show()


# In[22]:


features_df_errrplot=train_df1.columns.values[2:202]
for feature in features_df_errrplot:
        errorbarplot_train_df(feature,train_df1,"Train")
        errorbarplot_test_df(feature,test_df1,"Test")


# In[23]:


#Most features  in both train an test data have high varibality between the mean and their deviation 
#as it is visible form errorbarplot.


# In[24]:


#Feeature Selection
#Now for Feature Selection Checking the correaltion between features in train data
feature_corr_df1=train_df1.drop(['ID_code','target'],axis=1)
print(feature_corr_df1.columns)


# In[25]:


feature_corr=feature_corr_df1.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
feature_corr= feature_corr[feature_corr['level_0'] != feature_corr['level_1']]


# In[26]:


#Checking the least correalted features
print(feature_corr.head(10))


# In[27]:


#Most correlated features in train data.
print(feature_corr.tail(10))


# In[28]:


#Even the highest correaltion  among features are approachig zero only


# In[29]:


#Feature Engineering
featurs_derived= train_df1.columns.values[2:202]
for df1 in [test_df1, train_df1]:
    df1['sum'] = df1[featurs_derived].sum(axis=1)  
    df1['min'] = df1[featurs_derived].min(axis=1)
    df1['max'] = df1[featurs_derived].max(axis=1)
    df1['mean'] = df1[featurs_derived].mean(axis=1)
    df1['std'] = df1[featurs_derived].std(axis=1)
    df1['skew'] = df1[featurs_derived].skew(axis=1)
    df1['kurt'] = df1[featurs_derived].kurtosis(axis=1)
    df1['med'] = df1[featurs_derived].median(axis=1)


# In[30]:


#Checked the added features in Train data
print(train_df1[train_df1.columns[202:]].head())


# In[31]:


#Added Features in Test data
print(test_df1[test_df1.columns[201:]].head())


# In[32]:


print(train_df1.columns)


# In[33]:


train_df1=train_df1.drop(['ID_code'],axis=1)
print(train_df1.columns)


# In[34]:


X=train_df1.iloc[:,1:]
Y=pd.DataFrame(train_df1.iloc[:,0])
print(train_df1.shape)


# In[35]:


#Splitting the data into Train and Test Data
X_train1,X_test1,Y_train1,Y_test1=train_test_split(X,Y,test_size=0.3,stratify=Y)

print("X_train1.shape",X_train1.shape)


# In[36]:


print("Y_train1.shape",Y_train1.shape)


# In[37]:


print("X_test1.shape",X_test1.shape)


# In[38]:


print("Y_test1.shape",Y_test1.shape)


# In[39]:


print(Y_train1.target.value_counts())


# In[44]:


scaler = StandardScaler()
#Fitting on Train Data
scaler.fit(X_train1)
#Applying transform to both train and test data.
X_train1=scaler.transform(X_train1)
X_test1=scaler.transform(X_test1)


# In[45]:


print(X_train1.shape)


# In[46]:


print(X_test1.shape)


# In[47]:


#Using Principal Component Analysis for Feature Selection.
from sklearn.decomposition import PCA
#Setting the percentage of Variation expalained by the Components to be 95 percent
pca = PCA(.95)
pca=pca.fit(X_train1)
#Applying to both train and test data
X_train1=pca.transform(X_train1)
X_test1=pca.transform(X_test1)


# In[48]:


print("Number of PrincipalComponents",pca.n_components_)


# In[49]:


print('Explained Variance(Eigen Values) = ', pca.explained_variance_)
print('Principal Components(Eigen Vectors) = ', pca.components_)


# In[50]:


#Analysing Feature Importance through Scree plot for PCA
PC_List=["PC1","PC2","PC3","PC4",'PC5',"PC6","PC7","PC8","PC9","PC10"]
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
percent_variant=percent_variance[1:11]
plt.bar(x= range(1,11), height=percent_variant,tick_label=PC_List)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.show()


# In[51]:


#We see  that all the  Principal Components  are Explaining  same varinace as 
#the first Component so we take first Pricipal Component only. .

# So keeping number of principal components=2


# In[52]:


#Splitting the data into Train and Test Data
X_train2,X_test2,Y_train2,Y_test2=train_test_split(X,Y,test_size=0.3)


# In[53]:


print(X_train2.shape)


# In[54]:


print(X_test2.shape)


# In[55]:


print(Y_train2.shape)


# In[56]:


print(Y_test2.shape)


# In[57]:


scaler = StandardScaler()
#Fitting on Train Data
scaler.fit(X_train2)
#Applying transform to both train and test data.
X_train2=scaler.transform(X_train2)
X_test2=scaler.transform(X_test2)


# In[ ]:





# In[58]:


pca_applied = PCA(n_components=2)
pca_applied=pca_applied.fit(X_train2)
#Applying to both train and test data
X_train2=pca_applied.transform(X_train2)
X_test2=pca_applied.transform(X_test2)


# In[59]:


print("Number of Principal Components in Applied PCA is ",pca_applied .n_components_)


# In[60]:


print('Explained Variance(Eigen Values) = ', pca_applied.explained_variance_)
print('Principal Components(Eigen Vectors) = ', pca_applied.components_)


# In[62]:


print(" Final Train Data Shape X_train2 after",X_train2.shape)


# In[63]:


print(" Final Train Data Shape Y_train2 after PCA",Y_train2.shape)


# In[64]:


print(" Final Test Data Shape X_test2 after PCA",X_test2.shape)


# In[66]:


print(" Final Test Data Shape Y_test2 after PCA",Y_test2.shape)


# In[75]:


X_train3=pd.DataFrame(X_train2)


# In[77]:


X_train3.columns=["PC1","PC2"]


# In[78]:


print(X_train3.head())

X_test3=pd.DataFrame(X_test2)
# In[80]:


X_test3.columns=["PC1","PC2"]


# In[81]:


print(X_test3.head())


# In[87]:


Y_train3=Y_train2


# In[89]:


print(Y_train3.head())


# In[90]:


Y_test3=Y_test2


# In[91]:


print(Y_test3.head())


# In[84]:


#So using SMOTETomek Technique  to balance the dataset
import imblearn
from imblearn.combine import SMOTETomek


# In[ ]:





# In[86]:


#Taking 10 percent sample from X_train2
sample_features=X_train3.sample(frac=0.1)

print(len(sample_features))


# In[92]:


sample_target=Y_train3.sample(frac=0.1)
print(len(sample_target))


# In[ ]:





# In[93]:


Smt = SMOTETomek(ratio='auto')
X_smt, y_smt = Smt.fit_sample(sample_features,sample_target)


# In[94]:


print(np.unique(y_smt))


# In[95]:


X_train4=pd.DataFrame(X_smt)


# In[96]:


Y_train4=pd.DataFrame(y_smt)


# In[ ]:





# In[97]:


X_train4.columns=X_train3.columns


# In[98]:


Y_train4.columns=Y_train4.columns


# In[99]:


print(X_train4.head())


# In[101]:


print(X_train4.shape)


# In[102]:


print(Y_train4.head())


# In[103]:


print(Y_train4.shape)


# In[106]:


print(type(Y_train4))


# In[108]:


Y_train4.columns=['target']


# In[109]:


print(Y_train4.head())


# Y_train4 

# In[110]:


#Checking for Variation of Target Class
plt.hist(Y_train4["target"])


# In[ ]:


#Now we see the target classes are same in ratio


# In[ ]:


#Modelling


# In[120]:


#Modelling-Logistic Regression Model
from sklearn.linear_model import LogisticRegression
LR_Model = LogisticRegression(solver = 'lbfgs')

LR_Model.fit(X_train4,Y_train4)


# In[121]:


print(X_test3.shape)


# In[122]:


print(Y_test3.shape)


# In[124]:


#Making Predictions
Y_Predicted_LRModel=LR_Model.predict(X_test3)


# In[125]:


print(len(Y_Predicted_LRModel))


# In[126]:


from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(Y_test3,Y_Predicted_LRModel)
print(cnf_matrix)


# In[128]:


print("Accuracy:",metrics.accuracy_score(Y_test2, Y_Predicted_LRModel))
print("Precision:",metrics.precision_score(Y_test2, Y_Predicted_LRModel))
print("Recall:",metrics.recall_score(Y_test2, Y_Predicted_LRModel))


# In[ ]:


#We see the accuracy of the logistic regression model is bad with high False Postive Rate


# In[129]:


#Lets check ROC Curve
y_pred_proba = LR_Model.predict_proba(X_test3)[::,1]
fpr, tpr, _ = metrics.roc_curve(Y_test3,  y_pred_proba)
auc = metrics.roc_auc_score(Y_test3, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[130]:


#As we see AUC =0.5 means the Logistic Regression Model is unable to differntiate between the two target classes.
#Hence rejecting the model.


# In[131]:


#Random Forest Model
from sklearn.ensemble import RandomForestClassifier


# In[135]:


RF_model = RandomForestClassifier(n_estimators = 200).fit(X_train4, Y_train4)


# In[136]:


RF_Predictions = RF_model.predict(X_test3)


# In[137]:


#Computing Confusion Matrix
cnf_matrix_RF = metrics.confusion_matrix(Y_test3,RF_Predictions)
print(cnf_matrix_RF)


# In[139]:


print("Accuracy of Random Forest Model(Bagging):",metrics.accuracy_score(Y_test2, RF_Predictions))
print("Precision of Random Forest Model(Bagging):",metrics.precision_score(Y_test2, RF_Predictions))
print("Recall of Random Forest Model(Bagging):",metrics.recall_score(Y_test2, RF_Predictions))


# In[140]:


#As is seen the accuracy of model is increased but the False Positive rate is still high
#Checking AUC curve

RF_model_pred_proba = RF_model.predict_proba(X_test3)[::,1]
fpr_RF, tpr_RF, _ = metrics.roc_curve(Y_test3,RF_model_pred_proba )
auc_RF = metrics.roc_auc_score(Y_test3,RF_model_pred_proba)
plt.plot(fpr_RF,tpr_RF,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[141]:


#We see AUC =0.5 so hecnce classifier is not worth yet.Hence Rejecting the model.


# In[142]:


#XGBoost Model


# In[143]:


Y_train4['target'].value_counts()


# In[243]:


import xgboost as xgb

XGBoost_Model1  =xgb.XGBClassifier(n_estimators=300,objective="binary:logistic", eval_metric="auc",max_depth=3,gamma=1,
learning_rate=0.01,early_stopping_rounds=30)
XGBoost_Model1.fit(X_train4,Y_train4)


# In[244]:


XGBoost_Predictions1= XGBoost_Model1.predict(X_test3)


# In[245]:


#Computing Confusion Matrix

cnf_matrix_XGBoost = metrics.confusion_matrix(Y_test3,XGBoost_Predictions1)
print(cnf_matrix_XGBoost)


# In[246]:


print("Accuracy of XGBosst Model:",metrics.accuracy_score(Y_test2, XGBoost_Predictions1))
print("Precision of XGBosst Model:",metrics.precision_score(Y_test2, XGBoost_Predictions1))

print("Recall of XGBosst Model:",metrics.recall_score(Y_test2, XGBoost_Predictions1))


# In[191]:


#Lets check ROC Curve
XGBoost_pred_proba = XGBoost_Model1.predict_proba(X_test3)[::,1]
fpr_XG, tpr_XG, _ = metrics.roc_curve(Y_test3,  XGBoost_pred_proba)
auc = metrics.roc_auc_score(Y_test3,XGBoost_pred_proba)
plt.plot(fpr_XG,tpr_XG,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[192]:


XGBoost_pred_proba[1:10]


# In[196]:


# store the predicted probabilities for class 1
y_Prob = XGBoost_Model1.predict_proba(X_test3)[:,1]


# In[198]:


print(y_Prob)


# In[199]:


#histogram of predicted probabilities

# 8 bins
plt.hist(y_Prob, bins=8)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of 1s')
plt.ylabel('Frequency')


# In[200]:


y_Prob_0 = XGBoost_Model1.predict_proba(X_test3)[:,0]


# In[201]:


#histogram of predicted probabilities

# 8 bins
plt.hist(y_Prob, bins=8)

# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of 1s')
plt.ylabel('Frequency')


# In[203]:


#We see that both Classes has same range of predictions at thrshold=0.5.So we chosse XGBosst Model.


# In[ ]:





# In[225]:


#Applying PCA on test data
test_ID=test_df1["ID_code"]


# In[209]:


test_df2=test_df1.drop(["ID_code"],axis=1)


# In[211]:


scaler = StandardScaler()
#Fitting on Train Data
scaler.fit(test_df2)
#Applying transform to both train and test data.
test_df2=scaler.transform(test_df2)


# In[1]:


#Using Principal Component Analysis for Feature Selection.

#Setting the percentage of Variation expalained by the Components to be 95 percent
pca = PCA(n_components=2)
pca=pca.fit(test_df2)
#Applying to both train and test data
test_df2=pca.transform(test_df2)


# In[213]:


print("Number of PrincipalComponents",pca.n_components_)


# In[220]:


test_df3=pd.DataFrame(test_df2)
test_df3.columns=["PC1","PC2"]


# In[223]:


#Making prdeictions on test data
Predictions_Final= XGBoost_Model1.predict(test_df3)


# In[ ]:





# In[235]:


Submission_df=pd.DataFrame({"ID_Code":test_df1.ID_code.values})
Submission_df["target"]=Predictions_Final


# In[237]:


Submission_df.to_csv("G://Edwisor Projects//Santander Customer Transaction//Solution//Predictions.csv", index=False)


# In[ ]:




