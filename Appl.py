import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as si
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import sklearn.tree as tree
import plotly.graph_objects as go
import plotly.express as px

st.markdown("""<style>.main{background-color:#0E1117;}</style>""", unsafe_allow_html=True)
color='#0E1117'

header=st.container()
data=st.container()
feature=st.container()
Graphs=st.container()
Logistic_Regression=st.container()
KNN=st.container()
with header:
    st.title('Diabetes Prediction')
    st.text('This Apps predicts a patient who is Diabetic or not')

with data:
    df=pd.read_csv('diabetes.csv')
    st.write('Glimpse on the data')
    st.write('---')
    test_df=pd.read_csv('test_data.csv')

    
    fig=go.Figure(data=go.Table(columnwidth=[1,1,1,1,1,1,1,1,1],header=dict(values=list(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']].columns),align='center'),cells=dict(values=[df.Pregnancies.head(),df.Glucose.head(),df.BloodPressure.head(),df.SkinThickness.head(),df.Insulin.head(),df.BMI.head(),df.DiabetesPedigreeFunction.head(),df.Age.head(),df.Outcome.head()],fill_color='#E5ECF6',align='left')))
    fig.update_layout(margin=dict(l=1,r=1,t=18,b=1),paper_bgcolor=color)
    st.write(df.head())


    fig1=go.Figure(data=go.Table(columnwidth=[1,1,1,1,1,1,1,1],header=dict(values=list(df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].columns),align='center'),cells=dict(values=[test_df.Pregnancies.head(),test_df.Glucose.head(),test_df.BloodPressure.head(),test_df.SkinThickness.head(),test_df.Insulin.head(),test_df.BMI.head(),test_df.DiabetesPedigreeFunction.head(),test_df.Age.head()],fill_color='#E5ECF6',align='left')))
    fig1.update_layout(margin=dict(l=1,r=1,t=18,b=1),paper_bgcolor=color)
    st.write(test_df)
    



    Glucose=pd.DataFrame(df['Glucose'].value_counts()).head(10)
    Insulin=pd.DataFrame(df['Insulin'].value_counts()).head(10)
    BMI=pd.DataFrame(df['BMI'].value_counts()).head(10)
    BloodPressure=pd.DataFrame(df['BloodPressure'].value_counts()).head(10)

    DiabetesPedigree=pd.DataFrame(df['DiabetesPedigreeFunction'].value_counts()).head(10)
    Age=pd.DataFrame(df['Age'].value_counts()).head(10)

st.sidebar.header('Statistical Analyis')

corr_matrix = df.corr()
corre=corr_matrix['Outcome'].sort_values(ascending=False)
st.sidebar.write('Correlation of the features and the Label',corre)

with feature:
    st.header('Features of our Data')
    Features = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    Label='Outcome'
    Featur = st.text_input('Type a feature')

    if st.button('Check Feature'):
         data_features = Featur.lower() in Features 
         'this feature exists!' if data_features else 'the fearure doesn\t exist.'
         st.write('our Label is:',Label)

with Graphs:
     gluc_age=px.bar(df.head(50),x='Age',y='Glucose')
     st.write(gluc_age)
     gluco_age_corr=px.scatter(df,x='Glucose',y='Age')
     st.write(gluco_age_corr)
     bmi_pedi=px.scatter(df,x='DiabetesPedigreeFunction',y='BMI')
     st.write(bmi_pedi)
     Ins_bmi=px.scatter(df,x='BMI',y='Insulin')
     st.write(Ins_bmi)
     Blood_BMI=px.scatter(df,x='BMI',y='BloodPressure')
     st.write(Blood_BMI)

     pie_glucose=px.pie(df.head(50), values='Age') 
     st.write(pie_glucose)




# Probabilities
# 1

desc_stats=df['Glucose'].describe()
mean_Glucose= round(df['Glucose'].mean(),3)
std_Glucose=round(df['Glucose'].std(),3)
print('mean of Glucose:',mean_Glucose,'std Glucose:', std_Glucose)
Prob0=si.stats.norm.cdf((50-mean_Glucose)/std_Glucose)
probability=1-Prob0
st.sidebar.write("the probability of a patient to have at least a 50 level of Glucose is:? ")
st.sidebar.write(probability)
st.sidebar.write('---')

# 2
mean_Ins=round(df['Insulin'].mean(),3)
std_Ins=round(df['Insulin'].mean(),3)
Prob=si.stats.norm.cdf((250-mean_Ins)/std_Ins)
st.sidebar.write('the probability a patient to get at least 250 level of Insulin is:')
st.sidebar.write(1-Prob)
st.sidebar.write('---')

#3
st.sidebar.write('Does the Blood Pressure differ by Age?')
dif=df.groupby('Age').agg({'BloodPressure':['mean', 'std', 'var']}).reset_index()
st.sidebar.write(dif)

#
st.sidebar.subheader('Descriptive Statistics')
st.sidebar.write(df.describe())

st.sidebar.subheader('Age Average')
st.sidebar.write(df['Age'].mean())

st.sidebar.subheader('Variance of Age')
st.sidebar.write(df['Age'].std())

st.sidebar.subheader('Insulin Average')
st.sidebar.write(df['Insulin'].mean())

st.sidebar.subheader('Variance of Insulin')
st.sidebar.write(df['Insulin'].std())

st.sidebar.subheader('Average of Blood Pressure')
st.sidebar.write(df['BloodPressure'].mean())

st.sidebar.subheader('Variance of BloodPressure')
st.sidebar.write(df['BloodPressure'].std())

st.sidebar.subheader('Average of Diabetes Pedigree Function')
st.sidebar.write(df['DiabetesPedigreeFunction'].mean())

st.sidebar.subheader('Variance of Diabetes Pedigree Function')
st.sidebar.write(df['DiabetesPedigreeFunction'].std())

st.sidebar.subheader('BMI Average')
st.sidebar.write(df['BMI'].mean())

st.sidebar.subheader('Variance of BMI')
st.sidebar.write(df['BMI'].std())

st.sidebar.subheader('Glucose Average')
st.sidebar.write(df['Glucose'].mean())

st.sidebar.subheader('Variance of Glucose')
st.sidebar.write(df['Glucose'].std())

st.sidebar.subheader('Skin Thickness Average')
st.sidebar.write(df['SkinThickness'].mean())

st.sidebar.subheader('Variance of Skin Thickness')
st.sidebar.write(df['SkinThickness'].std())

st.sidebar.subheader('Pregnancies Average')
st.sidebar.write(df['Pregnancies'].mean())

st.sidebar.subheader('Variance of Pregnancies')
st.sidebar.write(df['Pregnancies'].std())







    
with Logistic_Regression:
        st.header('Logistic Regression')
        st.write('---')
        st.text('Specify your hyperparameters to tune')
        sel_col, disp_col=st.columns(2)
        Cw=sel_col.selectbox(' what is the value of C?',options=[0.01,0.1,1,10,100],index=0)
        Solver=sel_col.selectbox('what is the best solver?',options=['lbfgs'])
        size_test=sel_col.selectbox('what is the size of the Test data?',options=[0.1,0.2,0.25,.3])

        LR=LogisticRegression(C=Cw, solver=Solver)
        x=df.drop(columns='Outcome')
        y=df[['Outcome']]
        X=StandardScaler().fit_transform(x.astype(float))
        X_t=StandardScaler().fit_transform(test_df.astype(float))
        X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=size_test, random_state=4)
        LR.fit(X,y)
        y_hat=LR.predict(X_t)


        LR.fit(X_train,y_train)
        y_ht=LR.predict(X_test)



        disp_col.write('The Accuracy of the Model is:')
        disp_col.write(accuracy_score(y_test,y_ht))
        disp_col.write('The log Loss of the model is')
        disp_col.write(log_loss(y_test,y_ht))
    

        st.subheader('Prediction')
        for predi in y_hat:
          if predi==0:
               st.write('not diabetic')
          else:
               st.write('diabetic')       



with KNN:
    st.header('KNN')
    st.write('---')
    st.text('Specify hyperparameters to tune')
    sele_col,displ_col=st.columns(2)
    algo=sele_col.selectbox('what is the algorithm to be used?',options=['auto','ball_tree','kd_tree','brute'])
    K=sele_col.slider('what should be the number of neighbors?',min_value=1,max_value=10,value=10,step=1)
    Neigh=KNeighborsClassifier(n_neighbors=K,algorithm=algo).fit(X_train,y_train)
    y1=Neigh.predict(X_test)

    Neighs=KNeighborsClassifier(n_neighbors=K,algorithm=algo).fit(X,y)


    y2=Neighs.predict(X_t)
    st.subheader('**Prediction**')
    for predictions in y2:
         if predictions==0:
              st.write('not diabetic')
         else:
              st.write('diabetic')
              
              
    st.write('---')
    
    
    displ_col.write('Accuracy of the training model is:')
    displ_col.write(accuracy_score(y_train,Neigh.predict(X_train)))
    displ_col.write('Accuracy of the predictive model is:')
    displ_col.write(accuracy_score(y_test,y1)) 





Decision_Tree=st.container()

with Decision_Tree:
     st.header('Decision Tree')
     st.write('---')
     st.write('Specify your hyperparameters')
     col1,col2=st.columns(2)

     split=col1.selectbox('what type of splitter should be?',options=['best','random'])
     max_depths=col1.slider('what is the maximum depth?',min_value=1,max_value=10,value=10,step=1)
     criterio=col1.selectbox('what criterion should be used ', options=['entropy','gini'])
     Tree= DecisionTreeClassifier(criterion=criterio,max_depth=max_depths,splitter=split)  
     Tree.fit(X_train,y_train)
     predic=Tree.predict(X_test)
     col2.write('Accuracy of the training model is:')
     col2.write(accuracy_score(y_train,Tree.predict(X_train)))
     col2.write('Accuracy of the predictive model is:')
     col2.write(accuracy_score(y_test,predic)) 

     Tree.fit(X,y)
     Test=Tree.predict(X_t)
     st.subheader('Prediction')
     for predicti in Test:
          if predicti==0:
               st.write('not diabetic')
          else:
               st.write('diabetic')
              

     
        