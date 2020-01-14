import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

#Machine Learning Pkg
from sklearn import model_selection
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC



def main():

    """Do your Machine Learning Job"""

    st.title('Solve all your Machine Learning proplem with more than one WAY')
    st.subheader('Solve all your Machine Learning proplem with more than 60% ACCURACY')

    activity = ["EDA" , 'Plot' , 'Model Building' , 'About']

    chioce = st.sidebar.selectbox('Select Activity',activity)

    if chioce == 'EDA':
        st.subheader('Exploratory Data Analysis')

        data = st.file_uploader('Upload Your DataSet' ,type=['csv','txt'])
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head())

            if st.checkbox('Show Shape'):
                st.write(df.shape)

            if st.checkbox('Show Cloumns'):
                all_columns = df.columns.to_list()
                st.write(all_columns)

            if st.checkbox('Select Columns To Show'):
                selected_columns = st.multiselect("Select Columns",all_columns)
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox('Show Summary'):
                st.write(df.describe())

            if st.checkbox('Show Value Count'):
                st.write(df.iloc[:,-1].value_counts())




    elif chioce == 'Plot':
        st.subheader('Data Visualization')

        data = st.file_uploader('Upload Your DataSet', type=['csv', 'txt'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

        if st.checkbox('Correlation With Seaborn'):
            st.write(sns.heatmap(df.corr(), annot=True))
            st.pyplot()

        if st.checkbox('Pie Chart'):
            all_columns = df.columns.to_list()
            columns_to_plot = st.selectbox('Select 1 Columns',all_columns)
            pie_plot = df[columns_to_plot].value_counts().plot.pie(autopct='$1.1f%%')
            st.write(pie_plot)
            st.pyplot()


        all_columns_names = df.columns.to_list()
        type_of_plot = st.selectbox('Select type Of Plot', ["area","bar","line","hist","box","kde" ])
        selected_columns_name = st.multiselect('Select Columns To Plot' , all_columns_names)

        if st.button('Generate Plot') :
            st.success('Generating Customazible Plot of {} for {}'.format(type_of_plot,selected_columns_name))

            #Plot By streamlit
            if type_of_plot == 'area' :
                cust_data = df[selected_columns_name]
                st.area_chart (cust_data)

            elif type_of_plot == 'bar' :
                cust_data = df[selected_columns_name]
                st.bar_chart(cust_data)

            elif type_of_plot == 'line' :
                cust_data = df[selected_columns_name]
                st.line_chart(cust_data)



            #Customized Plot
            elif type_of_plot :
                cust_plot = df[selected_columns_name].plot(kind=type_of_plot)
                st.write(cust_plot)
                st.pyplot()







    elif chioce == 'Model Building':
        st.subheader('Building Machine Learning Model')
        data = st.file_uploader('Upload Your DataSet', type=['csv', 'txt'])
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())

            #Model Building

            x = df.iloc[: , 0:1]
            y = df.iloc[:,-1]

            seed = 7


            #Model
            models = []
            models.append(('LR',LogisticRegression()))
            models.append(('LDA', LinearDiscriminantAnalysis()))
            models.append(('KNN', KNeighborsClassifier()))
            models.append(('CART', DecisionTreeClassifier()))
            models.append(('NB', GaussianNB()))
            models.append(('SVM', SVC()))


            #Evaluate eatch model in true
            #List
            model_name = []
            model_mean =[]
            model_std = []
            all_models =[]
            scoring = 'accuracy'

            for name ,model in models:
                kfold = model_selection.KFold (n_splits=10,random_state=seed)
                cv_result = model_selection.cross_val_score (model,x,y,cv=kfold,scoring=scoring)
                model_name.append(name)
                model_mean.append(cv_result.mean())
                model_std.append(cv_result.std())

                accuracy_results = {'model_name':name ,'model_accuracy':cv_result.mean(),'standerd_deviation':cv_result.std()}
                all_models.append(accuracy_results)

            if st.checkbox('Metrics as Table'):
                st.dataframe(pd.DataFrame(zip(model_name,model_mean,model_std),columns=['Model Name' , 'Model Accuracy' , 'Standard Deviation']))

            if st.checkbox ('Metrics an JSON'):
                st.json(all_models)










    elif chioce == 'About':
        st.subheader('About The Author')
        st.text(' This Script is made with love , and deram of helping all people to ritch thier GOALS'
                'HOPE ALL ENJOY MACHINE LEARNING ')
        st.text('Email : qualitymeshal@gmail.com')
        st.text('Github : meshalalsutan')







if __name__=="__main__":
    main()