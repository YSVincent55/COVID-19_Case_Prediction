import streamlit as st 
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
import plotly.graph_objects as go

def lasso_kedah():

    final_merged = pd.read_csv('Dataset/final_merged.csv')
    q3_kedah = final_merged[final_merged['state'] == 'Kedah']

    X = q3_kedah.drop(columns=['cases_new','date','state'])
    y = q3_kedah['cases_new']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    pipeline = Pipeline([('scaler',StandardScaler()),('model',Lasso())])
    search = GridSearchCV(pipeline,{'model__alpha':np.arange(1,10,1)}, cv = 5, scoring="neg_mean_squared_error",verbose=3)
    search.fit(X_train,y_train)

    coef = search.best_estimator_.named_steps['model'].coef_

    featureCoef = pd.DataFrame(columns = ['Columns', 'Value'])
    featureCoef['Columns'] = X.columns
    featureCoef['Value'] = coef

    st.markdown('### Using LASSO')
    table = go.Figure(data=go.Table(
        header=dict(values=list(featureCoef.columns),
                    fill_color='lightcyan',height=30), 
        cells=dict(values=[featureCoef.Columns, featureCoef.Value],
                    fill_color='lavender',height=30)))

    table = table.update_layout(width=600, height=1500)
    st.write(table)

    st.markdown('For Kedah, we decided to use LASSO to find the strong features. LASSO determines the strong features by assigning a coefficient value larger than 0. The LASSO we used is cross validation 5 times with 9 different alpha values to find the most suitable model. Then, we trained the model and extracted the coefficient values of every column. After applying LASSO, we found that the strong features for Kedah are cases_recovered, pcr, total_tests, positivity_rate, deaths_new, deaths_bid, deaths_bid_dod, beds_x, pkrc_covid, pkrc_pui, beds_y, hosp_noncovid, beds_icu_rep, vent, vent_port, icu_covid, icu_noncovid, vent_noncovid, vent_used and vent_port_used')
