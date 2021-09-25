import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error, roc_curve, accuracy_score, classification_report, roc_auc_score
from sklearn import metrics
from catboost import CatBoostRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle
import plotly.graph_objects as go

def model_selangor():

    strong_feat = ['pcr', 'positivity_rate', 'cases_recovered', 'hosp_pui',
        'pkrc_admitted_total', 'vent_noncovid', 'rtk-ag',
        'hosp_admitted_pui', 'vent_covid', 'icu_noncovid', 'hosp_noncovid',
        'icu_covid', 'hosp_discharged_covid', 'hosp_covid',
        'hosp_discharged_total', 'deaths_new', 'pkrc_covid',
        'pkrc_admitted_covid', 'total_tests', 'pkrc_discharged_covid',
        'hosp_admitted_total', 'beds_covid', 'hosp_discharged_pui']

    final_merged = pd.read_csv('Dataset/final_merged.csv')
    df_selangor = final_merged[final_merged['state'] == 'Selangor']
    X_selangor  = df_selangor[strong_feat]
    y_selangor  = df_selangor['cases_new']

    scaler = StandardScaler()  
    scaler.fit(X_selangor)  
    X_selangor_norm = scaler.transform(X_selangor)
    X_selangor_norm = pd.DataFrame(X_selangor_norm, columns=X_selangor.columns)

    X_selangor_train, X_selangor_test, y_selangor_train, y_selangor_test = train_test_split(X_selangor_norm, y_selangor, test_size=0.3, random_state=2)

    with open('Model/cbr_selangor', 'rb') as file:  
        cbr = pickle.load(file)

    y_selangor_predcbr =cbr.predict(X_selangor_test)
    cbr_mse = mse(y_selangor_test, y_selangor_predcbr)
    cbr_rmse = mse(y_selangor_test, y_selangor_predcbr, squared=False)
    cbr_r2 = cbr.score(X_selangor_test,y_selangor_test)
    cbr_mae = mean_absolute_error(y_selangor_test, y_selangor_predcbr)

    with open('Model/lr_selangor', 'rb') as file:  
        lr = pickle.load(file)

    y_selangor_predlr =lr.predict(X_selangor_test)
    lr_mse = mse(y_selangor_test, y_selangor_predlr)
    lr_rmse = mse(y_selangor_test, y_selangor_predlr, squared=False)
    lr_r2 = lr.score(X_selangor_test,y_selangor_test)
    lr_mae = mean_absolute_error(y_selangor_test, y_selangor_predlr)

    st.markdown('# Selangor')
    st.markdown('## Regression')
    left_column, right_column = st.columns(2)
    with left_column:
        st.markdown('** CatBoost Regressor Score **')
        table1 = go.Figure(data=[go.Table(
        columnwidth=[1, 4],
        header=dict(values=['Metrics', 'Value'],
                    fill=dict(color=['paleturquoise']),
                    align='center',height=30),
        cells=dict(values=[['R2','MAE','MSE','RMSE'],
                        [str(cbr_r2),str(cbr_mae), str(cbr_mse), str(cbr_rmse)]],
                fill=dict(color=['lightcyan']),
                align='center',height=30))
        ])
        table1 = table1.update_layout(width=400,height=200, margin=dict(l=0,r=10,t=5,b=0))
        st.write(table1)

    with right_column:
        st.markdown('** Linear Regressor Score **')
        table2 = go.Figure(data=[go.Table(
        columnwidth=[1, 4],
        header=dict(values=['Metrics', 'Value'],
                    fill=dict(color=['paleturquoise']),
                    align='center',height=30),
        cells=dict(values=[['R2','MAE','MSE','RMSE'],
                        [str(lr_r2),str(lr_mae), str(lr_mse), str(lr_rmse)]],
                fill=dict(color=['lightcyan']),
                align='center',height=30))
        ])
        table2 = table2.update_layout(width=400,height=200, margin=dict(l=0,r=10,t=5,b=0))
        st.write(table2)

    y_selangor_bin =pd.cut(y_selangor,3,labels=['Low','Medium','High'])
    X_selangor_train, X_selangor_test, y_selangor_train, y_selangor_test = train_test_split(X_selangor_norm, y_selangor_bin, test_size=0.3, random_state=2)

    with open('Model/lgbm_selangor', 'rb') as file:  
        lgbm = pickle.load(file)

    y_pred_lgbm_selangor  = lgbm.predict(X_selangor_test)
    class_lgbm = classification_report(y_selangor_test, y_pred_lgbm_selangor )

    st.markdown('## Classification')
    st.markdown('** LGBM Classifier **')

    st.write(class_lgbm)

    fig1,ax1 = plt.subplots(figsize = (20,10))
    ax1.set_title("Confusion Matrix for LGBM Classifier")
    conf_matrix_lgbm = confusion_matrix(y_selangor_test, y_pred_lgbm_selangor)
    lgbm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lgbm,display_labels = lgbm.classes_)
    lgbm_display.plot(cmap = 'Greens',xticks_rotation ='vertical',ax=ax1)
    st.pyplot(fig1)

    with open('Model/rf_selangor', 'rb') as file:  
        rfc = pickle.load(file)

    y_pred_rfc_selangor = rfc.predict(X_selangor_test)
    class_rf = classification_report(y_selangor_test, y_pred_rfc_selangor )

    st.markdown('** Random Forest Classifier **')
    st.write(class_rf)

    fig2,ax2 = plt.subplots(figsize = (20,10))
    ax2.set_title("Confusion Matrix for Random Forest Classifier")
    conf_matrix_rfc = confusion_matrix(y_selangor_test, y_pred_rfc_selangor)
    rfc_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rfc,display_labels = rfc.classes_)
    rfc_display.plot(cmap = 'Greens',xticks_rotation ='vertical',ax=ax2)
    st.pyplot(fig2)

    encoder = LabelEncoder()
    y_selangor_test = encoder.fit_transform(y_selangor_test)

    pred_prob1 = lgbm.predict_proba(X_selangor_test)[:,1]
    pred_prob2 = rfc.predict_proba(X_selangor_test)[:,1]

    fpr1, tpr1, thresh1 = roc_curve(y_selangor_test, pred_prob1, pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(y_selangor_test, pred_prob2, pos_label=1)
    prec_lgbm, rec_lgbm, threshold_lgbm = precision_recall_curve(y_selangor_test, pred_prob1,pos_label=1)
    prec_rfc, rec_rfc, threshold_rfc = precision_recall_curve(y_selangor_test, pred_prob2,pos_label=1)
    random_probs = [0 for i in range(len(y_selangor_test))]
    p_fpr, p_tpr, _ = roc_curve(y_selangor_test, random_probs, pos_label=1)

    st.markdown('## Perfomance Comparasion')

    fig3,ax3 = plt.subplots()
    plt.style.use('seaborn')
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='LGBM Classifier')
    plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Random Forest Classifier')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    st.pyplot(fig3)

    fig4,ax4 = plt.subplots()
    plt.plot(rec_lgbm, prec_lgbm, color='orange', label='LGBM Classifier') 
    plt.plot(rec_rfc, prec_rfc, color='green', label='Random Forest Classifier') 
    baseline4 = len(y_selangor_test[y_selangor_test==1]) / len(y_selangor_test)
    plt.plot([1, 0], [baseline4,baseline4], color='black', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot(fig4)