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
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
import pickle
import plotly.graph_objects as go

def model_pahang(): 

    strong_feat = ['pcr', 'positivity_rate', 'cases_recovered', 'hosp_pui',
        'pkrc_admitted_total', 'vent_noncovid', 'rtk-ag',
        'hosp_admitted_pui', 'vent_covid', 'icu_noncovid', 'hosp_noncovid',
        'icu_covid', 'hosp_discharged_covid', 'hosp_covid',
        'hosp_discharged_total', 'deaths_new', 'pkrc_covid',
        'pkrc_admitted_covid', 'total_tests', 'pkrc_discharged_covid',
        'hosp_admitted_total', 'beds_covid', 'hosp_discharged_pui']

    final_merged = pd.read_csv('Dataset/final_merged.csv')
    df_pahang = final_merged[final_merged['state'] == 'Pahang']
    X_pahang  = df_pahang[strong_feat]
    y_pahang  = df_pahang['cases_new']

    scaler = StandardScaler()  
    scaler.fit(X_pahang)  
    X_pahang_norm = scaler.transform(X_pahang)
    X_pahang_norm = pd.DataFrame(X_pahang_norm, columns=X_pahang.columns)

    X_pahang_train, X_pahang_test, y_pahang_train, y_pahang_test = train_test_split(X_pahang_norm, y_pahang, test_size=0.3, random_state=2)

    with open('Model/cbr_pahang', 'rb') as file:  
        cbr = pickle.load(file)

    y_pahang_predcbr =cbr.predict(X_pahang_test)
    cbr_mse = mse(y_pahang_test, y_pahang_predcbr)
    cbr_rmse = mse(y_pahang_test, y_pahang_predcbr, squared=False)
    cbr_r2 = cbr.score(X_pahang_test,y_pahang_test)
    cbr_mae = mean_absolute_error(y_pahang_test, y_pahang_predcbr)

    with open('Model/lr_pahang', 'rb') as file:  
        lr = pickle.load(file)

    y_pahang_predlr =lr.predict(X_pahang_test)
    lr_mse = mse(y_pahang_test, y_pahang_predlr)
    lr_rmse = mse(y_pahang_test, y_pahang_predlr, squared=False)
    lr_r2 = lr.score(X_pahang_test,y_pahang_test)
    lr_mae = mean_absolute_error(y_pahang_test, y_pahang_predlr)

    st.markdown('# Pahang')
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

    y_pahang_bin =pd.cut(y_pahang,3,labels=['Low','Medium','High'])
    X_pahang_train, X_pahang_test, y_pahang_train, y_pahang_test = train_test_split(X_pahang_norm, y_pahang_bin, test_size=0.3, random_state=2)

    with open('Model/lgbm_pahang', 'rb') as file:  
        lgbm = pickle.load(file)

    y_pred_lgbm  = lgbm.predict(X_pahang_test)
    class_lgbm = classification_report(y_pahang_test, y_pred_lgbm)

    st.markdown('## Classification')
    st.markdown('** LGBM Classifier **')

    st.write(class_lgbm)

    fig1,ax1 = plt.subplots(figsize = (10,10))
    ax1.set_title("Confusion Matrix for LGBM Classifier")
    conf_matrix_lgbm = confusion_matrix(y_pahang_test, y_pred_lgbm)
    lgbm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lgbm,display_labels = lgbm.classes_)
    lgbm_display.plot(cmap = 'Greens',xticks_rotation ='vertical',ax=ax1)
    st.pyplot(fig1)

    with open('Model/rf_pahang', 'rb') as file:  
        rfc = pickle.load(file)

    y_pred_rfc = rfc.predict(X_pahang_test)
    class_rf = classification_report(y_pahang_test, y_pred_rfc)

    st.markdown('** Random Forest Classifier **')
    st.write(class_rf)

    fig2,ax2 = plt.subplots(figsize = (10,10))
    ax2.set_title("Confusion Matrix for Random Forest Classifier")
    conf_matrix_rfc = confusion_matrix(y_pahang_test, y_pred_rfc)
    rfc_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rfc,display_labels = rfc.classes_)
    rfc_display.plot(cmap = 'Greens',xticks_rotation ='vertical',ax=ax2)
    st.pyplot(fig2)

    encoder = LabelEncoder()
    y_pahang_test = encoder.fit_transform(y_pahang_test)

    pred_prob1 = lgbm.predict_proba(X_pahang_test)[:,1]
    pred_prob2 = rfc.predict_proba(X_pahang_test)[:,1]

    fpr1, tpr1, thresh1 = roc_curve(y_pahang_test, pred_prob1, pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(y_pahang_test, pred_prob2, pos_label=1)
    prec_lgbm, rec_lgbm, threshold_lgbm = precision_recall_curve(y_pahang_test, pred_prob1,pos_label=1)
    prec_rfc, rec_rfc, threshold_rfc = precision_recall_curve(y_pahang_test, pred_prob2,pos_label=1)
    random_probs = [0 for i in range(len(y_pahang_test))]
    p_fpr, p_tpr, _ = roc_curve(y_pahang_test, random_probs, pos_label=1)

    roc_lgbm = pd.DataFrame()
    roc_lgbm['fpr'] = fpr1
    roc_lgbm['tpr'] = tpr1
    roc_lgbm['Type'] = 'LGBM'

    roc_rf = pd.DataFrame()
    roc_rf['fpr'] = fpr2
    roc_rf['tpr'] = tpr2
    roc_rf['Type'] = 'Random Forest'

    roc_all = pd.concat([roc_lgbm,roc_rf])

    roc_p = pd.DataFrame()
    roc_p['fpr'] = p_fpr
    roc_p['tpr'] = p_tpr

    prc_lgbm = pd.DataFrame()
    prc_lgbm['prec'] = prec_lgbm
    prc_lgbm['rec'] = rec_lgbm
    prc_lgbm['Type'] = 'LGBM'

    prc_rf = pd.DataFrame()
    prc_rf['prec'] = prec_rfc
    prc_rf['rec'] = rec_rfc
    prc_rf['Type'] = 'Random Forest'

    prc_all = pd.concat([prc_lgbm,prc_rf])

    st.markdown('## Perfomance Comparasion')

    chart_all = alt.Chart(roc_all).mark_line().encode(
                                                    alt.X('fpr', title="False Positive Rate"),
                                                    alt.Y('tpr', title="True Positive Rate"),
                                                    alt.Color('Type', type='nominal'))

    chart_p = alt.Chart(roc_p).mark_line(strokeDash=[20,5], color = 'black').encode(
                                                                    alt.X('fpr'),
                                                                    alt.Y('tpr'))

    roc_chart =chart_all + chart_p.properties(title='ROC curve',width=800,height=600,)
    st.altair_chart(roc_chart)

    fig4, ax4 = plt.subplots()
    prec_lgbm, rec_lgbm, threshold_lgbm = precision_recall_curve(y_pahang_test, pred_prob1,pos_label=1)
    prec_rfc, rec_rfc, threshold_rfc = precision_recall_curve(y_pahang_test, pred_prob2,pos_label=1)
    plt.plot(rec_lgbm,prec_lgbm, color='orange', label='LGBM Classifier') 
    plt.plot(rec_rfc, prec_rfc, color='green', label='Random Forest Classifier') 
    baseline1 = len(y_pahang_test[y_pahang_test==1]) / len(y_pahang_test)
    plt.plot([1, 0], [baseline1,baseline1], color='black', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot(fig4)
