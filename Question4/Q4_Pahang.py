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
        st.markdown('** Linear Regression Score **')
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

    st.markdown('For regression, we choose CatBoostRegressor and Linear Regression to predict the daily case. After running these two regressors, we obtained the results and compared them side by side. We use mean squared error (MSE) and root mean squared errors (RMSE) as the indicator. The MSE and RMSE of Linear Regressor are lower than the MSE and RMSE of the CatBoostRegressor at 3235.37, 56.88, 4610.63, and 67.90 respectively. Hence, linear regression is performing better in predicting the daily cases for Johor. Besides, we have also used Mean Absolute Error(MAE) as one of the evaluation metrics for Linear Regression and CatBoostRegressor which is at 44.11 and 49.84 respectively. MAE is a good way to evaluate the model as it calculates the absolute difference between actual and predicted values. A lower MAE is always better so Linear Regression performs better than CatBoostRegressor.')

    y_pahang_bin =pd.cut(y_pahang,3,labels=['Low','Medium','High'])
    X_pahang_train, X_pahang_test, y_pahang_train, y_pahang_test = train_test_split(X_pahang_norm, y_pahang_bin, test_size=0.3, random_state=2)

    with open('Model/lgbm_pahang', 'rb') as file:  
        lgbm = pickle.load(file)

    y_pred_lgbm  = lgbm.predict(X_pahang_test)
    class_lgbm = classification_report(y_pahang_test, y_pred_lgbm)

    st.markdown('## Classification')
    st.markdown('We use LGBM and Random Forest Classifier to predict the daily cases and compare their performance. ')
    st.markdown('** LGBM Classifier **')
    st.write(class_lgbm)

    fig1,ax1 = plt.subplots(figsize = (10,10))
    ax1.set_title("Confusion Matrix for LGBM Classifier")
    conf_matrix_lgbm = confusion_matrix(y_pahang_test, y_pred_lgbm)
    lgbm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lgbm,display_labels = lgbm.classes_)
    lgbm_display.plot(cmap = 'Greens',xticks_rotation ='vertical',ax=ax1)
    st.pyplot(fig1)

    st.markdown('The first graph above shows the confusion matrix for the LGBM Classifier. This classifier has high accuracy for “high” and “medium” labels but has low accuracy for “low” labels.')

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

    st.markdown('The second graph above shows the confusion matrix for the Random Forest Classifier. This classifier has high accuracy for “medium” labels but has low accuracy for  “high” and “low” labels. This classifier has a lower accuracy than the LGBM classifier at 0.64 and 0.72 respectively.')

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

    st.markdown('## Perfomance Comparison')

    fig3, ax3 = plt.subplots()
    plt.style.use('seaborn')
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='LGBM Classifier')
    plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Random Forest Classifier')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    st.pyplot(fig3)

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

    st.markdown('The ROC curve and Precision-Recall Curve also shows that the LGBM Classifier is better than the Random Forest classifier. This is due to the fact that the line for LGBM Classifier is closer to the ideal line for ROC curve and Precision-Recall Curve. Therefore, the LGBM Classifier is the better performing model in predicting daily cases for Johor.')
    st.markdown('If we were forced to choose only one type of supervised learning technique, we will prefer a regression model in this case. Through our experiment above, the linear regression model can predict the number of cases quite accurately with only errors of around 56 cases. The classification model can only predict a label of “low”, “medium” or “high”. We cannot know the exact number of cases for that particular day. For example, if we are trying to predict the value beyond the range of our dataset, for instance 100k cases in a day, the model will only predict it as ‘high’, but we have no idea how high the number of cases is. Therefore, we think that regression works better in this task.')