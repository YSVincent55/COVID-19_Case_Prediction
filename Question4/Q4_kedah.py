 
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error, roc_curve, accuracy_score, classification_report, roc_auc_score, precision_recall_curve

def model_kedah():
    final_merged = pd.read_csv('Dataset/final_merged.csv')
    strong_feat = ['pcr', 'positivity_rate', 'cases_recovered', 'hosp_pui',
       'pkrc_admitted_total', 'vent_noncovid', 'rtk-ag',
       'hosp_admitted_pui', 'vent_covid', 'icu_noncovid', 'hosp_noncovid',
       'icu_covid', 'hosp_discharged_covid', 'hosp_covid',
       'hosp_discharged_total', 'deaths_new', 'pkrc_covid',
       'pkrc_admitted_covid', 'total_tests', 'pkrc_discharged_covid',
       'hosp_admitted_total', 'beds_covid', 'hosp_discharged_pui']
    df_kedah = final_merged[final_merged['state'] == 'Kedah']
    X_kedah = df_kedah[strong_feat]
    y_kedah = df_kedah['cases_new']
    scaler = StandardScaler()  
    scaler.fit(X_kedah)  
    X_kedah_norm = scaler.transform(X_kedah)
    X_kedah_norm = pd.DataFrame(X_kedah_norm, columns=X_kedah.columns)
    X_kedah_train, X_kedah_test, y_kedah_train, y_kedah_test = train_test_split(X_kedah_norm, y_kedah, test_size=0.3, random_state=2)


    st.markdown('# Kedah')
    st.markdown('## Regression')

    left_column1, right_column1 = st.columns(2)
    with left_column1:
        cbr = pickle.load(open('Model/cbr_kedah', 'rb'))
        y_kedah_predcbr =cbr.predict(X_kedah_test)
        cbr_mse = mse(y_kedah_test, y_kedah_predcbr)
        cbr_rmse = mse(y_kedah_test, y_kedah_predcbr, squared=False)
        cbr_r2 = cbr.score(X_kedah_test,y_kedah_test)
        cbr_mae = mean_absolute_error(y_kedah_test, y_kedah_predcbr)
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

    with right_column1:
        lr = pickle.load(open('Model/lr_kedah', 'rb'))
        y_kedah_predlr =lr.predict(X_kedah_test)
        lr_mse = mse(y_kedah_test, y_kedah_predlr)
        lr_rmse = mse(y_kedah_test, y_kedah_predlr, squared=False)
        lr_r2 = lr.score(X_kedah_test,y_kedah_test)
        lr_mae = mean_absolute_error(y_kedah_test, y_kedah_predlr)
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

    st.markdown('For regression, we choose CatBoostRegressor and Linear Regression to predict the daily case. After running these two regressors, we obtained the results and compared them side by side. We use mean squared error (MSE) and root mean squared errors (RMSE) as the indicator. The MSE and RMSE of linear regression are lower than the MSE and RMSE of the CatBoostRegressor at 7335.67, 85.64, 7483.84 and 86.50 respectively. Hence, linear regression is performing better in predicting the daily cases for Johor. Besides, we have also used Mean Absolute Error(MAE) as one of the evaluation metrics for Linear Regression and CatBoostRegressor which is at 91.49 and 128.45 respectively. MAE is a good way to evaluate the model as it calculates the absolute difference between actual and predicted values. A lower MAE is always better so Linear Regression performs better than CatBoostRegressor.')

    y_kedah_bin =pd.cut(y_kedah,3,labels=['Low','Medium','High'])
    X_kedah_train, X_kedah_test, y_kedah_train, y_kedah_test = train_test_split(X_kedah_norm, y_kedah_bin, test_size=0.3, random_state=2)


    st.markdown('## Classification')
    st.markdown('We use LGBM and Random Forest Classifier to predict the daily cases and compare their performance. ')
    st.markdown('** LGBM Classifier **')

    # Confusion Matrix
    lgbm = pickle.load(open('Model/lgbm_kedah', 'rb'))
    y_pred_lgbm  = lgbm.predict(X_kedah_test)
    st.write(classification_report(y_kedah_test, y_pred_lgbm))

    fig1,ax1 = plt.subplots(figsize = (20,10))
    ax1.set_title("Confusion Matrix for LGBM Classifier")
    conf_matrix_lgbm = confusion_matrix(y_kedah_test, y_pred_lgbm)
    lgbm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_lgbm,display_labels = lgbm.classes_)
    lgbm_display.plot(cmap = 'Greens',xticks_rotation ='vertical',ax=ax1)
    st.pyplot(fig1)
    
    st.markdown('The first graph above shows the confusion matrix for the LGBM Classifier. This classifier has high accuracy for “low” and “medium” labels but has low accuracy for “high” labels.')

    rfc = pickle.load(open('Model/rfc_kedah', 'rb'))
    y_pred_rfc = rfc.predict(X_kedah_test)
    
    st.markdown('** Random Forest Classifier **')
    st.write(classification_report(y_kedah_test, y_pred_rfc))

    fig2,ax2 = plt.subplots(figsize = (20,10))
    ax2.set_title("Confusion Matrix for Random Forest Classifier")
    conf_matrix_rfc = confusion_matrix(y_kedah_test, y_pred_rfc)
    rfc_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_rfc,display_labels = rfc.classes_)
    rfc_display.plot(cmap = 'Greens',xticks_rotation ='vertical',ax=ax2)
    st.pyplot(fig2)

    st.markdown('The second graph above shows the confusion matrix for the Random Forest Classifier. This classifier has high accuracy for “low” and “medium” labels but has low accuracy for “high” labels. This classifier has a similar accuracy with the LGBM classifier but the f1-score of this classifier is better than LGBM Classifier in overall. ')
    # ROC

    encoder = LabelEncoder()
    y_kedah_test = encoder.fit_transform(y_kedah_test)
    pred_prob1 = lgbm.predict_proba(X_kedah_test)[:,1]
    pred_prob2 = rfc.predict_proba(X_kedah_test)[:,1]
    fpr1, tpr1, thresh1 = roc_curve(y_kedah_test, pred_prob1, pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(y_kedah_test, pred_prob2, pos_label=1)
    random_probs = [0 for i in range(len(y_kedah_test))]
    p_fpr, p_tpr, _ = roc_curve(y_kedah_test, random_probs, pos_label=1)
    
    st.markdown('## Perfomance Comparison')
    fig3,ax3 = plt.subplots()
    plt.style.use('seaborn')
    plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='LGBM Classifier')
    plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Random Forest Classifier')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='black')
    plt.title('ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    st.pyplot(fig3)

    # Precision Recall Curve
    fig4,ax4 = plt.subplots()
    prec_lgbm, rec_lgbm, threshold_lgbm = precision_recall_curve(y_kedah_test, pred_prob1,pos_label=1)
    prec_rfc, rec_rfc, threshold_rfc = precision_recall_curve(y_kedah_test, pred_prob2,pos_label=1)
    plt.plot(rec_lgbm, prec_lgbm, color='orange', label='LGBM Classifier') 
    plt.plot(rec_rfc, prec_rfc, color='green', label='Random Forest Classifier') 
    baseline2 = len(y_kedah_test[y_kedah_test==1]) / len(y_kedah_test)
    plt.plot([1, 0], [baseline2,baseline2], color='black', linestyle='--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    st.pyplot(fig4)

    st.markdown('The ROC curve and Precision-Recall Curve also shows that the Random Forest classifier is better than LGBM Classifier. This is due to the fact that the line for Random Forest classifier is closer to the ideal line for ROC curve and Precision-Recall Curve. Therefore, the Random Forest classifier is the better performing model in predicting daily cases for Kedah.')
    st.markdown('If we can choose only one type of supervised learning technique, we will prefer a regression model in this case. By using our linear regression model, we can predict the number of cases quite accurately with only errors around 86 cases instead of just predicting a label to know whether the number of cases are low, medium or high. In addition, if we are trying to predict the value beyond the range of our dataset, for instance 100k cases in a day, the model will only predict it as ‘high’, but we have no idea how high the number of cases is. Therefore, we think that regression works better in this task.')