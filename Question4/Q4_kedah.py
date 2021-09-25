 
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error, roc_curve, accuracy_score, classification_report, roc_auc_score, precision_recall_curve

def q4_kedah():
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


    st.header('Regression Model')

    left_column1, right_column1 = st.columns(2)
    with left_column1:
        left_column1.header('Catboost')
        cbr = pickle.load(open('Model/cbr_kedah', 'rb'))
        y_kedah_predcbr =cbr.predict(X_kedah_test)
        cbr_mse = mse(y_kedah_test, y_kedah_predcbr)
        cbr_rmse = mse(y_kedah_test, y_kedah_predcbr, squared=False)
        st.markdown("r2: " + str(cbr.score(X_kedah_test,y_kedah_test)))
        st.markdown('mae: ' + str(mean_absolute_error(y_kedah_test, y_kedah_predcbr)))
        st.markdown('mse: ' + str(cbr_mse))
        st.markdown('rmse: ' + str(cbr_rmse))
        #st.markdown('Actual: ' + str(y_kedah_test.values))
        #st.markdown('Predicted: ' + str(y_kedah_predcbr))

    with right_column1:
        right_column1.header('Linear Regression')
        lr = pickle.load(open('Model/lr_kedah', 'rb'))
        y_kedah_predlr =lr.predict(X_kedah_test)
        lr_mse = mse(y_kedah_test, y_kedah_predlr)
        lr_rmse = mse(y_kedah_test, y_kedah_predlr, squared=False)
        st.markdown("r2: " + str(lr.score(X_kedah_test,y_kedah_test)))
        st.markdown('mae: ' + str(mean_absolute_error(y_kedah_test, y_kedah_predlr)))
        st.markdown('mse: ' + str(lr_mse))
        st.markdown('rmse: ' + str(lr_rmse))
        #st.markdown('Actual: ' + str(y_kedah_test.values))
        #st.markdown('Predicted: ' + str(y_kedah_predlr))


    y_kedah_bin =pd.cut(y_kedah,3,labels=['Low','Medium','High'])
    X_kedah_train, X_kedah_test, y_kedah_train, y_kedah_test = train_test_split(X_kedah_norm, y_kedah_bin, test_size=0.3, random_state=2)


    st.header('Classification Model')
    left_column2, right_column2 = st.columns(2)

    # Confusion Matrix

    with left_column2:
        left_column2.header('LGBM')
        
        lgbm = pickle.load(open('Model/lgbm_kedah', 'rb'))
        y_pred_lgbm  = lgbm.predict(X_kedah_test)
        st.write(classification_report(y_kedah_test, y_pred_lgbm))

        fig1,ax1 = plt.subplots(figsize = (20,10))
        cf_matrix = confusion_matrix(y_kedah_test, y_pred_lgbm)
        sns.heatmap(cf_matrix, annot=True)

        st.pyplot(fig1)
        

    with right_column2:
        right_column2.header('Random Forest Classifier')
        rfc = pickle.load(open('Model/rfc_kedah', 'rb'))
        y_pred_rfc = rfc.predict(X_kedah_test)
        st.write(classification_report(y_kedah_test, y_pred_rfc))

        fig2,ax2 = plt.subplots(figsize = (20,10))
        cf_matrix = confusion_matrix(y_kedah_test, y_pred_rfc)
        sns.heatmap(cf_matrix, annot=True)
        st.pyplot(fig2)

    # ROC

    with left_column2:
        fig1,ax1 = plt.subplots(figsize = (20,10))

        encoder = LabelEncoder()
        y_kedah_test = encoder.fit_transform(y_kedah_test)
        pred_prob1 = lgbm.predict_proba(X_kedah_test)[:,1]
        fpr1, tpr1, thresh1 = roc_curve(y_kedah_test, pred_prob1, pos_label=1)
        random_probs = [0 for i in range(len(y_kedah_test))]
        p_fpr, p_tpr, _ = roc_curve(y_kedah_test, random_probs, pos_label=1)
        ax1.plot(fpr1, tpr1, linestyle='--',color='orange', label='LGBM')
        ax1.plot(p_fpr, p_tpr, linestyle='--', color='black')
        ax1.set_title('ROC curve')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive rate')
        ax1.legend(loc='best')

        st.pyplot(fig1)

    with right_column2:
        fig2,ax2 = plt.subplots(figsize = (20,10))

        encoder = LabelEncoder()
        y_kedah_test = encoder.fit_transform(y_kedah_test)
        pred_prob2 = rfc.predict_proba(X_kedah_test)[:,1]
        fpr2, tpr2, thresh2 = roc_curve(y_kedah_test, pred_prob2, pos_label=1)
        random_probs = [0 for i in range(len(y_kedah_test))]
        p_fpr, p_tpr, _ = roc_curve(y_kedah_test, random_probs, pos_label=1)
        ax2.plot(fpr2, tpr2, linestyle='--',color='green', label='Random Forest Classifier')
        ax2.plot(p_fpr, p_tpr, linestyle='--', color='black')
        ax2.set_title('ROC curve')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive rate')
        ax2.legend(loc='best')

        st.pyplot(fig2)

    # Precision Recall Curve

    with left_column2:
        fig1,ax1 = plt.subplots(figsize = (20,10))

        prec_lgbm, rec_lgbm, threshold_lgbm = precision_recall_curve(y_kedah_test, pred_prob1,pos_label=1)
        ax1.plot(rec_lgbm, prec_lgbm, color='orange', label='LGBM Classifier') 
        ax1.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend()

        st.pyplot(fig1)

    with right_column2:
        fig2,ax2 = plt.subplots(figsize = (20,10))

        prec_rfc, rec_rfc, threshold_rfc = precision_recall_curve(y_kedah_test, pred_prob2,pos_label=1)
        ax2.plot(rec_rfc, prec_rfc, color='green', label='Random Forest Classifier') 
        ax2.plot([1, 0], [0.1, 0.1], color='black', linestyle='--')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title('Precision-Recall Curve')
        ax2.legend()

        st.pyplot(fig2)