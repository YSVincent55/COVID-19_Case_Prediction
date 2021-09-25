import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from boruta import BorutaPy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

def boruta_selangor():
    final_merged = pd.read_csv('Dataset/final_merged.csv')
    q3_selangor = final_merged[final_merged['state'] == 'Selangor']
    y_sel = q3_selangor['cases_new']
    X_sel = q3_selangor.drop(['cases_new','date','state'], 1)
    colnames = X_sel.columns

    rfc = RandomForestClassifier(max_depth =4)
    feat_selector = BorutaPy(rfc, n_estimators = 'auto',verbose=2,random_state=1)
    feat_selector.fit(X_sel.values,y_sel.values.ravel())

    boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order=-1)
    boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
    boruta_score = boruta_score.sort_values("Score", ascending = False)

    sns_boruta_plot = sns.catplot(x="Score", y="Features", data = boruta_score[0:23], kind = "bar", 
                height=12, aspect=1.5, palette='coolwarm')
    plt.title("Boruta Top 23 Features")
    st.pyplot(sns_boruta_plot)