import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
import matplotlib.pyplot as plt
import seaborn as sns

def q2_corr(): 
    st.header('Correlation between States')
    cases_state = pd.read_csv('Dataset/cases_state.csv')
    tests_state = pd.read_csv('Dataset/tests_state.csv')
    tests_state['total_tests'] = tests_state['rtk-ag'] + tests_state['pcr']
    df = cases_state.merge(tests_state, on=['date','state'], how='left')
    df['positivity_rate'] = df['cases_new'] / df['total_tests']
    df['date']= pd.to_datetime(df['date'])

    cases = pd.pivot_table(df, values='cases_new', index=['date'], columns=['state'])
    recover = pd.pivot_table(df, values='cases_recovered', index=['date'], columns=['state'])
    tests = pd.pivot_table(df, values='total_tests', index=['date'], columns=['state'])

    cases_corr = cases.corr()
    fig, ax = plt.subplots(figsize=(15,8), dpi=200)
    sns.heatmap(cases_corr, annot=True)
    ax.set_title('Correlation (New Cases)')
    st.pyplot(fig)

    tests_corr = tests.corr()
    fig, ax = plt.subplots(figsize=(15,8), dpi=200)
    sns.heatmap(tests_corr, annot=True)
    ax.set_title('Correlation (Total Tests)')
    st.pyplot(fig)

    recover_corr = recover.corr()
    fig, ax = plt.subplots(figsize=(15,8), dpi=200)
    sns.heatmap(recover_corr, annot=True)
    ax.set_title('Correlation (Recovered Cases)')
    st.pyplot(fig)

    st.markdown("From the correlation tables above, we can observe that every state does not exhibit a strong correlation in Death with Pahang and Johor. Therefore it is excluded and we have left three attributes, new cases, tests, and recovered cases. We filtered and picked the five highest correlation scores with Pahang and Johor in the perspective of those three attributes. The sequences are arranged from strongest to weakest.\n\n*Strong -> Weak\n\ni. Pahang\n\nNew Cases           : Kelantan(0.79) Johor(0.77) Perak(0.77) Terengganu(0.76) Pulau Pinang(0.73)\n\nTotal Tests            : Kedah(0.67) Perak(0.67) Pulau Pinang(0.65) Johor(0.64) Kelantan(0.59) \n\nRecovered Cases : Kedah(0.85) Kelantan(0.80) Perak(0.77) Johor(0.74) Sabah(0.66)\n\nii. Johor\n\nNew Cases           : Pulau Pinang(0.91) Perak(0.91) Kelantan(0.89) Terengganu(0.86) Sabah(0.83)\n\nTotal Tests             : Kedah(0.82) Pulau Pinang(0.82) Perak(0.80) Terengganu(0.77) Sarawak(0.73)\n\nRecovered Cases : Kelantan(0.87) Perak(0.83) Perlis(0.83) Sarawak(0.82) Kedah(0.79) Sabah(0.79)")
    