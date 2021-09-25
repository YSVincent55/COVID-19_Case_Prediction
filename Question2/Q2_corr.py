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

    st.markdown("i. Pahang\n\nNew Cases : Kelantan Johor Kedah Perak Terengganu\n\nTests : Perak Kedah Penang Johor Kelantan\n\nRecovered Cases : Kedah Perak Kelantan Johor Terengganu\n\nii. Johor\n\nNew Cases : (Penang Perak) Kelantan Kedah Sabah Terengganu\n\nTests : Kedah Penang Perak Terengganu Sarawak\n\nRecovered Cases : Kelantan Kedah Perak Sabah Perlis")
    