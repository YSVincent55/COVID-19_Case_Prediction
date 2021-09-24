 
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 

def q1_pkrc():  
    pkrc = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/pkrc.csv')

    month = {'2020-03-31 00:00:00':'March', '2020-04-30 00:00:00':'April','2020-05-31 00:00:00':'May','2020-06-30 00:00:00':'June','2020-07-31 00:00:00':'July','2020-08-31 00:00:00':'August','2020-09-30 00:00:00':'September','2020-10-31 00:00:00':'October','2020-11-30 00:00:00':'November','2020-12-31 00:00:00':'December',
            '2021-01-31 00:00:00':'January','2021-02-28 00:00:00':'February','2021-03-31 00:00:00':'March','2021-04-30 00:00:00':'April','2021-05-31 00:00:00':'May','2021-06-30 00:00:00':'June','2021-07-31 00:00:00':'July','2021-08-31 00:00:00':'August','2021-09-30 00:00:00':'September',}

    pkrc['date']= pd.to_datetime(pkrc['date'])
    pkrc.rename(columns= {
        "admitted_pui": "pkrc_admitted_pui",
        "admitted_covid": "pkrc_admitted_covid",
        "admitted_total": "pkrc_admitted_total",
        "discharge_pui": "pkrc_discharged_pui",
        "discharge_covid": "pkrc_discharged_covid",
        "discharge_total": "pkrc_discharged_total"
    }, inplace=True)

    selectShow = st.selectbox("Select an aspect to show", ['Outliers Detection','Data Analysis'])

    if selectShow == 'Outliers Detection': 
        c1,c2= st.columns(2)
        with c1:
            boxplot1 = alt.Chart(pkrc).mark_boxplot().encode(
                y='beds:Q'
            ).properties(
                        width=350,
                        height=200,
                        title='Total PKRC Beds'
                    )

            st.altair_chart(boxplot1)

        with c2:
            boxplot2 = alt.Chart(pkrc).mark_boxplot().encode(
                y='pkrc_admitted_pui:Q'
            ).properties(
                        width=350,
                        height=200,
                        title='Number of Individuals(PUI) Admitted to PKRCs'
                    )

            st.altair_chart(boxplot2)
        
        c3,c4= st.columns(2)
        with c3:
            boxplot1 = alt.Chart(pkrc).mark_boxplot().encode(
                y='beds:Q'
            ).properties(
                        width=350,
                        height=200,
                        title='Total PKRC Beds'
                    )

            st.altair_chart(boxplot1)

        with c4:
            boxplot2 = alt.Chart(pkrc).mark_boxplot().encode(
                y='pkrc_admitted_pui:Q'
            ).properties(
                        width=350,
                        height=200,
                        title='Number of Individuals(PUI) Admitted to PKRCs'
                    )

            st.altair_chart(boxplot2)
  
  