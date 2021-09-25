import streamlit as st 
import pandas as pd
import altair as alt 

def corr_johor():

    final_merged = pd.read_csv('Dataset/final_merged.csv')

    q3_johor = final_merged[final_merged['state'] == 'Johor']
    corr_johor = q3_johor.corr()
    filter_corr = ((corr_johor['cases_new'] > 0.5) & (corr_johor['cases_new'] < 1)) | ((corr_johor['cases_new'] < -0.5) & (corr_johor['cases_new'] > -1))
    johor_strongFeatures = corr_johor[filter_corr]
    johor_strongFeatures = johor_strongFeatures['cases_new']

    corr_johor = corr_johor.stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'})
    corr_johor['correlation_label'] = corr_johor['correlation'].map('{:.2f}'.format)
    #corr_johor

    johor_heatmap = alt.Chart(corr_johor).encode(
        x='variable2:O',
        y='variable:O',
    ).properties(
                    width=1400,
                    height=1400,
                    title='Correlation Map of Johor'
                )

    text = johor_heatmap.mark_text().encode(
        text='correlation_label',
        color=alt.condition(
            (alt.datum.correlation > 0.5) | ((alt.datum.correlation < -0.5) & (alt.datum.correlation > -1.0)), 
            alt.value('white'),
            alt.value('black')
        )
    )

    cor_plot = johor_heatmap.mark_rect().encode(
        color='correlation:Q'
    )

    st.altair_chart(cor_plot+text)
    st.markdown("The correlation map above has shown the correlation between features for Johor. To extract the strong features from the map, we decide to select the absolute correlation coefficients over 0.5 or -0.5 with cases_new. Hence, we found out the strong features are cases_recovered, rtk-ag, pcr, total_tests, positivity_rate , deaths_new, deaths_new_dod, deaths_bid, deaths_bid_dod, pkrc_noncovid, beds_covid, beds_noncrit, admitted_covid_y, admitted_total_y, discharged_covid, discharged_total, hosp_covid, hosp_noncovid, beds_icu, beds_icu_rep, beds_icu_total, vent, vent_port, vent_covid, vent_noncovid, vent_port_used as all of these features obtained a correlation coefficients over 0.5 or below -0.5.")