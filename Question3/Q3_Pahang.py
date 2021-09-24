def q3_pahang():
    import streamlit as st 
    import pandas as pd
    import altair as alt 

    final_merged = pd.read_csv('Dataset/final_merged.csv')
    q3_pahang = final_merged[final_merged['state'] == 'Pahang']
    corr_pahang = q3_pahang.corr()
    corr_pahang1 = corr_pahang[abs(corr_pahang['cases_new']) > 0.5]
    corr_pahang1 = corr_pahang1[abs(corr_pahang1['cases_new']) < 1]
    pahang_strongFeatures = corr_pahang1['cases_new']

    corr_pahang = corr_pahang.stack().reset_index().rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'})
    corr_pahang['correlation_label'] = corr_pahang['correlation'].map('{:.2f}'.format)
    #corr_pahang

    pahang_heatmap = alt.Chart(corr_pahang).encode(
        x='variable2:O',
        y='variable:O',
    ).properties(
                    width=1400,
                    height=1400,
                    title='Correlation Map of Pahang'
                )

    text = pahang_heatmap.mark_text().encode(
        text='correlation_label',
        color=alt.condition(
            alt.datum.correlation > 0.5, 
            alt.value('white'),
            alt.value('black')
        )
    )

    cor_plot = pahang_heatmap.mark_rect().encode(
        color='correlation:Q'
    )

    st.altair_chart(cor_plot+text)

    st.markdown('The strong features are cases_recovered, pcr, positivity_rate, beds_x, pkrc_admitted_covid, pkrc_admitted_total, pkrc_discharged_covid, pkrc_discharged_total, pkrc_covid,beds_y, beds_covid,beds_noncrit, beds_icu_rep, beds_icu_total, beds_icu_covid, vent, vent_port, vent_covid, vent_noncovid and vent_port_used ')

