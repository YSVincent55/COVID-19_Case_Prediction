import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 

raw_death = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_state.csv')
pop = pd.read_csv( "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/static/population.csv")

month = {'2020-03-31 00:00:00':'March', '2020-04-30 00:00:00':'April','2020-05-31 00:00:00':'May','2020-06-30 00:00:00':'June','2020-07-31 00:00:00':'July','2020-08-31 00:00:00':'August','2020-09-30 00:00:00':'September','2020-10-31 00:00:00':'October','2020-11-30 00:00:00':'November','2020-12-31 00:00:00':'December',
         '2021-01-31 00:00:00':'January','2021-02-28 00:00:00':'February','2021-03-31 00:00:00':'March','2021-04-30 00:00:00':'April','2021-05-31 00:00:00':'May','2021-06-30 00:00:00':'June','2021-07-31 00:00:00':'July','2021-08-31 00:00:00':'August','2021-09-30 00:00:00':'September',}

m_state = {'Johor','Kedah','Kelantan','Melaka','Negeri Sembilan','Pahang','Pulau Pinang','Perak','Perlis','Selangor','Terengganu','Sabah','Sarawak','W.P. Kuala Lumpur','W.P. Labuan','W.P. Putrajaya'}

raw_death['date']= pd.to_datetime(raw_death['date'])
final_death = raw_death.copy()
final_death = final_death.drop(columns={'deaths_new_dod', 'deaths_bid', 'deaths_bid_dod'})

group_death = final_death.groupby([pd.Grouper(key='date', axis=0, freq='M'),"state"]).agg('sum')
death_state = final_death.groupby(['state']).sum()
death_month = final_death.groupby(pd.Grouper(key='date', axis=0, freq='M')).sum()
maxMonth = str(death_month['deaths_new'].idxmax())

selectGroupBy = st.selectbox("View By", ['Day','Month'])
selectState = st.selectbox("Select a state to view", ['All','Johor','Kedah','Kelantan','Melaka','Negeri Sembilan','Pahang','Pulau Pinang','Perak','Perlis','Selangor','Terengganu','Sabah','Sarawak','W.P. Kuala Lumpur','W.P. Labuan','W.P. Putrajaya'])

if selectGroupBy == 'Day':
    if selectState == 'All':
        chart = alt.Chart(final_death).mark_line().encode(
            alt.X('date', type='temporal'),
            alt.Y('deaths_new', type='quantitative'),
            alt.Color('state', type='nominal')
        ).properties(
            width=1200,
            height=800
        )

        st.altair_chart(chart)
    else:
        state_total = final_death[final_death['state'] == selectState]
        state_chart = alt.Chart(state_total).mark_line().encode(
            alt.X('date', type='temporal'),
            alt.Y('deaths_new', type='quantitative'),
            alt.Color('state', type='nominal')
        ).properties(
            width=1200,
            height=800
        )

        st.altair_chart(state_chart)
else:
    if selectState == 'All':
        group_death = group_death.reset_index()
        monthChart = alt.Chart(group_death).mark_line().encode(
            alt.X('date', type='temporal'),
            alt.Y('deaths_new', type='quantitative'),
            alt.Color('state', type='nominal')
        ).properties(
            width=1200,
            height=800
        )

        st.altair_chart(monthChart)
    else:
        group_death = group_death.reset_index()
        state_month_total = group_death[group_death['state'] == selectState]
        stateMonthChart = alt.Chart(state_month_total).mark_line().encode(
            alt.X('date', type='temporal'),
            alt.Y('deaths_new', type='quantitative'),
            alt.Color('state', type='nominal')
        ).properties(
            width=1200,
            height=800
        )

        st.altair_chart(stateMonthChart)

st.markdown("The mean death of Malaysia is " + str(death_state['deaths_new'].mean()))
st.markdown("The state with the most death is " + str(death_state['deaths_new'].idxmax()) + ' at ' + str(death_state['deaths_new'].max()))
st.markdown("The month with the most death is " + month[maxMonth] + ' at ' + str(death_month['deaths_new'].max()))