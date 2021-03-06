 
import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 

def q1_death():  
    raw_death = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_state.csv')

    month = {'2020-03-31 00:00:00':'March', '2020-04-30 00:00:00':'April','2020-05-31 00:00:00':'May','2020-06-30 00:00:00':'June','2020-07-31 00:00:00':'July','2020-08-31 00:00:00':'August','2020-09-30 00:00:00':'September','2020-10-31 00:00:00':'October','2020-11-30 00:00:00':'November','2020-12-31 00:00:00':'December',
            '2021-01-31 00:00:00':'January','2021-02-28 00:00:00':'February','2021-03-31 00:00:00':'March','2021-04-30 00:00:00':'April','2021-05-31 00:00:00':'May','2021-06-30 00:00:00':'June','2021-07-31 00:00:00':'July','2021-08-31 00:00:00':'August','2021-09-30 00:00:00':'September',}

    m_state = {'Johor','Kedah','Kelantan','Melaka','Negeri Sembilan','Pahang','Pulau Pinang','Perak','Perlis','Selangor','Terengganu','Sabah','Sarawak','W.P. Kuala Lumpur','W.P. Labuan','W.P. Putrajaya'}

    raw_death['date']= pd.to_datetime(raw_death['date'])
    final_death = raw_death.copy()

    group_death = final_death.groupby([pd.Grouper(key='date', axis=0, freq='M'),"state"]).agg('sum')
    death_state = final_death.groupby(['state']).sum()
    death_month = final_death.groupby(pd.Grouper(key='date', axis=0, freq='M')).sum()
    maxMonth = str(death_month['deaths_new'].idxmax())

    selectShow = st.selectbox("Select an aspect to show", ['Outliers Detection','Data Analysis'])

    if selectShow == 'Outliers Detection': 
        left_column, right_column = st.columns(2)
        with left_column:
            boxplot1 = alt.Chart(final_death).mark_boxplot().encode(
                y='deaths_new:Q'
            ).properties(
                        width=400,
                        height=800,
                        title='deaths_new'
                    )

            st.altair_chart(boxplot1)

        with right_column:
            boxplot2 = alt.Chart(final_death).mark_boxplot().encode(
                y='deaths_new_dod:Q'
            ).properties(
                        width=400,
                        height=800,
                        title='deaths_new_dod'
                    )

            st.altair_chart(boxplot2)

        left_column1, right_column1 = st.columns(2)
        with left_column1:
            boxplot1 = alt.Chart(final_death).mark_boxplot().encode(
                y='deaths_bid:Q'
            ).properties(
                        width=400,
                        height=800,
                        title='deaths_bid'
                    )

            st.altair_chart(boxplot1)

        with right_column1:
            boxplot2 = alt.Chart(final_death).mark_boxplot().encode(
                y='deaths_bid_dod:Q'
            ).properties(
                        width=400,
                        height=800,
                        title='deaths_bid_dod'
                    )

            st.altair_chart(boxplot2)
        st.markdown('From the boxplots, above, we can see that deaths_new, deaths_new_dod, deaths_bid and deaths_bid_dod contain many outliers. This is due to the fact that at the early stage of COVID-19 has very less death occurs. The death surge after several waves of COVID-19 attacked Malaysia. We decided to not remove the outliers as they are very crucial to the assignment and training model.')
    else:
        left_column2, right_column2 = st.columns(2)
        with left_column2:
            selectGroupBy = st.selectbox("View By", ['Day','Month'])

        with right_column2:
            selectState = st.selectbox("Select a state to view", ['All','Johor','Kedah','Kelantan','Melaka','Negeri Sembilan','Pahang','Pulau Pinang','Perak','Perlis','Selangor','Terengganu','Sabah','Sarawak','W.P. Kuala Lumpur','W.P. Labuan','W.P. Putrajaya'])

        if selectGroupBy == 'Day':
            if selectState == 'All':
                chart = alt.Chart(final_death).mark_line().encode(
                    alt.X('date', type='temporal'),
                    alt.Y('deaths_new', type='quantitative'),
                    alt.Color('state', type='nominal')
                ).properties(
                    width=1200,
                    height=800,
                    title='Total Death of Each State by Day'
                )

                id = final_death['deaths_new'].idxmax()
                day = final_death.loc[id].date
                state = final_death.loc[id].state
                st.altair_chart(chart)
                st.markdown("The total death of Malaysia is " + str(death_state['deaths_new'].sum()))
                st.markdown("The mean death of Malaysia is " + str(death_state['deaths_new'].mean()))
                st.markdown("The state with the most death is " + str(death_state['deaths_new'].idxmax()) + ' at ' + str(death_state['deaths_new'].max()))
                st.markdown("The day with the most death is " + str(day) + ' in ' + state +' at ' + str(final_death['deaths_new'].max()))
                st.markdown("The graph above shows the total death of every state in Malaysia every day. We can see that Selangor has the highest amount of total death among other states. We also noticed that the total death surged for every state after May 2021. This may be due to the delta virus that attacks Malaysia around this period.")
            else:
                state_total = final_death[final_death['state'] == selectState]
                state_chart = alt.Chart(state_total).mark_line().encode(
                    alt.X('date', type='temporal'),
                    alt.Y('deaths_new', type='quantitative'),
                    alt.Color('state', type='nominal')
                ).properties(
                    width=1200,
                    height=800,
                    title='Total Death of ' + selectState + ' by Day'
                )
                day = state_total['deaths_new'].idxmax()
                day = state_total.loc[day].date

                st.altair_chart(state_chart)
                st.markdown("The total death of " + selectState + ' is ' + str(state_total['deaths_new'].sum()))        
                st.markdown("The mean death of " + selectState + ' is ' + str(state_total['deaths_new'].mean()))     
                st.markdown("The day with the most death is " + str(day) + ' at ' + str(state_total['deaths_new'].max()))
                st.markdown("The graph above shows the total death of every state in Malaysia every day. We can see that Selangor has the highest amount of total death among other states. We also noticed that the total death surged for every state after May 2021. This may be due to the delta virus that attacks Malaysia around this period.")
        else:
            if selectState == 'All':
                group_death = group_death.reset_index()
                monthChart = alt.Chart(group_death).mark_line().encode(
                    alt.X('date', type='temporal'),
                    alt.Y('deaths_new', type='quantitative'),
                    alt.Color('state', type='nominal')
                ).properties(
                    width=1200,
                    height=800,
                    title='Total Death of Each State by Month'
                )

                st.altair_chart(monthChart)
                st.markdown("The total death of Malaysia is " + str(death_state['deaths_new'].sum()))
                st.markdown("The mean death of Malaysia is " + str(death_state['deaths_new'].mean()))
                st.markdown("The state with the most death is " + str(death_state['deaths_new'].idxmax()) + ' at ' + str(death_state['deaths_new'].max()))
                st.markdown("The month with the most death is " + month[maxMonth] + ' at ' + str(death_month['deaths_new'].max()))
                st.markdown("The graph above shows the total death of every state in Malaysia for every month. We can see that Selangor topped the graph with a huge difference compared to other states. The pattern of the graph generally follows the pattern of total death grouped by day.")
            else:
                group_death = group_death.reset_index()
                state_month_total = group_death[group_death['state'] == selectState]
                stateMonthChart = alt.Chart(state_month_total).mark_line().encode(
                    alt.X('date', type='temporal'),
                    alt.Y('deaths_new', type='quantitative'),
                    alt.Color('state', type='nominal')
                ).properties(
                    width=1200,
                    height=800,
                    title='Total Death of ' + selectState + ' by Month'            
                )

                st.altair_chart(stateMonthChart)
                st.markdown("The total death of " + selectState + ' is ' + str(state_month_total['deaths_new'].sum()))        
                st.markdown("The mean death of " + selectState + ' is ' + str(state_month_total['deaths_new'].mean()))     
                st.markdown("The month with the most death is " + month[maxMonth] + ' at ' + str(state_month_total['deaths_new'].max()))
                st.markdown("The graph above shows the total death of every state in Malaysia for every month. We can see that Selangor topped the graph with a huge difference compared to other states. The pattern of the graph generally follows the pattern of total death grouped by day.")