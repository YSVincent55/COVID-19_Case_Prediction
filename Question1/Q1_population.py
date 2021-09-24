import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 

def q1_population():

    pop = pd.read_csv( "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/static/population.csv")
    pop = pop.drop(0)

    popChart = alt.Chart(pop).mark_bar().encode(
        alt.X('state', type='nominal'),
        alt.Y('pop', type='quantitative')
    ).properties(
                width=1200,
                height=800,
                title='Total Population of Each State'
            )

    st.altair_chart(popChart)

    pop18Chart = alt.Chart(pop).mark_bar().encode(
        alt.X('state', type='nominal'),
        alt.Y('pop_18', type='quantitative')
    ).properties(
                width=1200,
                height=800,
                title='Total Population Above 18 Years Old of Each State'
            )

    st.altair_chart(popChart)

    pop60Chart = alt.Chart(pop).mark_bar().encode(
        alt.X('state', type='nominal'),
        alt.Y('pop_60', type='quantitative')
    ).properties(
                width=1200,
                height=800,
                title='Total Population Above 60 Years Old of Each State'
            )

    st.altair_chart(pop60Chart)