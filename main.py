import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
import Q1_death as death

selectQuestion = st.sidebar.radio('Select a Question to View', ('Question 1','Question 2','Question 3','Question 4'))

if selectQuestion == 'Question 1':
    death.Q1_death()