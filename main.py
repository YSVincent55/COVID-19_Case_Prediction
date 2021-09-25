import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
from Question1 import Q1_death as death
from Question1 import Q1_population as population
from Question1 import Q1_PKRC as pkrc
from Question3 import Q3_Kedah as q3kedah
from Question3 import Q3_Pahang as q3pahang
from Question4 import Q4_Johor as q4johor

selectQuestion = st.sidebar.radio('Select a Question to View', ('Question 1','Question 2','Question 3','Question 4'))

if selectQuestion == 'Question 1':
    selectDatasets = st.selectbox("Select Datasets", ['Cases','Tests','Deaths','Population','PKRC','Hospital','ICU'])
   
    #if selectDatasets == 'Cases':
    #elif selectDatasets == 'Tests':
    if selectDatasets == 'Deaths':
        death.q1_death()
    elif selectDatasets == 'Population':
        population.q1_population()
    elif selectDatasets == 'PKRC':
        pkrc.q1_pkrc()
    #elif selectDatasets == 'Hospital':
    #else:

elif selectQuestion == 'Question 2':
    selectStates_1 = st.selectbox("Select States", ['Pahang','Johor'])
    
    #if selectStates_1 == 'Pahang':      
    #elif selectStates_1 == 'Johor':

elif selectQuestion == 'Question 3':
   selectStates_2 = st.selectbox("Select States", ['Pahang','Kedah','Johor','Selangor'])

    if selectStates_2 == 'Pahang': 
        q3pahang.q3_pahang()
    elif selectStates_2 == 'Kedah':
        q3kedah.q3_kedah()
    #elif selectStates_2 == 'Johor':
    #else:

else:
    selectStates_3 = st.selectbox("Select States", ['Pahang','Kedah','Johor','Selangor'])

    #if selectStates_3 == 'Pahang': 
    #elif selectStates_3 == 'Kedah':
    if selectStates_3 == 'Johor':
        q4johor.q4_johor()
    #else: