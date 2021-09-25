import streamlit as st 
import numpy as np
import pandas as pd
import altair as alt 
from Question1 import Q1_case as case
from Question1 import Q1_test as test
from Question1 import Q1_death as death
from Question1 import Q1_population as population
from Question1 import Q1_PKRC as pkrc
from Question1 import Q1_Hospital as hospital
from Question1 import Q1_ICU as icu
from Question2 import Q2_corr as corr
from Question3 import Q3_Kedah as q3kedah
from Question3 import Q3_Pahang as q3pahang
from Question3 import Q3_Johor as q3johor
from Question3 import Q3_Selangor as q3selangor
from Question4 import Q4_Pahang as q4pahang
from Question4 import Q4_Johor as q4johor
from Question4 import Q4_kedah as q4kedah
from Question4 import Q4_Selangor as q4selangor

selectQuestion = st.sidebar.radio('Select a Question to View', ('Question 1','Question 2','Question 3','Question 4'))
st.sidebar.markdown('# ')
st.sidebar.markdown('# ')
st.sidebar.markdown('# ')
st.sidebar.markdown('** TDS 3301 Data Mining **')
st.sidebar.markdown('** Group Assignment **')
st.sidebar.markdown('Prepared by: ')
st.sidebar.markdown('Yap Mou En - 1191301106')
st.sidebar.markdown('Lim Ying Shen - 1191301089')
st.sidebar.markdown('Aw Yew Lim - 1171103827')

if selectQuestion == 'Question 1':
    st.markdown('## Question 1')
    st.markdown('### Discuss the exploratory data analysis steps you have conducted including detection of outliers and missing values?')
    selectDatasets = st.selectbox("Select Datasets", ['Cases','Tests','Deaths','Population','PKRC','Hospital','ICU'])
   
    if selectDatasets == 'Cases':
        case.q1_case()
    elif selectDatasets == 'Tests':
        test.q1_test()
    elif selectDatasets == 'Deaths':
        death.q1_death()
    elif selectDatasets == 'Population':
        population.q1_population()
    elif selectDatasets == 'PKRC':
        pkrc.q1_pkrc()
    elif selectDatasets == 'Hospital':
        hospital.q1_hospital()
    else:
        icu.q1_icu()

elif selectQuestion == 'Question 2':
    st.markdown('## Question 2')
    st.markdown('### What are the states that exhibit strong correlation with (i) Pahang, and (ii) Johor?')
    corr.q2_corr()

elif selectQuestion == 'Question 3':
    st.markdown('## Question 3')
    st.markdown('### What are the strong features/indicators to daily cases for (i) Pahang, (ii) Kedah, (iii) Johor, and (iv) Selangor?')
    selectStates_2 = st.selectbox("Select States", ['Pahang','Kedah','Johor','Selangor'])

    if selectStates_2 == 'Pahang': 
        q3pahang.corr_pahang()
    elif selectStates_2 == 'Kedah':
        q3kedah.lasso_kedah()
    elif selectStates_2 == 'Johor':
        q3johor.corr_johor()
    else:
        q3selangor.boruta_selangor()

else:
    st.markdown('## Question 4')
    st.markdown('### Comparing regression and classification models, what model performs well in predicting the daily cases for (i) Pahang, (ii) Kedah, (iii) Johor, and (iv) Selangor?')
    selectStates_3 = st.selectbox("Select States", ['Pahang','Kedah','Johor','Selangor'])

    if selectStates_3 == 'Pahang': 
        q4pahang.model_pahang()
    elif selectStates_3 == 'Kedah':
        q4kedah.model_kedah()
    elif selectStates_3 == 'Johor':
        q4johor.model_johor()   
    else:
        q4selangor.model_selangor()
  