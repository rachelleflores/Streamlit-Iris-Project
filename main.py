import pandas as pd
import numpy as np
import pickle
from sklearn import datasets

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
import streamlit as st

st.title("My First App")

st.subheader("Everything works so far")

st.info("Testing")

st.write("Hoping to use this for the final project")

with st.columns(10)[9]:
    st.button('soft/light')

#st.set_page_config(layout=)
col1, col2, col3 = st.columns([8, 4, 8])

