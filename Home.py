#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st

st.set_page_config(page_title="NFL Models – 2025", layout="wide")

st.title("NFL Prediction Models – 2025")
st.write("""
Welcome! Choose a model from the left sidebar:

- **Moneyline Predictions** (Logistic Regression)
- **Over/Under Predictions** (K-Nearest Neighbors)
- **Points Predictor** (Linear Regression)
- **Against the Spread** (Logistic Regression)
""")

