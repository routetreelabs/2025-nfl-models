#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os

st.title("FanDuel NFL Over/Under Predictions")

# Display past records
st.markdown("""
**Model Weekly Record:**
- Week 1: 9–7 ✅
- Week 2: 10–6 ✅
- Week 3: 8–8 ➖
- Week 4: 8–8 ➖
- Week 5: 4–10 ❌
- Week 6: 9–6 ✅
- Week 7: 7–8 ❌
- Week 8: 8–5 ✅
""")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# SET CURRENT WEEK
current_week = 9  # Change this to the latest week when updating

# Load predictions for the current week
pred_file = f"week{current_week}_2025_predictions_fd.csv"
pred_path = os.path.join(BASE_DIR, pred_file)
preds = pd.read_csv(pred_path)

# Display predictions and neighbors
for i, row in preds.iterrows():
    st.markdown(
        f"**{row['Game']}** | Spread: {row['Spread']:.1f} | Total: {row['Total']:.1f} "
        f"| **Prediction:** {row['Prediction']}"
    )
    st.write(
        f"Confidence %: {row['ConfidencePercent']*100:.1f}% "
        f"| Avg Distance: {row['AvgDistance']} "
        f"| Score: {row['ConfidenceScore']:.3f}"
    )

    neighbors_file = f"neighbors_{i+1}_week{current_week}_fd.csv"
    neighbors_path = os.path.join(BASE_DIR, neighbors_file)

    if os.path.exists(neighbors_path):
        neighbors = pd.read_csv(neighbors_path)
        st.dataframe(neighbors.round(3))
    else:
        st.warning(f"Neighbors file not found: {neighbors_file}")
