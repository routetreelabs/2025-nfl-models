#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os

st.title("DraftKings NFL Over/Under Predictions - KNN")
# Display past records
st.markdown("""
**Model Weekly Record:**
- Week 1: 11–5 ✅
- Week 2: 12–4 ✅
- Week 3: 6–10 ❌
- Week 4: 5–11 ❌
- Week 5: 6–8 ❌
- Week 6: 9–6 ✅
- Week 7: 9–6 ✅
- Week 8: 6–7 ❌
- Week 9: 7–7 ➖
- Week 10: 8–6 ✅
- Week 11: 9–6 ✅
- Week 12: 8–6 ✅
- Week 13: 9–7 ✅
- Week 14: 6–8 ❌
- Week 15: 9–7 ✅
- Week 16: 13–3 ✅
- Week 17: 9–7 ✅
""")

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# SET CURRENT WEEK HERE
current_week = 17

# Load DK predictions file
pred_file = f"week{current_week}_2025_predictions_dk.csv"
preds = pd.read_csv(os.path.join(BASE_DIR, pred_file))

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

    neighbors_file = f"neighbors_{i+1}_week{current_week}_dk.csv"
    neighbors_path = os.path.join(BASE_DIR, neighbors_file)

    if os.path.exists(neighbors_path):
        neighbors = pd.read_csv(neighbors_path)
        st.dataframe(neighbors.round(3))
    else:
        st.warning(f"Neighbors file not found: {neighbors_file}")

