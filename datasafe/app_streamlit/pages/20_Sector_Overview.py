import streamlit as st
from datasafe.storage.sqlite_store import query_recent

st.title("Sector Overview")

sector = st.selectbox("Sector", ["Media","Finance","Healthcare","Retail","Telecoms","Energy","Other"], index=0)
severity = st.selectbox("Severity", ["All","Critical","High","Medium","Low"], index=0)
min_risk = st.slider("Min Risk", 0, 100, 30)
limit = st.number_input("Limit", 10, 500, 200)

items = query_recent(limit=int(limit), sector=None if sector=="All" else sector, severity=None if severity=="All" else severity, min_risk=min_risk)

st.subheader("Threats")
st.dataframe(items)


