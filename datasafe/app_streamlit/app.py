import streamlit as st

st.set_page_config(page_title="DataSafe", layout="wide")
st.sidebar.page_link("pages/10_Media_Risks.py", label="Media — Active Risks")
st.title("DataSafe")
st.write("Use the sidebar to open Media threat view.")
import streamlit as st

st.set_page_config(page_title="DataSafe", layout="wide")

st.sidebar.title("DataSafe")
st.sidebar.write("Navigation")

st.write("Welcome to DataSafe. Use the pages on the left to explore risks.")


