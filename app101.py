import streamlit as st
skepticism_ratio = st.slider(
    "Select skepticism ratio",
    min_value=0.1,
    max_value=0.7,
    value=0.2,
    step=0.1
)
st.write("Your chosen value is", skepticism_ratio)