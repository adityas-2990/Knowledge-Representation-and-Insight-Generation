import streamlit as st
import pandas as pd
import os
import sys


def main():
    st.title("Talk to your Data")
    uploaded_file = st.sidebar.file_uploader("Upload Your Database", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df)

    query = st.text_input("Enter your query")




if __name__ == "__main__":
    main()

