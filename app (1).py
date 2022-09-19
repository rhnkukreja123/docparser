# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:29:28 2022

@author: bhati
"""

import streamlit as st


def main_page():
    st.write("# Welcome to QA Model 👋")
    st.sidebar.markdown("# Upload a document")
    st.file_uploader('Choose your .pdf file', type="pdf",accept_multiple_files=True)
    st.selectbox('Select a database where you want to store your files',('Database1', 'Database2'))
    st.markdown("Add a corpus to the existing database")
    st.button("Click to add corpus")
    

def page2():
    st.write("# Ask a Question")
    st.text_input("Ask a Question:")
    st.selectbox('Select your database',('Database1', 'Database2'))
    
    st.sidebar.markdown("# Ask a Question")
    if st.button("Click to get answer"):
        st.header("Answer")
        st.write("answer: QA model","score:0.8999")
        




page_names_to_funcs = {
    "Upload a document": main_page,
    "Ask a Question": page2
    
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()