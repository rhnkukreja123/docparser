# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 14:46:24 2022

@author: bhati
"""
import os
import streamlit as st
import pdfplumber
#!pip install pdfplumber

import streamlit as st
import haystack
import PyPDF2
from haystack.preprocessor.preprocessor import PreProcessor
import logging
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack.utils import clean_wiki_text, convert_files_to_docs, fetch_archive_from_http, print_answers
from haystack.nodes import FARMReader, TransformersReader
import base64
import os
#launching elasticsearch
from haystack.utils import launch_es
launch_es()

from haystack.document_stores import ElasticsearchDocumentStore

from haystack.nodes import DensePassageRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline

import PyPDF2
import re
import glob
import pandas as pd
from pprint import pprint
import requests
from elasticsearch import Elasticsearch
from elasticsearch import helpers
import streamlit as st

class Document(object):
    
    def __init__(self,pdf_file):
        self.pdf_file = pdf_file
    
        
    def extract_content(pdf_file):
        this_loc=1
        df = pd.DataFrame(columns =('name','content'))
        path = r"C:\Users\bhati\Downloads\pdfs"
        #pdfobj = open(f'{pdf_file}','rb') 
        #pdfobj = open(f'{pdf_file.name}','rb')
        print("********")
        print(path + "\\"+ pdf_file.name)
        print("********")
        pdfobj = open(path + "\\"+ pdf_file.name,'rb')
        base64_pdf = PyPDF2.PdfFileReader(pdfobj)
        this_doc=''
        for page in range(base64_pdf.numPages):
          pageObj = base64_pdf.getPage(page)
          text = pageObj.extractText()
          this_doc+=text
          df.loc[this_loc]=pdf_file.name,this_doc
          this_loc=this_loc+1
          #return df   
        df_index=df.to_dict('records')                
        preprocessor = PreProcessor(
            clean_empty_lines=True,
            clean_whitespace=True,
            clean_header_footer=False,
            split_by="word",
            split_length=100,
            split_respect_sentence_boundary=True)
        
        preprocessed_docs = preprocessor.process(df_index)
        return preprocessed_docs
    

class Corpus(Document):
    def __init__(self):
        pass
    
    def new_corpus(self,name,preprocessed_docs):
        self.name = name
        self.preprocessed_docs = preprocessed_docs
        document_store_new = ElasticsearchDocumentStore(host="localhost", index=f'{name}',similarity="dot_product")                
        #document_store_new.delete_documents()
        document_store_new.write_documents(preprocessed_docs)
        return document_store_new
        
    def existing_elastic_db_store1(self,preprocessed_docs):
        self.preprocessed_docs = preprocessed_docs
        document_store = ElasticsearchDocumentStore(host="localhost", index='database1',similarity="dot_product")                
        #document_store.delete_documents()
        document_store.write_documents(preprocessed_docs)
        return document_store
    
    def existing_elastic_db_store2(self,preprocessed_docs):
        self.preprocessed_docs = preprocessed_docs
        document_store = ElasticsearchDocumentStore(host="localhost", index='database2',similarity="dot_product")                
        #document_store.delete_documents()
        document_store.write_documents(preprocessed_docs)
        return document_store



class Query:
    def __init__(self):
        pass
    
    def retrieval(self,document_store):
        self.document_store=document_store
        retriever_es = DensePassageRetriever(document_store)
        document_store.update_embeddings(
            retriever=retriever_es)
        reader = FARMReader(model_name_or_path="distilbert-base-uncased-distilled-squad", use_gpu=True, num_processes=0)
        pipe = ExtractiveQAPipeline(reader, retriever_es)
        return pipe
    
      
    def print_answer(results):
        fields = ["answer", "score"]  # "context",
        answers = results["answers"]
        filtered_answers = []
        
        for ans in answers:
            filtered_ans = {
                field: getattr(ans, field)
                for field in fields
                if getattr(ans, field) is not None
            }
            filtered_answers.append(filtered_ans)

        return filtered_answers
      
    def ask_ques(self,question,pipe):    
        self.question = question
        self.pipe = pipe
        
        result = pipe.run(query=question, params={
            "Retriever": {"top_k": 10}, "Reader": {"top_k": 1}})
        ans = Query.print_answer(result)
        return ans
       
        
    
# document_obj=Document.extract_content("C:/Users/bhati/Downloads/pdfs/document.pdf")
# document_obj

# corpus_obj = Corpus()

# corpus_obj2 = corpus_obj.new_corpus('doc1', document_obj)

# corpus_obj3 = corpus_obj.existing_elastic_db_store(document_obj)
    
# query_obj = Query().retrieval(corpus_obj3)

# ans_obj = Query().ask_ques("What is Machine Learning", query_obj)


###################################################################################
class Driver:
    def __init__(self):
        pass
    
    def upload_a_file_into_new_db(self,name,pdf_file):
        
        self.pdf_file=pdf_file
        self.name=name

        document_obj=Document.extract_content(pdf_file)
        corpus_obj=Corpus().new_corpus(f'{name}', document_obj)
        return corpus_obj
    
    def upload_a_file_into_existing_db1(self,pdf_file):
        
        self.pdf_file=pdf_file
        
        document_obj=Document.extract_content(pdf_file)
        corpus_obj=Corpus().existing_elastic_db_store1(document_obj)
        return corpus_obj
    
    def upload_a_file_into_existing_db2(self,pdf_file):
        
        self.pdf_file=pdf_file
        
        document_obj=Document.extract_content(pdf_file)
        corpus_obj=Corpus().existing_elastic_db_store2(document_obj)
        return corpus_obj
    
    
    def answers(self,question,corpus_obj):
        
        self.question=question
        self.corpus_obj=corpus_obj

        query_obj = Query().retrieval(corpus_obj)
        ans_obj = Query().ask_ques(question, query_obj)
        return ans_obj
    
#######################
import streamlit as st


def main_page():
    st.write("# Welcome to QA Model 👋")
    st.sidebar.markdown("# Upload a document")
    pdf_file=st.file_uploader('Choose your .pdf file', type="pdf")
    name = st.text_input("Enter index name:")
    selection = st.selectbox('Select a database where you want to store your files',('Database1','Database2'))
    st.markdown("Add a corpus to the existing database")
    st.markdown("Add a new corpus to the database")
    #st.button("Click to add corpus")
    

def page2():
    pdf_file=st.file_uploader('Choose your .pdf file', type="pdf")    
    st.write("# Ask a Question")
    question = st.text_input("Ask a Question:")
    selection = st.selectbox('Select your database',('Database1','Database2'))
    
    st.sidebar.markdown("# Ask a Question")
    if selection == "Database1":
        if st.button("Click to get answer"):
            #st.header("Answer")
            if pdf_file is not None:
                files = Driver().upload_a_file_into_existing_db1(pdf_file)
                get_answer = Driver().answers(question,files)
                st.write(get_answer)
                st.write("saved in database1!")
                
            else:
                st.write("none")
                
    if selection == "Database2":
        if st.button("Click to get answer"):
            #st.header("Answer")
            if pdf_file is not None:
                files = Driver().upload_a_file_into_existing_db2(pdf_file)
                get_answer = Driver().answers(question,files)
                st.write(get_answer)
                st.write("saved in database2!")
                
            else:
                st.write("none")
        # files = Driver().upload_a_file_into_existing_db2(pdf_file)
        # get_answer = Driver().answers(question,files)
        # st.write(get_answer)
        # st.write("saved in database2!")
        
    
                
                
                
page_names_to_funcs = {
    "Upload a document": main_page,
    "Ask a Question": page2
    
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()

