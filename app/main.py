from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import fetch_archive_from_http
import os
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from pprint import pprint
import streamlit as st
#from utils.openai_api import query, process_question
import importlib

report = importlib.import_module("../report")



#load
from dotenv import load_dotenv

load_dotenv('../.env')


@st.cache_data()
def engine():
    document_store = InMemoryDocumentStore(use_bm25=True)

    doc_dir = "./texts"

    files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
    indexing_pipeline = TextIndexingPipeline(document_store)
    indexing_pipeline.run_batch(file_paths=files_to_index)

    retriever = BM25Retriever(document_store=document_store)
    reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
    pipe = ExtractiveQAPipeline(reader, retriever)

    return pipe


pipe = engine()

# Title of the App

st.title("Lawyer's Assistant")


# Text Area input
st.subheader("Enter the text below")
query = st.text_area("Enter Text", "Describe your case here in full details")

# Button
btn = st.button("Generate ")

# Condition
if btn and query:
    
    prediction = pipe.run(
        query=f"{query}",
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}},
    )
    st.header("Search Engine Retrieval")
    st.write(prediction["documents"][0].content)


    st.header("Report")
    st.write(report.text)
