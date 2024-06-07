import streamlit as st
from streamlit_chat import message
import tempfile
from langchain_community.embeddings import SentenceTransformerEmbeddings
#from transformers import HuggingFaceEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import pandas as pd



DB_FAISS_PATH = "vectorstore/db_faiss"

# Load the model
def load_llm():
    llm = CTransformers(
        model = "llama2.bin",
        model_type = "llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm


def main():
    st.title("Talk to your Data")
    uploaded_file = st.sidebar.file_uploader("Upload Your Database", type="csv")

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = CSVLoader(file_path=tmp_file_path , encoding="utf-8" , csv_args={
            "delimiter": ","
        })
        data = loader.load()
        #st.json(data)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2" , model_kwargs= {'device' : 'cpu'})
        db = FAISS.from_documents(data , embeddings )
        db.save_local(DB_FAISS_PATH)
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm= llm, retriever=db.as_retriever())

        def conversation(query):
            response = chain({"question": query , "chat_history": st.session_state["history"]})
            st.session_state["history"].append(query , response["answer"])
            return response["answer"]
        
        if "history" not in st.session_state:
            st.session_state["history"] = []

        if "generated" not in st.session_state:
            st.session_state["generated"] = ["Ask me anything about "+ uploaded_file.name]

        if "past" not in st.session_state:
            st.session_state["past"] = ["Hello! :wave: "]

        #container for chat history
        response_container = st.container()
        container = st.container()
        
        with container:
            with st.form(key="my_form" , clear_on_submit=True):
                user_input = st.text_input("Query: " , placeholder= "Talk to Data" , key="input")
                submit_button = st.form_submit_button(label="Submit")
                
            if submit_button and user_input:
                output = conversation(user_input)
                st.session_state["past"].append(user_input)
                st.session_state["generated"].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state["generated"])):
                    message(st.session_state["past"][i] , is_user=True , key=str(i) + "_user")
                    message(st.session_state["generated"][i] , is_user=False , key=str(i) + "_bot")





if __name__ == "__main__":
    main()

