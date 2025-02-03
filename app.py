import streamlit as st
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import List

# ... (rest of the initialization code remains the same)

# Streamlit app
st.title("PDF Chatbot")

# Chat history (using Streamlit's session state)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history first
for entry in st.session_state.chat_history:
    st.write(f"**You:** {entry['prompt']}")
    st.write(f"**Bot:** {entry['answer']}")
    st.write("---")  # Separator between messages

# Input area for the user's query (moved to the bottom)
query = st.text_input("Enter your question:")

# Button to submit the query
if st.button("Submit"):
    if query:
        try:
            with st.spinner("Generating response..."):  # Show a spinner while processing
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                st.session_state.chat_history.append({"prompt": query, "answer": answer})

                # Refresh the chat display by rerunning the script
                st.experimental_rerun() 

        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question.")
