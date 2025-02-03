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

load_dotenv()

try:
    loader = PyPDFLoader("yolov9_paper.pdf")  # Replace with your PDF path
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Verify model name
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None) # Verify model name

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

except Exception as e:
    st.error(f"Error during initialization: {e}")
    st.stop()

st.title("EL Fayrouz")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:
    st.write(f"**You:** {entry['prompt']}")
    st.write(f"**Bot:** {entry['answer']}")
    st.write("---")

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

query = st.text_input("Enter your question:", value=st.session_state.user_input, key="user_question")  # Add key

# JavaScript to trigger button click on Enter
st.markdown(
    """
    <script>
    const input = document.getElementById('user_question');
    input.addEventListener('keydown', function(event) {
        if (event.key === 'Enter') {
            const submitButton = document.querySelector('button');
            submitButton.click();
            event.preventDefault(); // Prevent default form submission
        }
    });
    </script>
    """,
    unsafe_allow_html=True,
)

if st.button("Submit"):  # Keep the button for users without JavaScript
    if query:
        try:
            with st.spinner("Generating response..."):
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                st.session_state.chat_history.append({"prompt": query, "answer": answer})
                st.session_state.user_input = ""
                st.rerun()

        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question.")
