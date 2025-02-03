from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma  # Correct import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import List

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Initialize document loader and vector store
loader = PyPDFLoader("yolov9_paper.pdf")  # Make sure this file exists in the correct location
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
docs = text_splitter.split_documents(data)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") # Define embeddings separately
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings # Use the embeddings instance here
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

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

# Pydantic models for request and response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    prompt: str
    answer: str

# Chat history storage
chat_history: List[dict] = []

# Chat endpoint
@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    query = request.query

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})

    if "answer" not in response:
        raise HTTPException(status_code=500, detail="Failed to generate a response.")

    # Store the query (prompt) and the answer in chat history
    chat_history.append({
        'prompt': query,
        'answer': response["answer"]
    })

    return QueryResponse(prompt=query, answer=response["answer"])

# Debugging endpoint for chat history
@app.get("/chat_history", response_model=List[QueryResponse])
async def get_chat_history():
    # Return all stored queries (prompts) and their answers from chat history
    return [QueryResponse(prompt=entry['prompt'], answer=entry['answer']) for entry in chat_history]