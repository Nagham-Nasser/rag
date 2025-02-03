import streamlit as st
# ... (other imports)

# Initialize variables OUTSIDE the try block.  Give them a default value.
rag_chain = None  # Initialize rag_chain to None
initialization_error = None # To store any error during initialization

try:
    # ... (your PDF loading, splitting, embedding, etc. code)

    question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

except Exception as e:
    initialization_error = f"Error during initialization: {e}"  # Store the error message
    st.error(initialization_error) # Display the error immediately
    #st.stop() # You might want to stop here if initialization is critical.


# Streamlit app
st.title("PDF Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for entry in st.session_state.chat_history:
    st.write(f"**You:** {entry['prompt']}")
    st.write(f"**Bot:** {entry['answer']}")
    st.write("---")

query = st.text_input("Enter your question:")

if st.button("Submit"):
    if query:
        if rag_chain: # Check if rag_chain was successfully initialized
            try:
                with st.spinner("Generating response..."):
                    response = rag_chain.invoke({"input": query})
                    answer = response["answer"]
                    st.session_state.chat_history.append({"prompt": query, "answer": answer})
                    st.rerun()
            except Exception as e:
                st.error(f"Error generating response: {e}")  # Handle query errors separately
        else:
            st.error("Chatbot initialization failed. Please check the error message above.") # Inform the user about the initialization failure

    else:
        st.warning("Please enter a question.")
