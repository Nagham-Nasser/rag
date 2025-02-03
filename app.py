import streamlit as st
# ... (rest of your imports and initialization code)

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
        try:
            with st.spinner("Generating response..."):
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
                st.session_state.chat_history.append({"prompt": query, "answer": answer})
                st.rerun()  # Use st.rerun() instead of st.experimental_rerun()

        except Exception as e:
            st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a question.")
