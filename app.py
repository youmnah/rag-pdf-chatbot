import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# App title
st.set_page_config(page_title="PDF Assistant", page_icon="🤖")
st.title("📄 Chat with your PDF")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# API Key input in sidebar
with st.sidebar:
    st.header("🔑 OpenAI API Key")
    api_key = st.text_input(
        "Enter your OpenAI API key",
        type="password",
        placeholder="sk-..."
    )
    st.caption("Your key is never stored or shared. [Get a key](https://platform.openai.com/api-keys)")
    
    if not api_key:
        st.warning("Enter your API key to use the app")
        st.stop()
    
    os.environ["OPENAI_API_KEY"] = api_key

# Sidebar for PDF upload
with st.sidebar:
    st.header("Choose your PDF file")
    uploaded_file = st.file_uploader("Click on Upload", type="pdf")

    if uploaded_file is not None:
        file_path = os.path.join("docs", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF..."):
            # Load PDF
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Create vector store
            embeddings = OpenAIEmbeddings()
            st.session_state.vectorstore = Chroma.from_documents(chunks, embeddings)

        st.success(f"✅ {uploaded_file.name} processed!")

# Chat interface
if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started")
else:
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    user_question = st.chat_input("Ask something about your PDF...")

    if user_question:
        # Display user message
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve relevant chunks
                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 3}
                )
                docs = retriever.invoke(user_question)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Build prompt with history
                history_text = ""
                for msg in st.session_state.chat_history[-6:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['content']}\n"

                prompt = ChatPromptTemplate.from_messages([
                    ("system", """You are a helpful assistant that answers questions 
                    based on the provided document context. Always base your answers 
                    on the context. If the answer is not in the context, say so.
                    
                    Context:
                    {context}
                    
                    Previous conversation:
                    {history}"""),
                    ("human", "{question}")
                ])

                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                chain = prompt | llm | StrOutputParser()

                answer = chain.invoke({
                    "context": context,
                    "history": history_text,
                    "question": user_question
                })

                st.write(answer)

                # Show sources
                with st.expander("📚 Sources"):
                    for i, doc in enumerate(docs):
                        st.write(f"**Chunk {i+1}** — Page {doc.metadata.get('page', 'N/A')}")
                        st.write(doc.page_content[:300] + "...")

        # Save to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})