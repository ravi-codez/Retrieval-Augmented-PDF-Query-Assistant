import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import PyPDF2
import os
import openai

# --- Load environment variables ---
load_dotenv()

# --- Streamlit UI ---
st.title("Chat with your PDF")
st.sidebar.title("Settings")

api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    openai.api_key = api_key

    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        # --- Check if embeddings are already stored ---
        if "vectorstore" not in st.session_state:

            # Save uploaded file temporarily
            temp_pdf = "./temp.pdf"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Extract text from PDF
            reader = PyPDF2.PdfReader(temp_pdf)
            text = []
            for page in reader.pages:
                content = page.extract_text()
                if content:
                    text.append(content)

            # Split text into chunks
            docs = [Document(page_content=t) for t in text]
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)

            # Create embeddings once
            st.write("🔄 Creating embeddings")
            embeddings = OpenAIEmbeddings()

            # Store embeddings in a persistent local Chroma DB
            persist_dir = "./chroma_store"
            vectorstore = Chroma.from_documents(splits, embeddings, persist_directory=persist_dir)
            st.session_state.vectorstore = vectorstore
            st.sidebar.success("✅ PDF processed and embeddings stored!")
        else:
            st.sidebar.info("✅ Using existing embeddings from session.")
            vectorstore = st.session_state.vectorstore

        # Create retriever
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k":3, "lambda_mult":0.25})

        # Initialize chat model
        if "model" not in st.session_state:
            st.session_state.model = ChatOpenAI(model="gpt-3.5-turbo")
        model = st.session_state.model

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Prompt template
        prompt_template = PromptTemplate(
            template="""
            You are a helpful assistant that answers questions based on the given context.
            Use only the information from the context to answer the user's question.
            If the answer is not found in the context, say "I don't know."

            Context:
            {context}

            Question:
            {query}
            """,
            input_variables=["context", "query"]
        )

        # Input field for user query
        input_query = st.text_input("Ask any question about your PDF:")

        if input_query:
            retrieved_docs = retriever.invoke(input_query)
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            final_prompt = prompt_template.invoke({"context": context_text, "query": input_query})
            response = model.invoke(final_prompt)

            st.session_state.chat_history.append({"question": input_query, "answer": response.content})
            st.write(response.content)

            # Optional: display chat history
            with st.expander("🧾 Chat History"):
                for chat in st.session_state.chat_history:
                    st.write(f"**You:** {chat['question']}")
                    st.write(f"**Bot:** {chat['answer']}")

else:
    st.sidebar.warning("Please enter your OpenAI API key to continue.")
