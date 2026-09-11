import os
import tempfile
import streamlit as st

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocuMind AI | PDF Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Caching & Document Processing Functions
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def process_pdf(uploaded_file, embeddings_model):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)

        vector_db = FAISS.from_documents(chunks, embeddings_model)
        return vector_db, len(docs), len(chunks)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------------
# State Management
# ---------------------------------------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_doc_name" not in st.session_state:
    st.session_state.processed_doc_name = None

embeddings_model = load_embedding_model()

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configuration")
    
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your GSK API key")
    else:
        st.success("Groq API Key Detected", icon="🔒")

    st.divider()

    with st.expander("🛠️ Advanced Settings"):
        selected_model = st.selectbox(
            "LLM Model",
            options=[
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-120b",
                "llama-3.1-8b-instant",
                "qwen/qwen3-32b"
            ],
            index=0
        )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("📄 DocuMind AI")
st.caption("Upload your PDF document to perform semantic vector search and get instant answers.")

# Main Upload Area
uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

if uploaded_file and (st.session_state.processed_doc_name != uploaded_file.name):
    with st.spinner("Processing PDF and generating vector embeddings..."):
        try:
            v_db, total_pages, total_chunks = process_pdf(uploaded_file, embeddings_model)
            st.session_state.vector_db = v_db
            st.session_state.processed_doc_name = uploaded_file.name
            st.session_state.doc_stats = {"pages": total_pages, "chunks": total_chunks}
            st.success(f"Successfully processed: {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# Display Metrics
if st.session_state.vector_db:
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Total Pages", st.session_state.doc_stats["pages"])
    col2.metric("Indexed Vector Chunks", st.session_state.doc_stats["chunks"])
    st.divider()

# Chat Conversation Area
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask a question about your uploaded document...")

if user_query:
    if not groq_api_key:
        st.error("Please set your Groq API Key in the sidebar.")
        st.stop()
        
    if not st.session_state.vector_db:
        st.warning("Please upload a PDF document first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Searching document..."):
            docs = st.session_state.vector_db.similarity_search(user_query, k=4)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])

            system_prompt = (
                "You are an assistant for question answering tasks. "
                "Use the following retrieved context to answer the question accurately. "
                "If the answer is not in the context, state that you cannot find it in the document.\n\n"
                f"CONTEXT:\n{context}"
            )

            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2,
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                
                with st.expander("🔍 View Context Sources"):
                    for idx, doc in enumerate(docs):
                        st.caption(f"**Chunk {idx+1}:** {doc.page_content[:250]}...")

                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Groq API Error: {str(e)}")
