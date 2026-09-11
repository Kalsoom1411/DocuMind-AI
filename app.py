import os
import tempfile
import streamlit as st

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------
# Page Configuration & Professional Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="DocuMind RAG | Enterprise PDF Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Dark Theme Aesthetics
st.markdown("""
<style>
    /* Main App Background and Base Font */
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Container Styling */
    .main-header {
        background: linear-gradient(135deg, #1E2640 0%, #0F172A 100%);
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #1E293B;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .main-header h1 {
        color: #38BDF8;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: #94A3B8;
        font-size: 1.05rem;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #21262D;
    }

    /* Chat Bubbles */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
    }
    
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 10px;
    }

    /* Custom Metric Cards */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8B949E;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Caching Functions (Optimized Processing)
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Loads open-source Hugging Face embedding model locally."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def process_pdf(uploaded_file, embeddings_model):
    """Processes PDF file: extracts text, chunks it, and builds a FAISS index."""
    # Write uploaded stream to temporary local file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Extract Text
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # 2. Chunking
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(docs)

        # 3. Create Vectors & Store in FAISS
        vector_db = FAISS.from_documents(chunks, embeddings_model)
        
        return vector_db, len(docs), len(chunks)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------------
# Sidebar Setup & API Management
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/brainstorm-skill.png", width=64)
    st.title("Configuration")
    
    # Check for Groq API key in Secrets or User Input
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your GSK API key from Groq Console")
        st.caption("Get your key at [console.groq.com](https://console.groq.com)")
    else:
        st.success("API Key detected securely!", icon="🔒")

    st.divider()
    
    # Model Selector
    model_option = st.selectbox(
        "Select LLM Engine",
        options=[
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen/qwen3-32b"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("### Document Pipeline")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

# Initialize Global Embedding Model
embeddings_model = load_embedding_model()

# Session State Initialization
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_doc_name" not in st.session_state:
    st.session_state.processed_doc_name = None

# Process Document Action
if uploaded_file and (st.session_state.processed_doc_name != uploaded_file.name):
    with st.spinner("Processing document (Extracting, Chunking & Embedding)..."):
        try:
            v_db, total_pages, total_chunks = process_pdf(uploaded_file, embeddings_model)
            st.session_state.vector_db = v_db
            st.session_state.processed_doc_name = uploaded_file.name
            st.session_state.doc_stats = {"pages": total_pages, "chunks": total_chunks}
            st.toast("PDF successfully indexed in Vector DB!", icon="✅")
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

# ---------------------------------------------------------
# Main Interface Layout
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Enterprise Knowledge Assistant</h1>
    <p>Upload your PDF document to perform semantic vector search and generate answers using open-source models via Groq.</p>
</div>
""", unsafe_allow_html=True)

# Document Details Section
if st.session_state.vector_db:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.processed_doc_name}</div><div class="metric-label">Active Document</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.doc_stats["pages"]}</div><div class="metric-label">Total Pages</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.doc_stats["chunks"]}</div><div class="metric-label">Vector Chunks</div></div>', unsafe_allow_html=True)
    st.divider()

# Display Chat Messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input & RAG Execution
user_query = st.chat_input("Ask any question about your document...")

if user_query:
    if not groq_api_key:
        st.warning("Please provide a Groq API key in the sidebar to proceed.")
        st.stop()
        
    if not st.session_state.vector_db:
        st.warning("Please upload a PDF document first.")
        st.stop()

    # Append user question
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Perform RAG Pipeline
    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant context..."):
            # 1. Similarity Search in FAISS
            docs = st.session_state.vector_db.similarity_search(user_query, k=4)
            context = "\n\n---\n\n".join([doc.page_content for doc in docs])

            # 2. Construct Prompt
            system_prompt = (
                "You are an expert document analysis assistant. Answer the user's question "
                "strictly using only the provided context. If the answer is not contained "
                "within the context, state clearly that you cannot find the answer in the document.\n\n"
                f"CONTEXT FROM PDF:\n{context}"
            )

            # 3. Call Groq API using Native SDK Client
            try:
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model=model_option,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.2,
                )
                
                answer = response.choices[0].message.content
                st.markdown(answer)
                
                # Expandable Source Viewer
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for idx, doc in enumerate(docs):
                        st.markdown(f"**Chunk {idx+1}:**")
                        st.caption(doc.page_content)

                # Save assistant history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Groq API Error: {str(e)}")
