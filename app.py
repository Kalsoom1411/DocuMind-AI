import os
import time
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
    page_title="DocuMind AI | Advanced PDF RAG",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Hardcoded Model
DEFAULT_MODEL = "openai/gpt-oss-120b"

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

        # Text Splitting with Metadata Preservation (Page Numbers)
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
if "preset_query" not in st.session_state:
    st.session_state.preset_query = None

embeddings_model = load_embedding_model()

# ---------------------------------------------------------
# Sidebar (Analytics & Export)
# ---------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Workspace")
    
    groq_api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_api_key:
        groq_api_key = st.text_input("Groq API Key", type="password", help="Enter your GSK API key")
    else:
        st.success("Groq API Active", icon="🔒")

    st.divider()

    # Chat Export Option
    if st.session_state.chat_history:
        st.markdown("### 📥 Export Conversation")
        formatted_chat = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button(
            label="Download Chat (.txt)",
            data=formatted_chat,
            file_name="documind_chat_history.txt",
            mime="text/plain",
            use_container_width=True
        )

    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------
# Main UI
# ---------------------------------------------------------
st.title("📄 DocuMind AI")
st.caption("Upload your PDF to enable fast semantic search, citation metadata, and real-time streaming answers.")

# Upload Container
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

# Dashboard Summary & Quick Actions
if st.session_state.vector_db:
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Total Pages", st.session_state.doc_stats["pages"])
    col2.metric("Indexed Chunks", st.session_state.doc_stats["chunks"])
    
    st.markdown("**💡 Quick Prompt Suggestions:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    if q_col1.button("📋 Summarize Document", use_container_width=True):
        st.session_state.preset_query = "Provide a comprehensive summary of this entire document with bullet points."
    if q_col2.button("🔑 Key Takeaways", use_container_width=True):
        st.session_state.preset_query = "What are the top 5 key takeaways or main findings in this document?"
    if q_col3.button("❓ Important Questions", use_container_width=True):
        st.session_state.preset_query = "What are 3 important questions answered by this document?"
    st.divider()

# Display Chat Conversation
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle preset queries from buttons or manual text input
user_input = st.chat_input("Ask a question about your document...")
active_query = st.session_state.preset_query or user_input

if active_query:
    # Reset preset state
    st.session_state.preset_query = None

    if not groq_api_key:
        st.error("Please set your Groq API Key in the sidebar.")
        st.stop()
        
    if not st.session_state.vector_db:
        st.warning("Please upload a PDF document first.")
        st.stop()

    st.session_state.chat_history.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    with st.chat_message("assistant"):
        # 1. Vector Search
        docs = st.session_state.vector_db.similarity_search(active_query, k=4)
        
        # Build Context with Page Numbers
        context_blocks = []
        for doc in docs:
            page_num = doc.metadata.get("page", 0) + 1
            context_blocks.append(f"[Source Page {page_num}]:\n{doc.page_content}")
        context = "\n\n---\n\n".join(context_blocks)

        system_prompt = (
            "You are an expert document assistant. "
            "Use the provided context chunks below to answer the user's question. "
            "If the answer is not present in the context, explicitly state that the information is not in the document.\n\n"
            f"CONTEXT FROM PDF:\n{context}"
        )

        try:
            client = Groq(api_key=groq_api_key)
            
            # Measure Latency
            start_time = time.time()
            
            # 2. Streaming Completion Call
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": active_query}
                ],
                temperature=0.2,
                stream=True
            )

            # 3. Dynamic Token Stream Rendering
            response_container = st.empty()
            full_response = ""
            
            for chunk in stream:
                content = chunk.choices[0].delta.content or ""
                full_response += content
                response_container.markdown(full_response + "▌")
            
            response_container.markdown(full_response)
            
            end_time = time.time()
            elapsed_time = round(end_time - start_time, 2)

            # Display Performance Metric
            st.caption(f"⚡ *Response generated in {elapsed_time}s using {DEFAULT_MODEL}*")

            # 4. Citations & Sources Expander
            with st.expander("🔍 View Context Sources & Page Citations"):
                for idx, doc in enumerate(docs):
                    page_num = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Chunk {idx+1} (Page {page_num}):**")
                    st.caption(f"{doc.page_content[:300]}...")

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"Groq API Error: {str(e)}")
