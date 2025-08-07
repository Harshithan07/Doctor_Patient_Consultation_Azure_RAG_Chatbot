import os
import streamlit as st
from dotenv import load_dotenv
from pathlib import Path
import tempfile

from langchain.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import SystemMessage, HumanMessage

from fpdf import FPDF
import datetime
import unicodedata

# Import processing modules
from transcribe import transcribe_audio
from chunker import split_text_into_chunks
from embed_and_store import store_in_faiss

load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_loaded" not in st.session_state:
    st.session_state.index_loaded = False
if "current_index" not in st.session_state:
    st.session_state.current_index = "RES0215"


def clean_text(text: str) -> str:
    """Clean text for PDF generation"""
    text = unicodedata.normalize("NFKD", text)
    replacements = {
        '–': '-', '—': '-', '−': '-',
        ''': "'", ''': "'",
        '"': '"', '"': '"',
        '•': '-', '●': '-',
        '…': '...',
        '©': '(c)', '®': '(r)',
    }
    for src, target in replacements.items():
        text = text.replace(src, target)
    return text.encode("ascii", "ignore").decode("ascii")


def generate_pdf(title: str, diagnosis: str, medications: str, filename: str) -> str:
    """Generate PDF report"""
    pdf = FPDF()
    
    # Title Page
    pdf.add_page()
    pdf.set_font("Arial", 'B', 20)
    pdf.cell(0, 100, clean_text("Doctor-Patient Summary Report"), ln=True, align="C")
    pdf.set_font("Arial", size=14)
    pdf.cell(0, 10, clean_text(f"Date: {datetime.date.today().strftime('%B %d, %Y')}"), ln=True, align="C")
    
    # Diagnosis
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, clean_text("Diagnosis"), ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, clean_text(diagnosis))
    
    # Medications
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, clean_text("Medications"), ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, clean_text(medications))
    
    output_dir = Path("rag_chatbot")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    pdf.output(str(output_path))
    
    return str(output_path)


def get_azure_embeddings():
    """Create MiniLM embeddings instance"""
    from langchain.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )


def get_azure_llm():
    """Create Azure ChatOpenAI instance"""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        temperature=0
    )


def process_uploaded_file(uploaded_file):
    """Process uploaded audio file and create FAISS index"""
    with st.spinner("Processing audio file..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Step 1: Transcribe audio
            st.info("🎵 Transcribing audio...")
            transcript = transcribe_audio(tmp_file_path)
            
            if not transcript:
                st.error("❌ Transcription failed. Please check your audio file and API credentials.")
                return False
            
            # Step 2: Chunk the transcript
            st.info("🧩 Splitting transcript into chunks...")
            chunks = split_text_into_chunks(transcript, max_tokens=300)
            st.success(f"✅ Created {len(chunks)} chunks")
            
            # Step 3: Create embeddings and store
            st.info("💾 Creating embeddings and storing in FAISS...")
            index_name = uploaded_file.name.replace('.mp3', '').replace(' ', '_')
            store_in_faiss(chunks, index_dir=f"faiss_store/{index_name}")
            
            # Update session state
            st.session_state.current_index = index_name
            st.session_state.index_loaded = True
            
            st.success(f"✅ Successfully processed {uploaded_file.name}!")
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return False


@st.cache_resource
def load_vectorstore(index_name):
    """Load FAISS vectorstore for given index"""
    embeddings = get_azure_embeddings()
    index_path = Path(f"faiss_store/{index_name}")
    
    if not index_path.exists():
        return None
        
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True
    )


def process_query(query, index_name):
    """Process user query and return response"""
    vs = load_vectorstore(index_name)
    
    if vs is None:
        return {
            "type": "error",
            "answer": "❌ No index loaded. Please upload an audio file first.",
            "sources": []
        }
    
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    llm = get_azure_llm()
    
    if "generate report" in query.lower() or "download report" in query.lower():
        chunks = retriever.get_relevant_documents("patient summary, diagnosis and medications")
        context = "\n\n".join([doc.page_content for doc in chunks])
        prompt = f"""Based on the following doctor-patient conversation, generate:
- A clear diagnosis summary
- A list of medications prescribed

Conversation:\n{context}
"""
        messages = [
            SystemMessage(content="You are a medical assistant generating structured reports."),
            HumanMessage(content=prompt)
        ]
        response = llm(messages).content
        
        diagnosis = "Not found."
        meds = "Not found."
        
        if "Diagnosis" in response:
            parts = response.split("Medications")
            diagnosis = parts[0].replace("Diagnosis", "").strip()
            meds = parts[1].strip() if len(parts) > 1 else meds
        else:
            diagnosis = response
        
        pdf_path = generate_pdf("Patient Report", diagnosis, meds, filename="patient_report.pdf")
        
        return {
            "type": "report",
            "answer": "📄 Report generated successfully! Click below to download.",
            "pdf_path": pdf_path,
            "sources": chunks
        }
    else:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa_chain({"query": query})
        
        return {
            "type": "chat",
            "answer": result["result"],
            "sources": result["source_documents"]
        }


# Page configuration
st.set_page_config(page_title="Doctor Chatbot", page_icon="🩺", layout="wide")
st.title("🩺 Doctor Transcript Chatbot (RAG)")

# File upload section
if not st.session_state.index_loaded:
    st.markdown("### Upload Consultation Audio")
    st.info("Please upload an MP3 file of the doctor-patient consultation to begin.")
    
    uploaded_file = st.file_uploader(
        "Choose an MP3 file",
        type=['mp3'],
        help="Upload the consultation audio file in MP3 format"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"📎 Selected file: {uploaded_file.name}")
        with col2:
            if st.button("Process File", type="primary"):
                if process_uploaded_file(uploaded_file):
                    st.rerun()
else:
    # Show current loaded file
    st.success(f"✅ Currently loaded: {st.session_state.current_index}")
    
    # Chat interface
    st.caption("Ask anything about the consultation.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar=message.get("avatar")):
            st.markdown(message["content"])
            
            if message.get("pdf_path"):
                with open(message["pdf_path"], "rb") as f:
                    st.download_button(
                        "Download Report PDF", 
                        f, 
                        file_name="patient_report.pdf", 
                        mime="application/pdf",
                        key=f"download_{message.get('timestamp', '')}"
                    )
            
            if message.get("show_sources") and message.get("sources"):
                with st.expander("🔎 Source Chunks"):
                    for doc in message["sources"]:
                        st.markdown(doc.page_content)
                        st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Your question:"):
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "avatar": "🧑"
        })
        
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Thinking..."):
                response = process_query(prompt, st.session_state.current_index)
            
            st.markdown(response["answer"])
            
            message_data = {
                "role": "assistant",
                "content": response["answer"],
                "avatar": "🤖",
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            if response["type"] == "report":
                message_data["pdf_path"] = response["pdf_path"]
                with open(response["pdf_path"], "rb") as f:
                    st.download_button(
                        "Download Report PDF", 
                        f, 
                        file_name="patient_report.pdf", 
                        mime="application/pdf",
                        key=f"download_current"
                    )
            
            if response.get("sources"):
                message_data["sources"] = response["sources"]
                message_data["show_sources"] = True
                
                with st.expander("🔎 Source Chunks"):
                    for doc in response["sources"]:
                        st.markdown(doc.page_content)
                        st.markdown("---")
            
            st.session_state.messages.append(message_data)

# Sidebar
with st.sidebar:
    st.header("Chat Controls")
    
    if st.session_state.index_loaded:
        if st.button("Upload New Audio"):
            st.session_state.index_loaded = False
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.markdown("---")
        st.markdown("### Quick Actions")
        
        if st.button("Generate Report"):
            st.session_state.messages.append({
                "role": "user",
                "content": "Generate report",
                "avatar": "🧑"
            })
            
            with st.spinner("Generating report..."):
                response = process_query("Generate report", st.session_state.current_index)
            
            message_data = {
                "role": "assistant",
                "content": response["answer"],
                "avatar": "🤖",
                "pdf_path": response.get("pdf_path"),
                "sources": response.get("sources"),
                "show_sources": True,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            st.session_state.messages.append(message_data)
            st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This chatbot uses RAG to answer questions about doctor-patient consultations and can generate PDF reports.")