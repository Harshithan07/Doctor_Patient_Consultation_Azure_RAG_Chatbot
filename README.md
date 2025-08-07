## Doctor–Patient Consultation RAG Chat Assistant Pipeline

A simple end-to-end pipeline that turns doctor–patient conversation audio into a retrieval-augmented chat assistant.

### What this does

1. **Transcribe audio**
   Uses Azure Speech (or any speech-to-text) to convert recorded consultations into raw text.
2. **Process transcript**
   Cleans and normalizes the text: removes filler words, timestamps, artifacts.
3. **Chunk and embed**
   Splits the cleaned transcript into manageable chunks, generates vector embeddings, and stores them in a vector database (e.g. Pinecone, FAISS).
4. **Run the chat app**
   A lightweight web server that takes user queries, retrieves relevant context from the vector store, and uses an LLM for a RAG-style response.

---

### Folder structure

```
.
├── .gitignore
├── .env                  # API keys, endpoints, secret configs
├── requirements.txt
├── transcribe.py         # Audio → raw text
├── process_transcript.py # Clean & normalize
├── chunker.py            # Split text into chunks
├── embed_and_store.py    # Embed & push to vector DB
└── chatbot_app.py        # Flask/FastAPI RAG chat server
```

---

### Prerequisites

* Python 3.8+
* Azure account (for Speech and OpenAI) **or** equivalent services
* Vector DB account (Pinecone, FAISS server, etc.)

---

### Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/Harshithan07/Doctor_Patient_Consultation_Azure_RAG_Chatbot.git
   cd Doctor_Patient_Consultation_Azure_RAG_Chatbot
   ```

2. Create & activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows PowerShell
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

### Configuration

Copy `.env.example` to `.env` (or create `.env`) and fill in:

```env
# Azure Speech
AZURE_SPEECH_KEY=your_speech_key
AZURE_SPEECH_REGION=your_speech_region

# Azure OpenAI (or other LLM endpoint)
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment

# Vector DB (Pinecone example)
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_pinecone_env
PINECONE_INDEX_NAME=consultations

# General
CHUNK_SIZE=1000       # characters per chunk
CHUNK_OVERLAP=200     # overlap between chunks
```

---

### Usage

1. **Transcribe an audio file**

   ```bash
   python transcribe.py --input path/to/consultation.wav --output transcript.txt
   ```

2. **Clean up the raw transcript**

   ```bash
   python process_transcript.py --input transcript.txt --output cleaned.txt
   ```

3. **Chunk, embed, and store**

   ```bash
   python chunker.py --input cleaned.txt --output chunks.json
   python embed_and_store.py --input chunks.json
   ```

4. **Start the chat server**

   ```bash
   python chatbot_app.py
   ```

   Then open `http://localhost:8000` (or whatever host/port you’ve set) to ask questions about that consultation.

---

### Tips & Troubleshooting

* If you hit quota errors, double-check your Azure Speech and OpenAI quotas in the portal.
* Make sure your vector DB index exists before running `embed_and_store.py`.
* Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `.env` if you find responses miss context or are too slow.

---

### Contributing

1. Fork the repo
2. Create a feature branch
3. Submit a PR with a brief description of your change

---

### License

MIT License. Feel free to use and modify.
