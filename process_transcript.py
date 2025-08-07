import sys
from pathlib import Path
from dotenv import load_dotenv
from chunker import split_text_into_chunks
from embed_and_store import store_in_faiss
from transcribe import transcribe_audio

load_dotenv()

def process_audio_file(audio_path: str, index_name: str = "RES0215"):
    """
    Process an audio file: transcribe, chunk, and store in FAISS.
    
    Args:
        audio_path: Path to audio file (.mp3)
        index_name: Name for the FAISS index
    """
    audio_path = Path(audio_path)
    
    # Step 1: Transcribe audio to text
    if audio_path.suffix == '.mp3':
        print(f"🎵 Processing audio file: {audio_path}")
        transcript = transcribe_audio(str(audio_path))
        if not transcript:
            print("❌ Transcription failed")
            return
        txt_path = audio_path.with_suffix(".txt")
    else:
        # If already a text file, just read it
        txt_path = audio_path
        if not txt_path.exists():
            print(f"❌ File not found: {txt_path}")
            return
        transcript = txt_path.read_text(encoding="utf-8")
    
    # Step 2: Split into chunks
    print("🧩 Splitting transcript into chunks...")
    chunks = split_text_into_chunks(transcript, max_tokens=300)
    print(f"📊 Created {len(chunks)} chunks")
    
    # Step 3: Store in FAISS
    store_in_faiss(chunks, index_dir=f"faiss_store/{index_name}")
    
    print("✅ Processing complete!")

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        index_name = sys.argv[2] if len(sys.argv) > 2 else "RES0215"
        process_audio_file(file_path, index_name)
    else:
        # Default example
        process_audio_file("RES0215.mp3", "RES0215")