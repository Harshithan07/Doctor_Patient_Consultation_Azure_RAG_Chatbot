import os
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

WHISPER_ENDPOINT = os.environ.get("WHISPER_DEPLOYMENT")
AZURE_API_KEY = os.environ.get("AZURE_API_KEY")

def transcribe_audio(file_path: str) -> str:
    """
    Transcribes .mp3 audio using Azure GPT-4o transcribe model.
    Saves full transcript as .txt file.

    Args:
        file_path: path to the .mp3 audio

    Returns:
        Full transcript as string.
    """
    print("🗣️ Transcribing audio with Azure GPT-4o...")
    
    # Check if environment variables are loaded
    if not WHISPER_ENDPOINT or not AZURE_API_KEY:
        print("❌ Error: Missing environment variables. Check WHISPER_ENDPOINT and AZURE_API_KEY in .env file")
        return ""
    
    # Convert to Path object and resolve absolute path
    file_path = Path(file_path).resolve()
    
    # Check if file exists
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        return ""
    
    print(f"📁 Processing file: {file_path}")
    
    # Headers
    headers = {
        "Authorization": f"Bearer {AZURE_API_KEY}"
    }
    
    try:
        # Prepare file for upload
        with open(file_path, 'rb') as audio_file:
            files = {
                'file': audio_file
            }
            data = {
                'model': 'gpt-4o-transcribe'
            }
            
            # Make API request
            response = requests.post(WHISPER_ENDPOINT, headers=headers, files=files, data=data)
        
        if response.status_code == 200:
            transcript = response.json().get('text', '')
            
            # Save transcript to .txt file
            output_path = Path(file_path).with_suffix(".txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(transcript)
            
            print(f"✅ Full transcript saved to {output_path}")
            return transcript.strip()
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return ""
            
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return ""
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return ""