import os
import json
import time
import asyncio
from google import genai
from google.genai import types
from google.genai import errors
from pydub import AudioSegment
import audioop

# Python 3.13 Fix
import sys
if 'audioop' not in sys.modules:
    sys.modules['audioop'] = audioop

# --- RETRY DECORATOR ---
def retry_with_backoff(retries=3, delay=5):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    return await func(*args, **kwargs)
                except errors.ClientError as e:
                    if e.code == 429:
                        wait_time = delay * (2 ** i)
                        print(f"⚠️ Rate Limit Hit (429). Waiting {wait_time}s before retry...")
                        await asyncio.sleep(wait_time)
                    else:
                        raise e
            raise Exception("Max retries exceeded. Please try again later.")
        return wrapper
    return decorator


# 1. Merge Logic
async def merge_audios(file_paths: list, output_path: str):
    if not file_paths:
        raise ValueError("No files provided for merging.")

    abs_output_path = os.path.abspath(output_path)
    print(f"DEBUG: Merging {len(file_paths)} files...")

    combined = AudioSegment.empty()

    for path in file_paths:
        try:
            abs_path = os.path.abspath(path)
            if os.path.getsize(abs_path) == 0:
                continue
            audio = AudioSegment.from_file(abs_path)
            combined += audio
        except Exception:
            continue

    combined.export(abs_output_path, format="mp3", bitrate="192k")

    if os.path.getsize(abs_output_path) < 100:
        raise ValueError("Merge failed: Output file is too small.")

    return abs_output_path


# 2. Transcribe Logic
# Returns a tuple: (transcription_dict, token_usage_dict)
@retry_with_backoff(retries=3, delay=5)
async def transcribe_audio(audio_path: str):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in .env file")

    client = genai.Client(api_key=api_key)
    abs_audio_path = os.path.abspath(audio_path)

    print(f"DEBUG: Uploading file: {abs_audio_path}...")

    audio_file = client.files.upload(
        file=abs_audio_path,
        config={'mime_type': 'audio/mpeg'}
    )

    prompt = """
    Listen to this audio. Generate a highly accurate transcription with timestamps for every single word.

    Return a JSON object with this exact schema:
    {
        "full_transcript": "The complete text of the audio.",
        "words": [
            {"word": "Hello", "start": "00:00.00", "end": "00:00.50"},
            {"word": "world", "start": "00:00.51", "end": "00:01.00"}
        ]
    }
    """

    print("DEBUG: Transcribing with Gemini Flash Latest...")

    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=audio_file.uri,
                        mime_type=audio_file.mime_type
                    ),
                    types.Part.from_text(text=prompt)
                ]
            )
        ],
        config={
            'response_mime_type': 'application/json'
        }
    )

    # ── Extract token usage ──
    token_usage = None
    try:
        um = response.usage_metadata
        print(f"DEBUG transcribe usage_metadata: {um}")
        print(f"DEBUG transcribe usage_metadata dir: {dir(um)}")
        input_tokens = (
            getattr(um, 'prompt_token_count', None) or
            getattr(um, 'input_tokens', None) or
            getattr(um, 'prompt_tokens', None) or 0
        )
        output_tokens = (
            getattr(um, 'candidates_token_count', None) or
            getattr(um, 'output_tokens', None) or
            getattr(um, 'completion_tokens', None) or 0
        )
        token_usage = {
            "input":  input_tokens,
            "output": output_tokens,
            "model":  "flash-latest"
        }
        print(f"DEBUG transcribe token_usage resolved: {token_usage}")
    except Exception as e:
        print(f"DEBUG transcribe token_usage error: {e}")

    return json.loads(response.text), token_usage