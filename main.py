import os
import shutil
import uuid
from typing import List
import aiofiles
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from services import merge_audios, transcribe_audio

# Load .env
import os
import shutil
import uuid
from typing import List
import aiofiles
from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from services import merge_audios, transcribe_audio

# Load .env
load_dotenv()

app = FastAPI(title="Oliveboard Audio Transcriber")

# --- FIX START: Create ALL required directories ---
os.makedirs("uploads", exist_ok=True)
os.makedirs("processed", exist_ok=True)
os.makedirs("static", exist_ok=True)  # <--- THIS LINE IS CRITICAL
# --- FIX END ---

# Mounts (Now this won't crash because we just created the folder above)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/process/")
async def process_audio(
    files: List[UploadFile] = File(...),
    file_order: str = Form(...) 
):
    try:
        session_id = str(uuid.uuid4())
        session_dir = os.path.join("uploads", session_id)
        os.makedirs(session_dir, exist_ok=True)

        # 1. Save uploaded files
        saved_paths = {}
        for file in files:
            file_path = os.path.join(session_dir, file.filename)
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            saved_paths[file.filename] = file_path

        # 2. Reorder files
        ordered_filenames = [f.strip() for f in file_order.split(',') if f.strip()]
        ordered_paths = [saved_paths[fname] for fname in ordered_filenames if fname in saved_paths]

        # 3. Merge
        merged_file_path = os.path.join("processed", f"{session_id}_merged.mp3")
        await merge_audios(ordered_paths, merged_file_path)

        # 4. Transcribe
        transcription_json = await transcribe_audio(merged_file_path)

        # Cleanup
        shutil.rmtree(session_dir)

        return JSONResponse(content={
            "status": "success",
            "merged_audio_url": f"/download/{session_id}_merged.mp3",
            "transcription": transcription_json
        })

    except Exception as e:
        print(f"SERVER ERROR: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)

@app.post("/generate-timeline/")
async def generate_timeline(
    transcription_json: UploadFile = File(...),
    slide_doc: UploadFile = File(...)
):
    """
    Accepts:
      - transcription_json: the JSON output from /process/
      - slide_doc: a PDF or DOCX of the slide script
    Returns: master_plan.json as a downloadable file.
    """
    import json as _json
    from google import genai
    from google.genai import types

    try:
        session_id = str(uuid.uuid4())
        session_dir = os.path.join("uploads", session_id)
        os.makedirs(session_dir, exist_ok=True)

        # Save uploaded files
        json_path = os.path.join(session_dir, "transcription.json")
        async with aiofiles.open(json_path, 'wb') as f:
            await f.write(await transcription_json.read())

        doc_ext = os.path.splitext(slide_doc.filename)[1].lower()
        doc_path = os.path.join(session_dir, f"slides{doc_ext}")
        async with aiofiles.open(doc_path, 'wb') as f:
            await f.write(await slide_doc.read())

        # Load transcription
        with open(json_path, "r", encoding="utf-8") as f:
            transcription_data = _json.load(f)
        transcription_text = _json.dumps(transcription_data, indent=2)

        # Determine MIME type
        mime_map = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
        }
        doc_mime = mime_map.get(doc_ext, "application/pdf")

        api_key = os.getenv("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)

        # Upload document
        doc_file = client.files.upload(file=doc_path, config={"mime_type": doc_mime})

        TIMELINE_PROMPT = open(os.path.join(os.path.dirname(__file__), "timeline_prompt.txt")).read()

        response = client.models.generate_content(
            model="gemini-2.5-pro-preview-06-05",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(text=TIMELINE_PROMPT),
                        types.Part.from_text(text=f"\n\n## AUDIO TRANSCRIPTION JSON\n\n```json\n{transcription_text}\n```"),
                        types.Part.from_uri(file_uri=doc_file.uri, mime_type=doc_mime),
                    ]
                )
            ],
            config={"response_mime_type": "application/json"}
        )

        # Parse result
        try:
            result = _json.loads(response.text)
        except _json.JSONDecodeError:
            text = response.text.strip()
            if text.startswith("```"):
                text = "\n".join(text.split("\n")[1:-1])
            result = _json.loads(text)

        # Save output
        output_path = os.path.join("processed", f"{session_id}_master_plan.json")
        with open(output_path, "w", encoding="utf-8") as f:
            _json.dump(result, f, indent=2, ensure_ascii=False)

        shutil.rmtree(session_dir)

        return JSONResponse(content={
            "status": "success",
            "download_url": f"/download-json/{session_id}_master_plan.json",
            "clip_count": len(result.get("clips", [])),
            "plan": result
        })

    except Exception as e:
        print(f"TIMELINE ERROR: {e}")
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)


@app.get("/download-json/{filename}")
async def download_json(filename: str):
    file_path = os.path.join("processed", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="application/json", filename="master_plan.json")
    return JSONResponse(content={"error": "File not found"}, status_code=404)


@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("processed", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg", filename="merged_transcript.mp3")
    return JSONResponse(content={"error": "File not found"}, status_code=404)