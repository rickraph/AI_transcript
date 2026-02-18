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

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("processed", filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg", filename="merged_transcript.mp3")
    return JSONResponse(content={"error": "File not found"}, status_code=404)