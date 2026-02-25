"""
generate_timeline.py

Usage:
    python generate_timeline.py --json transcription.json --doc slides.pdf
    python generate_timeline.py --json transcription.json --doc slides.docx

Output: master_plan.json in the current directory.
"""

import argparse
import json
import os
import sys
import mimetypes

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

PROMPT = """
# DaVinci Resolve Timeline Planner

## ROLE
You are a deterministic DaVinci Resolve timeline planner. You convert educational presentation content into precisely-timed visual clips based on audio narration and slide content.

---

## INPUTS
You will receive exactly 2 files:
- Slide Script (.docx/.pdf) — source of all text content and structure
- Audio Transcription (.json) — source of all timing and emphasis cues

These are your ONLY sources. Do not invent, assume, or generate anything beyond what is explicitly present in these files.

---

## CORE PRINCIPLES

1. Deterministic — Same inputs always produce identical output. Every decision is traceable to the input files.
2. Timing from audio — All start_sec/end_sec values come from audio timestamps. Content appears when first mentioned in narration.
3. Text from document — All texts values are exact quotes from the document. Preserve line breaks (\\n), punctuation, and structure. Never paraphrase.
4. Effects from emphasis — Effects are triggered by vocal cues (stress, repetition, definitions, key terms). Timing aligns with the exact moment of emphasis.

---

## PROCESSING WORKFLOW

### STEP 1: Parse Inputs

Audio JSON — extract:
- Full transcript
- Segments: start, end, text
- Words: word, start, end

Identify in audio:
- Section transitions: "Let's start with...", "Now let's tackle...", "Moving on to..."
- Question intros: "Can you tell me...", "Why is..."
- Emphasis cues: "important", "key point", "crucially", repetition, spelling out
- List enumerations: "First...", "Second...", "Third..."

Document — extract with structure preserved:
- Main titles, section headings, subsection headings
- Paragraph text (preserve line breaks)
- Numbered/bulleted lists (preserve numbering and formatting)
- Tables and structured data

---

### STEP 2: Map Sections to Audio

For each major section:
- Match audio transition phrases to document section headings
- Section start = first mention of topic in audio
- Section end = start of next section or natural conclusion
- Map each piece of content to the timestamp when it is spoken

---

### STEP 3: Assign Templates

Use this decision tree for every piece of content:

| Check                                          | Effect          |
|------------------------------------------------|-----------------|
| Main course/unit title (first slide)?          | Title           |
| Major section heading?                         | SlideTitle      |
| Interrogative question?                        | Question        |
| Numbered item with heading + explanation?      | Explanation Box |
| Multiple choice option (A/B/C/D)?              | Option (4 clips)|
| Brief label, name, or short item (<15 words)?  | Textbox         |
| Paragraph, definition, or substantive content? | Paragraph       |

---

### STEP 4: Calculate Timing

Section titles (base clips):
  start_sec = when topic first mentioned in audio
  end_sec   = when section ends or next section starts

Content clips (layered over base):
  start_sec = when specific content is mentioned
  end_sec   = base clip's end_sec (synchronized)

List items (progressive reveal — all share same end_sec):
  item_1: start = when first item mentioned,  end = section_end
  item_2: start = when second item mentioned, end = section_end
  item_3: start = when third item mentioned,  end = section_end

MCQ options (progressive reveal — all share same end_sec):
  option_1: start = when A mentioned, end = section_end
  option_2: start = when B mentioned, end = section_end, over = option_1
  option_3: start = when C mentioned, end = section_end, over = option_2
  option_4: start = when D mentioned, end = section_end, over = option_3

Effects:
  start_sec = exact timestamp when emphasis begins
  end_sec   = section end, or when speaker moves to next topic

---

### STEP 5: Apply Layering

Use "over" to build visual hierarchy. No two overlapping clips may omit "over".

Pattern A — Title with content:
  { "id": "clip_1", "effect_name": "SlideTitle", "texts": ["Introduction"], "start_sec": 60.0, "end_sec": 100.0 }
  { "id": "clip_2", "effect_name": "Paragraph",  "texts": ["Definition..."], "start_sec": 65.0, "end_sec": 100.0, "over": "clip_1" }

Pattern B — Progressive list:
  { "id": "heading", "effect_name": "Paragraph",      "texts": ["Objectives of HRA"],  "start_sec": 100.0, "end_sec": 150.0 }
  { "id": "item_1",  "effect_name": "Explanation Box", "texts": ["1", "First..."],       "start_sec": 105.0, "end_sec": 150.0, "over": "heading" }
  { "id": "item_2",  "effect_name": "Explanation Box", "texts": ["2", "Second..."],      "start_sec": 120.0, "end_sec": 150.0, "over": "item_1" }
  { "id": "item_3",  "effect_name": "Explanation Box", "texts": ["3", "Third..."],       "start_sec": 135.0, "end_sec": 150.0, "over": "item_2" }

Pattern C — MCQ:
  { "id": "question", "effect_name": "Question", "texts": ["Question", "Which policy is regarded as the Economic Constitution?"], "start_sec": 6.94,  "end_sec": 27.24 }
  { "id": "opt_1",    "effect_name": "Option",   "texts": ["A", "Industrial Policy, 1948"], "start_sec": 10.5, "end_sec": 27.24, "over": "question" }
  { "id": "opt_2",    "effect_name": "Option",   "texts": ["B", "Industrial Policy, 1956"], "start_sec": 11.0, "end_sec": 27.24, "over": "opt_1" }
  { "id": "opt_3",    "effect_name": "Option",   "texts": ["C", "Industrial Policy, 1991"], "start_sec": 11.5, "end_sec": 27.24, "over": "opt_2" }
  { "id": "opt_4",    "effect_name": "Option",   "texts": ["D", "NITI Aayog Formation"],    "start_sec": 12.0, "end_sec": 27.24, "over": "opt_3" }

---

### STEP 6: Detect Effects

Highlight — trigger when speaker says: "important", "key", "crucial", "critical", "note that"; repeats a term; or enumerates key items.

Underline — trigger on definitions ("is the process of...", "defined as..."), first mention of critical terminology, or "let me emphasize."

Text+ (Handwriting) — trigger on "remember", "don't forget", "crucial point," or key answers being called out. Always use font: "MV Boli".

Effect timing:
  start_sec = exact timestamp when emphasis word/phrase begins
  end_sec   = section end, or when speaker moves on

---

## RULES

### Critical (never violate)

- V1 is reserved — never place clips on V1 (background only)
- No invented content — every text must trace to document or audio
- No overlaps without "over" — clips sharing timeline must use "over"
- Exact text counts per effect:

| Effect          | texts count | Notes                                          |
|-----------------|-------------|------------------------------------------------|
| Title           | 1           |                                                |
| SlideTitle      | 1           |                                                |
| Question        | 2           | ["Question", "actual question text"]           |
| Paragraph       | 1           |                                                |
| Textbox         | 1           |                                                |
| Explanation Box | 2           | ["heading", "body"]                            |
| Option          | 2           | ["letter", "option text"] — 4 clips for MCQ   |
| Highlight       | 0           | kind: "effect"                                 |
| Underline       | 0           | kind: "effect"                                 |
| Text+           | 1           | font: "MV Boli"                                |

### Operational

- Clip IDs: sequential strings — clip_1, clip_2, etc.
- FPS: always 30.0
- Version: always 1
- Timing precision: up to 6 decimal places (e.g., 2.966667)
- Related clips in the same section share end_sec

---

## OUTPUT FORMAT

Output only valid JSON. No explanations, preambles, or markdown.

{
  "version": 1,
  "fps": 30.0,
  "clips": [
    {
      "id": "clip_1",
      "kind": "title",
      "effect_name": "SlideTitle",
      "start_sec": 60.666667,
      "end_sec": 98.166667,
      "texts": ["Introduction to Emerging Accounting Practices"]
    },
    {
      "id": "clip_2",
      "kind": "title",
      "effect_name": "Paragraph",
      "start_sec": 68.066667,
      "end_sec": 98.166667,
      "texts": ["With the evolving business environment..."],
      "over": "clip_1"
    },
    {
      "id": "clip_3",
      "kind": "effect",
      "effect_name": "Highlight",
      "start_sec": 88.966667,
      "end_sec": 98.166667,
      "over": "clip_2"
    },
    {
      "id": "clip_4",
      "kind": "title",
      "effect_name": "Text+",
      "start_sec": 90.0,
      "end_sec": 98.166667,
      "texts": ["Key Point"],
      "font": "MV Boli",
      "over": "clip_3"
    }
  ]
}

Required fields: id, kind, effect_name, start_sec, end_sec
texts: required if kind is "title", forbidden if kind is "effect"
font: optional, only for Text+ — use "MV Boli"
over: optional — references another clip's id for layering

---

## PRE-OUTPUT VALIDATION

Before outputting, verify every clip:

[ ] All texts content exists verbatim in document or audio
[ ] Line breaks (\\n) preserved from document
[ ] Text count matches effect_name requirement
[ ] start_sec matches audio timestamp for when content is mentioned
[ ] end_sec > start_sec (no negative durations)
[ ] Related clips share end_sec
[ ] effect_name is a valid value from the template table
[ ] Effects (Highlight, Underline) have no texts
[ ] Text+ includes font: "MV Boli"
[ ] "over" used whenever clips overlap
[ ] Referenced "over" IDs exist and have no circular references
[ ] Clip IDs are sequential with no gaps
[ ] JSON is valid and complete

File name: master_plan.json
"""


def get_mime_type(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    mime_map = {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc": "application/msword",
    }
    return mime_map.get(ext, mimetypes.guess_type(filepath)[0] or "application/octet-stream")


def main():
    parser = argparse.ArgumentParser(description="Generate DaVinci Resolve master_plan.json from transcription + slides")
    parser.add_argument("--json", required=True, help="Path to transcription JSON file (output from oliveboard-transcriber)")
    parser.add_argument("--doc", required=True, help="Path to slide script PDF or DOCX file")
    parser.add_argument("--output", default="master_plan.json", help="Output file path (default: master_plan.json)")
    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.json):
        print(f"ERROR: JSON file not found: {args.json}")
        sys.exit(1)
    if not os.path.exists(args.doc):
        print(f"ERROR: Document file not found: {args.doc}")
        sys.exit(1)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in environment or .env file")
        sys.exit(1)

    # Load transcription JSON
    print(f"Loading transcription: {args.json}")
    with open(args.json, "r", encoding="utf-8") as f:
        transcription_data = json.load(f)
    transcription_text = json.dumps(transcription_data, indent=2)

    client = genai.Client(api_key=api_key)

    # Upload document file
    doc_mime = get_mime_type(args.doc)
    print(f"Uploading document ({doc_mime}): {args.doc}")
    doc_file = client.files.upload(
        file=args.doc,
        config={"mime_type": doc_mime}
    )
    print(f"Document uploaded: {doc_file.uri}")

    print("Sending to Gemini for timeline generation...")

    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-06-05",
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=PROMPT),
                    types.Part.from_text(text=f"\n\n## AUDIO TRANSCRIPTION JSON\n\n```json\n{transcription_text}\n```"),
                    types.Part.from_uri(
                        file_uri=doc_file.uri,
                        mime_type=doc_mime
                    ),
                ]
            )
        ],
        config={
            "response_mime_type": "application/json",
        }
    )

    # Parse and save output
    print("Parsing response...")
    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        # Try to extract JSON from the response text
        text = response.text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
        result = json.loads(text)

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    clip_count = len(result.get("clips", []))
    print(f"\n✅ Done! master_plan.json saved to: {output_path}")
    print(f"   Total clips generated: {clip_count}")
    print(f"   Version: {result.get('version')}, FPS: {result.get('fps')}")


if __name__ == "__main__":
    main()