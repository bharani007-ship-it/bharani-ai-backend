import os
import json
import urllib.request
import urllib.parse
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pymupdf

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for Android WebView & web clients

# ==========================================================
# CONFIGURATION & GEMINI API CALLER
# ==========================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

def call_gemini_api(contents, system_instruction=None, response_schema=None) -> str:
    """
    Calls the Gemini REST API directly using standard urllib (0 dependencies, 100% resilient).
    Works on all cloud platforms without SDK version conflicts.
    """
    key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    if not key:
        raise ValueError("GEMINI_API_KEY is not configured in the server environment variables. Please get a free key from https://aistudio.google.com and add it to your hosting settings.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={key}"

    payload = {
        "contents": contents
    }

    if system_instruction:
        payload["systemInstruction"] = {
            "parts": [{"text": system_instruction}]
        }

    generation_config = {
        "temperature": 0.4,
        "maxOutputTokens": 2048
    }

    if response_schema:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema

    payload["generationConfig"] = generation_config

    data_bytes = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=40) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            candidates = res_json.get("candidates", [])
            if not candidates:
                raise ValueError("Gemini returned empty response.")
            parts = candidates[0].get("content", {}).get("parts", [])
            if not parts:
                raise ValueError("No text part in Gemini response.")
            return parts[0].get("text", "").strip()
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        if e.code == 429:
            raise ValueError("Gemini free-tier rate limit reached. Please wait 30 seconds and try again.")
        elif e.code == 400:
            raise ValueError(f"Invalid request to Gemini: {err_msg}")
        elif e.code == 403:
            raise ValueError("Invalid GEMINI_API_KEY or API access denied.")
        else:
            raise ValueError(f"Gemini API HTTP {e.code}: {err_msg}")
    except Exception as e:
        raise ValueError(f"Connection error to Gemini: {str(e)}")

# ==========================================================
# ROUTES
# ==========================================================

@app.route("/")
def home():
    """Serves the mobile-optimized PWA / Android dashboard."""
    return render_template("index.html")

@app.route("/api/health")
def health():
    key_configured = bool(os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY).strip())
    return jsonify({
        "status": "online",
        "service": "Bharani AI Cloud Backend",
        "api_key_configured": key_configured,
        "model": MODEL_NAME,
        "cost": "₹0 Free Tier"
    })

@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    history = data.get("history", []) # Client passes recent conversation
    memories = data.get("memories", []) # Client passes relevant memories
    doc_text = str(data.get("doc_text", "")).strip()

    if not message:
        return jsonify({"answer": "Please type a message."})

    # Prepare system instruction
    memory_section = ""
    if memories:
        memory_section = "\nSAVED USER MEMORIES:\n" + "\n".join([f"- {m.get('key')}: {m.get('value')}" for m in memories]) + "\n"

    doc_section = ""
    if doc_text:
        doc_section = f"\nDOCUMENT CONTENT FOR CONTEXT:\n{doc_text[:8000]}\n"
        system_instruction = (
            "You are Bharani AI, a personal AI assistant.\n"
            "The user uploaded a document. RULES:\n"
            "1. Answer questions based on the provided document content.\n"
            "2. If information is not in the document, say: 'I couldn't find that information in the uploaded document.'\n"
            "3. Support English, Tamil (தமிழ்), and Tanglish smoothly based on the user's language.\n"
            f"{memory_section}\n{doc_section}"
        )
    else:
        system_instruction = (
            "You are Bharani AI, an intelligent, helpful personal AI assistant. "
            "You support English, தமிழ் (Tamil), and Tanglish. "
            "Provide helpful, friendly, and concise responses.\n"
            f"{memory_section}"
        )

    # Format contents for Gemini
    contents = []
    # Include up to last 6 messages from client-provided history
    for msg in history[-6:]:
        role = "user" if msg.get("role") == "user" else "model"
        contents.append({
            "role": role,
            "parts": [{"text": str(msg.get("content", ""))}]
        })

    # Current user message
    contents.append({
        "role": "user",
        "parts": [{"text": message}]
    })

    try:
        answer = call_gemini_api(contents, system_instruction=system_instruction)
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "answer": f"⚠️ {str(e)}"})

@app.route("/api/quiz", methods=["POST"])
def quiz_endpoint():
    data = request.get_json(silent=True) or {}
    doc_text = str(data.get("doc_text", "")).strip()
    num_questions = int(data.get("num_questions", 10))

    if not doc_text:
        return jsonify({"success": False, "message": "No document text provided."})

    system_instruction = (
        f"You are creating a study quiz from an uploaded document. "
        f"Generate exactly {num_questions} multiple-choice questions based ONLY on the document text. "
        "Each question must have 4 options (A, B, C, D), a single correct answer letter, and a concise 1-sentence explanation."
    )

    schema = {
        "type": "OBJECT",
        "properties": {
            "questions": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "question": {"type": "STRING"},
                        "A": {"type": "STRING"},
                        "B": {"type": "STRING"},
                        "C": {"type": "STRING"},
                        "D": {"type": "STRING"},
                        "answer": {"type": "STRING", "enum": ["A", "B", "C", "D"]},
                        "explanation": {"type": "STRING"}
                    },
                    "required": ["question", "A", "B", "C", "D", "answer", "explanation"]
                }
            }
        },
        "required": ["questions"]
    }

    contents = [{
        "role": "user",
        "parts": [{"text": f"Generate {num_questions} multiple-choice questions strictly from this text:\n\n{doc_text[:12000]}"}]
    }]

    try:
        raw_json = call_gemini_api(contents, system_instruction=system_instruction, response_schema=schema)
        parsed = json.loads(raw_json)
        return jsonify({"success": True, "questions": parsed.get("questions", [])})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/api/study_notes", methods=["POST"])
def study_notes_endpoint():
    data = request.get_json(silent=True) or {}
    doc_text = str(data.get("doc_text", "")).strip()

    if not doc_text:
        return jsonify({"success": False, "message": "No document text provided."})

    system_instruction = "You are Bharani AI Study Tutor. Create concise revision notes and flashcards (Question & Answer pairs) from the document."
    contents = [{
        "role": "user",
        "parts": [{"text": f"Create study summary and flashcards from this text:\n\n{doc_text[:12000]}"}]
    }]

    try:
        notes = call_gemini_api(contents, system_instruction=system_instruction)
        return jsonify({"success": True, "notes": notes})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@app.route("/api/upload_doc", methods=["POST"])
def upload_doc():
    if "file" not in request.files:
        return jsonify({"success": False, "message": "No file uploaded."})
    file = request.files["file"]
    if not file or not file.filename:
        return jsonify({"success": False, "message": "Empty filename."})

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    try:
        file_bytes = file.read()
        extracted_text = ""
        if ext == ".pdf":
            doc = pymupdf.open(stream=file_bytes, filetype="pdf")
            for page in doc:
                extracted_text += page.get_text() + "\n"
            doc.close()
        elif ext in [".txt", ".md", ".json", ".csv"]:
            extracted_text = file_bytes.decode("utf-8", errors="ignore")
        else:
            return jsonify({"success": False, "message": "Supported formats: PDF, TXT, MD, CSV, JSON."})

        extracted_text = extracted_text.strip()
        if not extracted_text:
            return jsonify({"success": False, "message": "Could not extract text from document."})

        return jsonify({
            "success": True,
            "filename": filename,
            "text": extracted_text[:35000]
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
