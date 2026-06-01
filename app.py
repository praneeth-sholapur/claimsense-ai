import logging
import os
import io
import json
import shutil
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import google.genai as genai
from google.cloud import vision
from google.oauth2 import service_account
from pinecone import Pinecone
from pdf2image import convert_from_bytes
from docx import Document
import PyPDF2
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_VISION_KEY = os.getenv("GOOGLE_VISION_KEY")
# On Linux, find poppler from PATH. On Windows, fall back to the POPPLER_PATH env var.
_pdftoppm = shutil.which("pdftoppm")
POPPLER_PATH = os.path.dirname(_pdftoppm) if _pdftoppm else (os.getenv("POPPLER_PATH") or None)

HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# External client initialisation — log clearly so Render logs show any failure
# ---------------------------------------------------------------------------

def _init_vision():
    if not GOOGLE_VISION_KEY:
        log.error("GOOGLE_VISION_KEY env var is not set")
        return None
    try:
        info = json.loads(GOOGLE_VISION_KEY)
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-vision"]
        )
        client = vision.ImageAnnotatorClient(credentials=creds)
        log.info("Vision client initialised OK")
        return client
    except Exception:
        log.error("Failed to initialise Vision client:\n%s", traceback.format_exc())
        return None


try:
    claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    log.info("Anthropic client initialised OK")
except Exception:
    log.error("Failed to initialise Anthropic client:\n%s", traceback.format_exc())
    claude = None

try:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    log.info("Gemini client initialised OK")
except Exception:
    log.error("Failed to initialise Gemini client:\n%s", traceback.format_exc())
    gemini_client = None

try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("knowledge-assistant")
    log.info("Pinecone client initialised OK")
except Exception:
    log.error("Failed to initialise Pinecone client:\n%s", traceback.format_exc())
    pc = None
    index = None

vision_client = _init_vision()

# ---------------------------------------------------------------------------
# Conversation state
# ---------------------------------------------------------------------------

conversation_history = []
document_text = ""
document_loaded = False

SYSTEM_PROMPT = """You are ClaimSense, an AI insurance claims assistant built to analyze insurance documents.

How to communicate:
- Answer directly. No preamble, no filler phrases, no "Great question" or "Certainly" or "Absolutely".
- No emojis. No exclamation marks.
- Short by default. Go longer only when the question genuinely needs it.
- Warm but not performative. Like a colleague who knows insurance well and gets to the point.
- If asked how you are, respond the way a person would — brief and genuine. One line, then move on.
- If you do not know something or it is not in the document, say so plainly. Do not guess.
- Do not repeat what the user just said back to them.
- Do not start every sentence with I.
- Do not over-explain. If something can be said in one sentence, say it in one sentence.

CONVERSATION MODE — when no document is loaded:
- Have a normal conversation. Answer questions about what you do, how you work, or anything the user asks.
- Steer naturally toward uploading a document when it makes sense. Do not force it.
- If someone says hi or hello, respond briefly and ask how you can help. Do not give a long introduction unless asked.

ANALYSIS MODE — when a document has been uploaded:
- You have the full document text in the conversation history.
- Analyze it immediately and thoroughly.
- Extract all key data fields, classify the claim type, identify anomalies, and recommend a next action.
- Answer any follow-up questions accurately using only what is in the document.
- If something is not in the document, say so directly. Never make up information.
- If numbers do not add up or information is missing, flag it clearly.

When a document is uploaded, always structure your analysis response exactly like this:

CLAIM TYPE: [type of claim]
RISK LEVEL: [Low / Medium / High]

EXTRACTED DATA:
[list the key fields found in the document — claim number, parties, amounts, dates, vehicle or property details, deadlines, etc.]

FLAGS:
[list any anomalies, missing information, inconsistencies, or time-sensitive items — or write None found if everything looks clean]

RECOMMENDATION:
[one to three sentences of plain English next steps the person should take]

After the analysis, ask if they have any questions about the document."""

# ---------------------------------------------------------------------------
# OCR / extraction helpers
# ---------------------------------------------------------------------------

def ocr_image_bytes(image_bytes):
    if not vision_client:
        raise RuntimeError("Vision client is not available — check GOOGLE_VISION_KEY")
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""


def extract_from_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    for i, page in enumerate(reader.pages):
        t = page.extract_text()
        if t:
            text += f"\n[Page {i+1}]\n{t}"
    return text, len(reader.pages)


def extract_from_pdf_ocr(file_bytes):
    images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
    text = ""
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        text += f"\n[Page {i+1}]\n{ocr_image_bytes(buf.getvalue())}"
    return text, len(images)


def extract_from_image(file_bytes):
    return ocr_image_bytes(file_bytes), 1


def extract_from_docx(file_bytes):
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    return text, 1


def extract_from_txt(file_bytes):
    return file_bytes.decode("utf-8", errors="ignore"), 1


def extract_text(filename, file_bytes):
    name = filename.lower()
    log.info("Extracting text from: %s", name)
    if name.endswith(".txt"):
        return extract_from_txt(file_bytes)
    elif name.endswith(".docx"):
        return extract_from_docx(file_bytes)
    elif name.endswith((".jpg", ".jpeg", ".png")):
        return extract_from_image(file_bytes)
    elif name.endswith(".pdf"):
        text, pages = extract_from_pdf_text(file_bytes)
        if len(text.strip()) < 100:
            log.info("PDF text too short (%d chars), falling back to OCR", len(text.strip()))
            return extract_from_pdf_ocr(file_bytes)
        return text, pages
    return "", 0

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "claude": claude is not None,
        "vision": vision_client is not None,
        "pinecone": index is not None,
        "gemini": gemini_client is not None,
        "poppler_path": POPPLER_PATH,
        "poppler_available": POPPLER_PATH is not None,
    })


@app.route("/upload", methods=["POST"])
def upload():
    global document_text, document_loaded, conversation_history
    try:
        file = request.files.get("file")
        if not file:
            return jsonify({"error": "No file provided"}), 400

        log.info("Upload received: %s", file.filename)
        file_bytes = file.read()

        try:
            text, pages = extract_text(file.filename, file_bytes)
        except Exception:
            msg = traceback.format_exc()
            log.error("Text extraction failed:\n%s", msg)
            return jsonify({"error": f"Text extraction failed: {msg}"}), 500

        if not text.strip():
            return jsonify({"error": "Could not extract text from this file"}), 400

        document_text = text
        document_loaded = True
        conversation_history = []

        analysis_prompt = (
            f"A new insurance document has been uploaded: {file.filename}\n\n"
            f"Here is the full document content:\n\n{document_text}\n\n"
            "Please analyze this document now. Provide the structured ClaimSense report "
            "with claim type, risk level, extracted data, flags, and recommendation. "
            "Then ask if they have any questions."
        )
        conversation_history.append({"role": "user", "content": analysis_prompt})

        try:
            response = claude.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=conversation_history,
            )
        except Exception:
            msg = traceback.format_exc()
            log.error("Claude API call failed:\n%s", msg)
            return jsonify({"error": f"Claude API error: {msg}"}), 500

        assistant_reply = response.content[0].text
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        log.info("Upload complete, returning analysis")
        return jsonify({"reply": assistant_reply, "filename": file.filename, "pages": pages})

    except Exception:
        msg = traceback.format_exc()
        log.error("Unhandled error in /upload:\n%s", msg)
        return jsonify({"error": f"Unexpected server error: {msg}"}), 500


@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history
    try:
        data = request.json
        message = data.get("message", "").strip()
        if not message:
            return jsonify({"error": "No message provided"}), 400

        conversation_history.append({"role": "user", "content": message})

        response = claude.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=conversation_history,
        )
        assistant_reply = response.content[0].text
        conversation_history.append({"role": "assistant", "content": assistant_reply})
        return jsonify({"reply": assistant_reply})

    except Exception:
        msg = traceback.format_exc()
        log.error("Unhandled error in /chat:\n%s", msg)
        return jsonify({"error": f"Unexpected server error: {msg}"}), 500


@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history, document_text, document_loaded
    conversation_history = []
    document_text = ""
    document_loaded = False
    return jsonify({"status": "reset"})


@app.route("/")
def serve():
    return send_from_directory(HERE, "claimsense.html")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    app.run(port=port, debug=False)
