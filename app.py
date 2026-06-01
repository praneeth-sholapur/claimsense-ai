from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import anthropic
import google.genai as genai
from pinecone import Pinecone
from google.cloud import vision
from pdf2image import convert_from_bytes
from docx import Document
import PyPDF2
import os
import io
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\shola\vision-key.json"
POPPLER_PATH = r"C:\Users\shola\poppler\poppler-26.02.0\Library\bin"

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

claude = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("knowledge-assistant")
vision_client = vision.ImageAnnotatorClient()

app = Flask(__name__)
CORS(app)

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

conversation_history = []
document_text = ""
document_loaded = False

def ocr_image_bytes(image_bytes):
    image = vision.Image(content=image_bytes)
    response = vision_client.document_text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ""

def extract_from_pdf_text(file_bytes):
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""
    pages = len(reader.pages)
    for i, page in enumerate(reader.pages):
        t = page.extract_text()
        if t:
            text += f"\n[Page {i+1}]\n{t}"
    return text, pages

def extract_from_pdf_ocr(file_bytes):
    images = convert_from_bytes(file_bytes, poppler_path=POPPLER_PATH)
    text = ""
    for i, img in enumerate(images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        ocr_text = ocr_image_bytes(buf.getvalue())
        text += f"\n[Page {i+1}]\n{ocr_text}"
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
    if name.endswith(".txt"):
        return extract_from_txt(file_bytes)
    elif name.endswith(".docx"):
        return extract_from_docx(file_bytes)
    elif name.endswith((".jpg", ".jpeg", ".png")):
        return extract_from_image(file_bytes)
    elif name.endswith(".pdf"):
        text, pages = extract_from_pdf_text(file_bytes)
        if len(text.strip()) < 100:
            return extract_from_pdf_ocr(file_bytes)
        return text, pages
    return "", 0

@app.route("/upload", methods=["POST"])
def upload():
    global document_text, document_loaded, conversation_history

    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file provided"}), 400

    file_bytes = file.read()
    text, pages = extract_text(file.filename, file_bytes)

    if not text.strip():
        return jsonify({"error": "Could not extract text from this file"}), 400

    document_text = text
    document_loaded = True
    conversation_history = []

    analysis_prompt = f"""A new insurance document has been uploaded: {file.filename}

Here is the full document content:

{document_text}

Please analyze this document now. Provide the structured ClaimSense report with claim type, risk level, extracted data, flags, and recommendation. Then ask if they have any questions."""

    conversation_history.append({
        "role": "user",
        "content": analysis_prompt
    })

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=conversation_history
    )

    assistant_reply = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    return jsonify({
        "reply": assistant_reply,
        "filename": file.filename,
        "pages": pages
    })

@app.route("/chat", methods=["POST"])
def chat():
    global conversation_history, document_text, document_loaded

    data = request.json
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    if document_loaded and document_text:
        conversation_history.append({
            "role": "user",
            "content": message
        })
    else:
        conversation_history.append({
            "role": "user",
            "content": message
        })

    response = claude.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=conversation_history
    )

    assistant_reply = response.content[0].text

    conversation_history.append({
        "role": "assistant",
        "content": assistant_reply
    })

    return jsonify({"reply": assistant_reply})

@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history, document_text, document_loaded
    conversation_history = []
    document_text = ""
    document_loaded = False
    return jsonify({"status": "reset"})

@app.route("/")
def serve():
    return send_from_directory(r"C:\Users\shola", "claimsense.html")

if __name__ == "__main__":
    app.run(port=5001, debug=False)
