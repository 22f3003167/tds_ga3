"""Invoice field-extraction API.

POST /extract  {"invoice_text": "..."}  ->
{"invoice_no": "...", "date": "YYYY-MM-DD", "vendor": "...",
 "amount": 123.45, "tax": 12.34, "currency": "INR"}
"""
import json
import os
import re
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-2.5-flash,gemini-flash-latest,gemini-2.0-flash,gemini-2.0-flash-lite",
)
GEMINI_MODELS = [m.strip() for m in GEMINI_MODEL.split(",") if m.strip()]

FIELDS = ["invoice_no", "date", "vendor", "amount", "tax", "currency"]

SYSTEM_PROMPT = """You are a precise invoice-data extraction engine. You are given the raw \
plain text of an invoice (formats vary widely: business invoices, tax invoices, service \
invoices, receipts). Extract exactly these 6 fields and return ONLY a single JSON object \
with exactly these keys, nothing else — no markdown, no explanation:

{
  "invoice_no": string or null,   // the invoice/reference/bill number, e.g. "INV-2026-0041" or "NS/2026/778"
  "date": string or null,         // the invoice/issue date, converted to ISO format YYYY-MM-DD
  "vendor": string or null,       // the seller/vendor/company name issuing the invoice (not the buyer/client/bill-to party)
  "amount": number or null,       // the SUBTOTAL — the amount BEFORE tax is added. Do not include tax. Do not use the grand total.
  "tax": number or null,          // the tax amount only (GST, IGST, VAT, sales tax, etc.) as a plain number
  "currency": string or null      // ISO-style currency code, e.g. "INR", "USD". Infer from symbols (Rs./₹ -> INR, $ -> USD, £ -> GBP, € -> EUR) if not stated explicitly.
}

Rules:
- If a field cannot be determined from the text, use null for that field (never guess or invent a value).
- Numbers must be plain JSON numbers: no currency symbols, no commas, no units (e.g. 2199.00 not "Rs. 2,199.00").
- "amount" is the pre-tax subtotal, NOT the grand total. If there is only a single total with no separate subtotal/tax breakdown, and no tax is mentioned at all, put that total in "amount" and set "tax" to null.
- "date" must always be YYYY-MM-DD when a date can be determined, regardless of the input format (e.g. "15 March 2026" -> "2026-03-15").
- Output must be valid JSON and contain exactly the 6 keys above — no extra keys, no trailing commentary.
"""

app = FastAPI(title="Invoice Extraction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    invoice_text: str


def _model_url(model: str) -> str:
    return f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _coerce_number(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        cleaned = re.sub(r"[^\d.\-]", "", v)
        if not cleaned or cleaned in ("-", "."):
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _coerce_date(v):
    if not v or not isinstance(v, str):
        return None
    v = v.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        return v
    fmts = ["%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%d/%m/%Y",
            "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d"]
    for fmt in fmts:
        try:
            return datetime.strptime(v, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _coerce_str(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def normalize(raw: dict) -> dict:
    raw = raw or {}
    return {
        "invoice_no": _coerce_str(raw.get("invoice_no")),
        "date": _coerce_date(raw.get("date")),
        "vendor": _coerce_str(raw.get("vendor")),
        "amount": _coerce_number(raw.get("amount")),
        "tax": _coerce_number(raw.get("tax")),
        "currency": _coerce_str(raw.get("currency")),
    }


EMPTY_RESULT = {k: None for k in FIELDS}


def ask_gemini(invoice_text: str) -> dict:
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [{"text": f"Invoice text:\n\n{invoice_text}"}]}
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
    }

    last_err = None
    with httpx.Client(timeout=60) as client:
        for model in GEMINI_MODELS:
            try:
                resp = client.post(
                    _model_url(model), params={"key": GEMINI_API_KEY}, json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPStatusError as e:
                last_err = e
                body = e.response.text[:200] if e.response is not None else ""
                print(f"[{model}] {e.response.status_code}: {body}")
                continue
            try:
                parts = data["candidates"][0]["content"]["parts"]
                text = "".join(p.get("text", "") for p in parts)
            except (KeyError, IndexError):
                text = ""
            parsed = _extract_json_object(text)
            if parsed is not None:
                return normalize(parsed)
    if last_err is not None:
        raise last_err
    return dict(EMPTY_RESULT)


@app.get("/")
def root():
    return {"status": "ok", "endpoint": "POST /extract"}


@app.get("/health")
def health():
    return {"status": "healthy", "gemini_key_set": bool(GEMINI_API_KEY)}


@app.post("/extract")
def extract(req: ExtractRequest):
    try:
        result = ask_gemini(req.invoice_text)
    except Exception as e:  # noqa: BLE001
        print(f"Error: {e}")
        result = dict(EMPTY_RESULT)
    # Guarantee exactly the 6 keys are present even on partial failure.
    return {k: result.get(k) for k in FIELDS}
