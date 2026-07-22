# Invoice Extraction API

`POST /extract` → accepts raw invoice text, returns 6 structured fields as JSON.
Uses Google Gemini to parse varied plain-text invoice formats. CORS is open (`*`)
so the grader's Cloudflare Worker can call it.

## Endpoints
- `POST /extract` — body `{"invoice_text": "..."}` → always returns:
  ```json
  {"invoice_no": "...", "date": "YYYY-MM-DD", "vendor": "...", "amount": 0.0, "tax": 0.0, "currency": "..."}
  ```
  Missing fields are `null`.
- `GET /` and `GET /health` — status checks

## Rules implemented
- All 6 keys always present (`null` when not found).
- `date` normalized to ISO `YYYY-MM-DD` regardless of input format.
- `amount` = pre-tax subtotal (not the grand total); `tax` = tax amount only.
- Numbers stripped of currency symbols/commas and returned as plain JSON numbers.
- Currency inferred from symbols (Rs./₹→INR, $→USD, £→GBP, €→EUR) when not stated.

## Local run
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GEMINI_API_KEY="your_key"
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"invoice_text": "Invoice No: INV-1\nDate: 15 March 2026\nVendor: Acme\nSubtotal: Rs. 100\nGST: Rs. 18\nTotal: Rs. 118"}'
```

## Deploy to Render
1. Push this folder to a GitHub repo.
2. Render → **New → Web Service** → connect the repo (auto-detects `render.yaml`).
3. Manual settings if needed:
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. **Environment** tab → add `GEMINI_API_KEY` = your key.
5. Deploy → base URL is `https://<name>.onrender.com`.
6. Submit that base URL — the grader calls `POST <url>/extract`.

## Notes
- `GEMINI_MODEL` env var accepts a comma-separated fallback list (default already set);
  the first model that returns valid JSON wins. This works around intermittent
  404/429 responses from specific Gemini model aliases.
- Free Render instances cold-start (~30–50s) after idle.
