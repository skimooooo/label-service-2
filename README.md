# TrackFlow HEYBIKE Label Image Generator

Generates a realistic HEYBIKE package photo with a dynamic `SHIP TO` address and dynamic TrackFlow tracking number composited onto a fixed warehouse base image.

Cost: $0 per image. No AI image generation is used at runtime.

## Files

- `base.png` — the HEYBIKE warehouse base photo.
- `label_gen.py` — core deterministic compositing logic.
- `main.py` — FastAPI wrapper exposing `/generate-label`.
- `requirements.txt` — Python dependencies.

## What changes per image

Only these fields are patched:

- `SHIP TO` block
- tracking number text under the barcode

These stay fixed from the base photo:

- HEYBIKE package and warehouse scene
- TrackFlow header
- `SHIP FROM` block
- barcode pixels from `base.png`
- table, box, tape, shadows, and background

The barcode is intentionally copied back from `base.png` as the final image operation so it stays visually static across generations.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

POST `/generate-label`

```json
{
  "recipient": {
    "name": "Helen Deli",
    "line1": "Rosenweg 7",
    "line2": "Bad Mergentheim",
    "line3": "97980",
    "line4": "DE"
  },
  "tracking_number": "TF-Q8ZH7PDVRH",
  "response_format": "url"
}
```

`response_format`:

- `"url"` or `"image"` → returns raw PNG bytes (`image/png`)
- `"base64"` → returns `{ "image_base64": "...", "mime_type": "image/png" }`

## Tracking format

You can send either:

- `TF-Q8ZH7PDVRH`
- `Q8ZH7PDVRH`

The script normalizes the second format to `TF-Q8ZH7PDVRH`.

## Important customization notes

If you replace `base.png` with a different photo, you must re-measure these constants in `label_gen.py`:

- `LABEL_CORNERS`
- `SHIP_TO_RECT`
- `TRACKING_TEXT_RECT`
- `BARCODE_LOCK_RECT`

The current values are calibrated for the included HEYBIKE base photo.
