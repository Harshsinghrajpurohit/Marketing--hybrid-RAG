# prepare_martech.py — strip each SEC 10-K bundle to clean narrative text
# (Item 1 "Business" + Item 7 "MD&A"). Stdlib only. v2 (fixed markers).

import html
import os
import re

VENDORS = {
    "salesforce": "data/salesforce_10k.txt",
    "hubspot":    "data/hubspot_10k.txt",
    "adobe":      "data/adobe_10k.txt",
    "twilio":     "data/twilio_10k.txt",
}
MAX_PER_VENDOR = 25000  # char budget per vendor (tunable later)


def extract_narrative(path):
    raw = open(path, encoding="utf-8", errors="replace").read()

    # 1) The bundled filing is split by <SEC-DOCUMENT>; the 10-K block has <TYPE>10-K.
    blocks = re.split(r"<SEC-DOCUMENT>", raw)
    main = next((b for b in blocks if "<TYPE>10-K" in b), raw)

    # 2) Unwrap HTML, decode entities, normalize whitespace (incl. \xa0 nbsp)
    text = html.unescape(re.sub(r"<[^>]+>", " ", main))
    text = re.sub(r"[\s\xa0]+", " ", text)

    # 3) Slice sections, taking the LAST marker (skips TOC / cross-references).
    def slice_last(pattern, end_pattern):
        matches = [m for m in re.finditer(pattern, text, re.I)]
        if not matches:
            return ""
        m = matches[-1]
        start = m.end()
        e = re.search(end_pattern, text[start:], re.I)
        return text[start: start + e.start()] if e else text[start: start + 80000]

    business = slice_last(r"ITEM\s*1\.\s*BUSINESS", r"ITEM\s*(2|1A)\.")
    mda = slice_last(r"ITEM\s*7\.", r"ITEM\s*8\.")
    return (business + "\n" + mda)[:MAX_PER_VENDOR]


os.makedirs("data/martech", exist_ok=True)
for name, path in VENDORS.items():
    narrative = extract_narrative(path)
    out = f"data/martech/{name}.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(narrative)
    print(f"{name}: {len(narrative):,} chars -> {out}")
