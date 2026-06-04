#!/usr/bin/env python3
"""
Earnings Call Transcript Scraper
---------------------------------
Downloads earnings call transcripts for the top 7 companies from The Motley Fool.

Companies: Apple (AAPL), Microsoft (MSFT), Nvidia (NVDA), Amazon (AMZN),
           Alphabet (GOOGL), Meta (META), Tesla (TSLA)

Output: data/text_alpha/earnings_transcripts/<TICKER>/<TICKER>_<Quarter>_<Year>.txt

How to run:
  pip install requests beautifulsoup4
  python3 src/text_alpha/scraper.py

Then drag the 'data/text_alpha/earnings_transcripts' folder into Google Drive.
"""

import os
import re
import time
import textwrap
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "data/text_alpha/earnings_transcripts"
DELAY = 2   # seconds between requests – be polite to the server

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Verified transcript URLs (most recent first) ─────────────────────────────────
TRANSCRIPTS = {
    "AAPL": {
        "name": "Apple",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2026/01/29/apple-aapl-q1-2026-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/10/31/apple-q4-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/08/01/apple-aapl-q3-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/01/30/apple-aapl-q1-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2024/10/31/apple-aapl-q4-2024-earnings-call-transcript/",
        ],
    },
    "MSFT": {
        "name": "Microsoft",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2026/01/28/microsoft-msft-q2-2026-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/10/29/microsoft-msft-q1-2026-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/08/05/microsoft-msft-q4-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/01/29/microsoft-msft-q2-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2024/10/30/microsoft-msft-q1-2025-earnings-call-transcript/",
        ],
    },
    "NVDA": {
        "name": "Nvidia",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2026/02/25/nvidia-nvda-q4-2026-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/11/19/nvidia-nvda-q3-2026-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/02/26/nvidia-nvda-q4-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2024/11/20/nvidia-nvda-q3-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2024/08/28/nvidia-nvda-q2-2025-earnings-call-transcript/",
        ],
    },
    "AMZN": {
        "name": "Amazon",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2026/02/05/amazon-amzn-q4-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/10/31/amazon-amzn-q3-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/02/06/amazoncom-amzn-q4-2024-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2024/08/01/amazoncom-amzn-q2-2024-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2024/04/30/amazoncom-amzn-q1-2024-earnings-call-transcript/",
        ],
    },
    "GOOGL": {
        "name": "Alphabet",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2026/02/04/alphabet-googl-q4-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/11/27/alphabet-googl-q3-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/10/30/alphabet-goog-q3-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/07/23/alphabet-googl-q2-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/02/05/alphabet-goog-q4-2024-earnings-call-transcript/",
        ],
    },
    "META": {
        "name": "Meta",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2026/01/28/meta-meta-q4-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/10/29/meta-platforms-meta-q3-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/01/29/meta-platforms-meta-q4-2024-earnings-call-transcri/",
            "https://www.fool.com/earnings/call-transcripts/2024/10/30/meta-platforms-meta-q3-2024-earnings-call-transcri/",
        ],
    },
    "TSLA": {
        "name": "Tesla",
        "urls": [
            "https://www.fool.com/earnings/call-transcripts/2026/01/28/tesla-tsla-q4-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/10/22/tesla-tsla-q3-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/07/23/tesla-tsla-q2-2025-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2025/01/29/tesla-tsla-q4-2024-earnings-call-transcript/",
            "https://www.fool.com/earnings/call-transcripts/2024/04/23/tesla-tsla-q1-2024-earnings-call-transcript/",
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_soup(url: str, session: requests.Session) -> BeautifulSoup | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except requests.RequestException as e:
        print(f"  [ERROR] {url}: {e}")
        return None


def slug_to_quarter(url: str, title: str) -> str:
    text = url + " " + title
    m = re.search(r"(q[1-4])[\s\-_]*(20\d{2})", text, re.IGNORECASE)
    if m:
        return f"{m.group(1).upper()}_{m.group(2)}"
    m = re.search(r"(20\d{2})[\s\-_]*(q[1-4])", text, re.IGNORECASE)
    if m:
        return f"{m.group(2).upper()}_{m.group(1)}"
    return re.sub(r"[^\w\s-]", "", text)[:60].strip().replace(" ", "_")


def sanitise_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def extract_text(soup: BeautifulSoup, url: str) -> str:
    body = soup.find("div", class_=re.compile(r"article-body"))
    if not body:
        body = soup.find("article") or soup.find("main")
    if not body:
        return f"[Could not locate article body]\n{url}"

    for tag in body(["script", "style", "aside", "figure", "nav", "footer", "button", "form"]):
        tag.decompose()

    lines = [line.strip() for line in body.get_text(separator="\n").splitlines()]
    return "\n".join(line for line in lines if line)


def scrape_company(ticker: str, company: dict, session: requests.Session):
    name = company["name"]
    urls = company["urls"]

    print(f"\n{'='*60}")
    print(f"  {name} ({ticker})  –  {len(urls)} transcripts")
    print(f"{'='*60}")

    company_dir = os.path.join(OUTPUT_DIR, ticker)
    os.makedirs(company_dir, exist_ok=True)

    for i, url in enumerate(urls, start=1):
        print(f"\n  [{i}/{len(urls)}] {url.split('/call-transcripts/')[-1].rstrip('/')}")
        time.sleep(DELAY)

        soup = get_soup(url, session)
        if not soup:
            continue

        # Try to get the page title for a readable filename
        title_tag = soup.find("h1")
        title = title_tag.get_text(strip=True) if title_tag else url

        text = extract_text(soup, url)
        quarter = slug_to_quarter(url, title)
        filename = sanitise_filename(f"{ticker}_{quarter}.txt")
        filepath = os.path.join(company_dir, filename)

        header = (
            f"COMPANY  : {name} ({ticker})\n"
            f"TITLE    : {title}\n"
            f"URL      : {url}\n"
            f"SCRAPED  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{'─'*70}\n\n"
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header + text)

        kb = os.path.getsize(filepath) / 1024
        print(f"  ✓ Saved → {filepath}  ({kb:.1f} KB)")


def write_readme():
    readme = textwrap.dedent(f"""
    # Earnings Call Transcripts
    Generated: {datetime.now().strftime('%Y-%m-%d')}

    Companies: Apple · Microsoft · Nvidia · Amazon · Alphabet · Meta · Tesla

    Folder structure:
        data/text_alpha/earnings_transcripts/
          AAPL/  AAPL_Q1_2026.txt  …
          MSFT/  MSFT_Q2_2026.txt  …
          …

    Source: The Motley Fool – https://www.fool.com/earnings-call-transcripts/

    Google Drive upload:
    1. Select the entire 'data/text_alpha/earnings_transcripts' folder.
    2. Drag it onto https://drive.google.com
    """).strip()

    path = os.path.join(OUTPUT_DIR, "README.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(readme)
    print(f"\n  README written → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    total = sum(len(c["urls"]) for c in TRANSCRIPTS.values())
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Earnings Call Transcript Scraper – Top 7 Companies    ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"Output directory : ./{OUTPUT_DIR}/")
    print(f"Total transcripts: {total}")
    print(f"Source           : The Motley Fool\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with requests.Session() as session:
        for ticker, company in TRANSCRIPTS.items():
            scrape_company(ticker, company, session)

    write_readme()

    print("\n" + "="*60)
    print("  Done! Drag the 'data/text_alpha/earnings_transcripts' folder into Google Drive.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
