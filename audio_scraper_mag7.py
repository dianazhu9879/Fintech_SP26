# Audio scraper

import re
import time
import json
import random
import requests
from pathlib import Path
from urllib.parse import urljoin, urldefrag
from datetime import datetime
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    NoSuchWindowException,
    WebDriverException,
    TimeoutException,
)
from webdriver_manager.chrome import ChromeDriverManager

# Config

MAG7 = {
    # ticker: (display name, slug, exchange)
    # Slugs verified from live MarketBeat URLs April 2026.
    # If a slug is wrong the auto-discovery fallback will correct it.
    "AAPL":  ("Apple",     "apple-inc-stock",      "NASDAQ"),
    "MSFT":  ("Microsoft", "microsoft-co-stock",   "NASDAQ"),
    "GOOG":  ("Alphabet",  "alphabet-inc-stock",   "NASDAQ"),
    "AMZN":  ("Amazon",    "amazoncom-inc-stock",  "NASDAQ"),
    "META":  ("Meta",      "facebook-inc-stock",   "NASDAQ"),
    "NVDA":  ("NVIDIA",    "nvidia-co-stock",      "NASDAQ"),
    "TSLA":  ("Tesla",     "tesla-inc-stock",      "NASDAQ"),
}

N_MOST_RECENT = 4

OUTPUT_DIR = Path("./audio_downloads")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADLESS = False  # True to make window invisible

# How long (seconds) to wait for a page to load before giving up
PAGE_LOAD_TIMEOUT = 60

# Browser setup + window guard

def make_driver() -> webdriver.Chrome:
    opts = Options()
    if HEADLESS:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1440,900")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--disable-popup-blocking")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_experimental_option("prefs", {
        "profile.default_content_setting_values.popups": 2
    })
    opts.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=opts,
    )
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )

    # Set a page load timeout so driver.get() never hangs indefinitely.
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)

    return driver


def ensure_main_window(driver: webdriver.Chrome):
    # Close other windows to ensure only main window open
    handles = driver.window_handles
    if len(handles) > 1:
        for handle in handles[1:]:
            try:
                driver.switch_to.window(handle)
                driver.close()
            except Exception:
                pass
        driver.switch_to.window(handles[0])


def safe_get(driver: webdriver.Chrome, url: str, retries: int = 2):
    # Navigate to URL and close extra windows
    for attempt in range(1, retries + 2):
        try:
            driver.get(url)
            break  # success
        except TimeoutException:
            print(f"  Page load timed out (attempt {attempt}), stopping load and continuing...")
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass
            break  # page content is likely still usable despite unfinished load
        except WebDriverException as e:
            if attempt <= retries:
                print(f"  WebDriverException on load (attempt {attempt}): {e}. Retrying...")
                time.sleep(3)
            else:
                raise

    time.sleep(0.5)
    ensure_main_window(driver)


def human_pause(lo: float = 1.5, hi: float = 3.5):
    time.sleep(random.uniform(lo, hi))

# Step 1: Find the N most recent earnings report page URLs

def autodiscover_slug(driver: webdriver.Chrome, ticker: str, exchange: str) -> str | None:
    # Use hardcoded slug, if incorrect scrape for other possible slugs
    index_url = f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/"
    soup = BeautifulSoup(driver.page_source, "html.parser")

    # MarketBeat report URLs look like:
    #   /earnings/reports/<date>-<slug>/
    pattern = re.compile(r"/earnings/reports/\d{4}-\d{1,2}-\d{1,2}-([a-z0-9-]+)/?")

    for a in soup.find_all("a", href=True):
        m = pattern.search(a["href"])
        if m:
            discovered = m.group(1)
            print(f"  Auto-discovered slug for {ticker}: '{discovered}'")
            return discovered

    return None


def get_call_report_urls(
    driver: webdriver.Chrome, ticker: str, slug: str, exchange: str
) -> list[dict]:
    # Return n = 4 most recent earnings per ticker
    index_url = f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/"
    print(f"[{ticker}] Navigating to: {index_url}")

    safe_get(driver, index_url)
    human_pause(2, 3)

    soup = BeautifulSoup(driver.page_source, "html.parser")
    calls = []
    seen_urls = set()

    pattern = re.compile(
        r"/earnings/reports/(\d{4}-\d{1,2}-\d{1,2})-" + re.escape(slug) + r"/?"
    )

    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = pattern.search(href)
        if not m:
            continue

        full_url = urljoin("https://www.marketbeat.com", href)
        clean_url, _ = urldefrag(full_url)

        if clean_url in seen_urls:
            continue
        seen_urls.add(clean_url)

        date_str = m.group(1)
        calls.append({
            "ticker": ticker,
            "date": date_str,
            "label": a.get_text(strip=True) or date_str,
            "url": clean_url,
        })

    # Attempt auto discovery if no results
    if not calls:
        print(f"  Slug '{slug}' matched nothing — attempting auto-discovery...")
        discovered_slug = autodiscover_slug(driver, ticker, exchange)
        if discovered_slug and discovered_slug != slug:
            alt_pattern = re.compile(
                r"/earnings/reports/(\d{4}-\d{1,2}-\d{1,2})-"
                + re.escape(discovered_slug) + r"/?"
            )
            for a in soup.find_all("a", href=True):
                href = a["href"]
                m = alt_pattern.search(href)
                if not m:
                    continue
                full_url = urljoin("https://www.marketbeat.com", href)
                clean_url, _ = urldefrag(full_url)
                if clean_url in seen_urls:
                    continue
                seen_urls.add(clean_url)
                date_str = m.group(1)
                calls.append({
                    "ticker": ticker,
                    "date": date_str,
                    "label": a.get_text(strip=True) or date_str,
                    "url": clean_url,
                })

    def parse_date(c):
        parts = c["date"].split("-")
        try:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return datetime.min

    calls.sort(key=parse_date, reverse=True)
    recent = calls[:N_MOST_RECENT]

    print(f"  Found {len(calls)} unique report pages, using {len(recent)} most recent:")
    for c in recent:
        print(f"    {c['date']} -> {c['url']}")

    return recent


# Step 2: Extract audio URL from report page

PLAY_BUTTON_SELECTORS = [
    (By.ID,          "playBtn"),
    (By.CSS_SELECTOR, "[id*='play']"),
    (By.CSS_SELECTOR, "button[aria-label*='lay']"),   # 'Play' or 'play'
    (By.CSS_SELECTOR, ".play-button"),
    (By.CSS_SELECTOR, ".playButton"),
    (By.CSS_SELECTOR, "[class*='play']"),
    (By.XPATH,        "//*[contains(@class,'play') and (self::button or self::div or self::span)]"),
]


def dismiss_overlays(driver: webdriver.Chrome):
    dismiss_selectors = [
        "button#onetrust-accept-btn-handler",
        "button.cookie-accept",
        "button[aria-label='Close']",
        ".modal-close",
        "#closeBtn",
    ]
    for sel in dismiss_selectors:
        try:
            btn = driver.find_element(By.CSS_SELECTOR, sel)
            btn.click()
            time.sleep(0.5)
        except Exception:
            pass


def click_play(driver: webdriver.Chrome) -> bool:
    # Try every play button lmao
    dismiss_overlays(driver)

    for by, selector in PLAY_BUTTON_SELECTORS:
        try:
            play_btn = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((by, selector))
            )
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", play_btn)
            time.sleep(0.4)
            dismiss_overlays(driver)
            driver.execute_script("arguments[0].click();", play_btn)
            print(f"  Clicked play button via selector: {selector}")
            return True
        except TimeoutException:
            continue
        except Exception as e:
            print(f"  Selector '{selector}' failed: {e}")
            continue

    print("  Play button not found on page (tried all selectors)")
    return False


def scan_logs_for_audio(driver: webdriver.Chrome) -> str | None:
    try:
        logs = driver.get_log("performance")
    except Exception:
        return None

    for log in logs:
        try:
            msg = json.loads(log["message"])
            method = msg.get("message", {}).get("method", "")
            if method in ("Network.requestWillBeSent", "Network.responseReceived"):
                params = msg["message"].get("params", {})
                url = (
                    params.get("request", {}).get("url")
                    or params.get("response", {}).get("url")
                    or ""
                )
                if "files.quartr.com/audio-files" in url:
                    return url
        except Exception:
            continue
    return None


def scan_source_for_audio(page_source: str) -> str | None:
    m = re.search(r'https://files\.quartr\.com/audio-files/[^\s"\'<>]+', page_source)
    return m.group(0) if m else None


def extract_audio_url(driver: webdriver.Chrome, call: dict) -> str | None:
    url = call["url"]
    ticker = call["ticker"]
    date = call["date"]

    print(f"\n  [{ticker} {date}] Loading: {url}")

    try:
        driver.get_log("performance")
    except Exception:
        pass

    safe_get(driver, url)
    human_pause(2, 3)

    clicked = click_play(driver)
    if clicked:
        human_pause(4, 6)

    audio_url = scan_logs_for_audio(driver)
    if audio_url:
        print(f"  Found in network logs: {audio_url[:90]}...")
        return audio_url

    try:
        audio_url = scan_source_for_audio(driver.page_source)
    except WebDriverException:
        audio_url = None

    if audio_url:
        print(f"  Found in page source: {audio_url[:90]}...")
    else:
        print(f"  [{ticker} {date}] No audio URL found")

    return audio_url


# Step 3: Download audio

def get_browser_cookies(driver: webdriver.Chrome) -> dict:
    try:
        return {c["name"]: c["value"] for c in driver.get_cookies()}
    except Exception:
        return {}


def download_audio(call: dict, audio_url: str, cookies: dict) -> Path:
    ticker = call["ticker"]
    date = call["date"].replace("-", "_")
    filename = OUTPUT_DIR / f"{ticker}_{date}_earnings_call.mp3"

    if filename.exists():
        print(f"  Already exists, skipping: {filename.name}")
        return filename

    print(f"  Downloading {filename.name} ...")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        ),
        "Referer": call["url"],
        "Accept": "*/*",
        "Accept-Encoding": "identity;q=1, *;q=0",
        "Range": "bytes=0-",
    }

    with requests.get(audio_url, headers=headers, cookies=cookies, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0

        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 64):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    mb = downloaded // 1024 // 1024
                    mb_total = total // 1024 // 1024
                    print(f"\r  Progress: {pct:.1f}%  ({mb} MB / {mb_total} MB)", end="", flush=True)

    print(f"\n  Saved: {filename}")
    return filename


# Main

def main():
    results = []

    print("Starting browser...")
    driver = make_driver()

    try:
        print("Warming up — visiting MarketBeat homepage...")
        safe_get(driver, "https://www.marketbeat.com/")
        human_pause(2, 3)
        dismiss_overlays(driver)

        for ticker, (name, slug, exchange) in MAG7.items():
            print(f"\n{'='*55}")
            print(f"  {name} ({ticker})")
            print(f"{'='*55}")

            try:
                calls = get_call_report_urls(driver, ticker, slug, exchange)

                if not calls:
                    print(f"  No report pages found even after auto-discovery.")
                    print(f"  Manual check: https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/")
                    continue

                for call in calls:
                    try:
                        audio_url = extract_audio_url(driver, call)
                        entry = {**call, "audio_url": audio_url, "file": None}

                        if audio_url:
                            cookies = get_browser_cookies(driver)
                            path = download_audio(call, audio_url, cookies)
                            entry["file"] = str(path)
                        else:
                            print(f"  Skipping download for {ticker} {call['date']}")

                        results.append(entry)

                    except (NoSuchWindowException, WebDriverException) as e:
                        print(f"  Browser window lost on {ticker} {call['date']}: {e}")
                        print("  Restarting browser and continuing...")
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        driver = make_driver()
                        safe_get(driver, "https://www.marketbeat.com/")
                        human_pause(2, 3)
                        results.append({**call, "audio_url": None, "file": None})

                    human_pause(2, 4)

            except Exception as e:
                print(f"  Error processing {ticker}: {e}")
                import traceback
                traceback.print_exc()

            human_pause(3, 6)

    finally:
        try:
            driver.quit()
        except Exception:
            pass
        print("\nBrowser closed.")

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    found = sum(1 for r in results if r["audio_url"])
    print(f"\nDone! {found}/{len(results)} audio files found.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()