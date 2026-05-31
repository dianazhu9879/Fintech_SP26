# Audio scraper v3
# Changes from v2:
#   - Narrowed TICKERS to 40 targets (large-cap tech, small/mid-cap tech, other sectors)
#   - Replaced N_MOST_RECENT with DATE_RANGE filter (2025-01-01 → 2025-12-31)
#     covering: 2024 Q4 reports (Jan-Mar 2025), 2025 Q1 (Apr-Jun), Q2 (Jul-Sep), Q3 (Oct-Dec)
#   - Duplicate-skip logic unchanged (checks for existing file before downloading)

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

# ── Config ────────────────────────────────────────────────────────────────────

TICKERS = {
    # ticker: (display name, slug, exchange)
    # Slugs follow MarketBeat's earnings-report URL pattern:
    #   marketbeat.com/earnings/reports/<date>-<slug>/
    # Auto-discovery fallback will self-correct any wrong slugs at runtime.

    # Large-cap tech / AI / software / semis / cybersecurity
    "MSFT":  ("Microsoft",          "microsoft-co-stock",           "NASDAQ"),
    "AAPL":  ("Apple",              "apple-inc-stock",              "NASDAQ"),
    "NVDA":  ("NVIDIA",             "nvidia-co-stock",              "NASDAQ"),
    "GOOG": ("Alphabet",           "alphabet-inc-stock",           "NASDAQ"),
    "AMZN":  ("Amazon",             "amazoncom-inc-stock",          "NASDAQ"),
    "META":  ("Meta",               "facebook-inc-stock",           "NASDAQ"),  # MB uses FB slug
    "AVGO":  ("Broadcom",           "broadcom-inc-stock",           "NASDAQ"),
    "ORCL":  ("Oracle",             "oracle-co-stock",              "NYSE"),
    "CRM":   ("Salesforce",         "salesforce-inc-stock",         "NYSE"),
    "NOW":   ("ServiceNow",         "servicenow-inc-stock",         "NYSE"),
    "SNOW":  ("Snowflake",          "snowflake-inc-stock",          "NYSE"),
    "DDOG":  ("Datadog",            "datadog-inc-stock",            "NASDAQ"),
    "PLTR":  ("Palantir",           "palantir-technologies-stock",  "NASDAQ"),
    "NET":   ("Cloudflare",         "cloudflare-inc-stock",         "NYSE"),
    "AMD":   ("AMD",                "advanced-micro-devices-stock", "NASDAQ"),
    "QCOM":  ("Qualcomm",           "qualcomm-inc-stock",           "NASDAQ"),
    "MU":    ("Micron",             "micron-technology-stock",      "NASDAQ"),
    "AMAT":  ("Applied Materials",  "applied-materials-stock",      "NASDAQ"),
    "PANW":  ("Palo Alto Networks", "palo-alto-networks-stock",     "NASDAQ"),
    "CRWD":  ("CrowdStrike",        "crowdstrike-holdings-stock",   "NASDAQ"),
    "ZS":    ("Zscaler",            "zscaler-inc-stock",            "NASDAQ"),

    # Small / mid-cap tech / higher-growth tech
    "APP":   ("AppLovin",       "applovin-co-stock",           "NASDAQ"),
    "GTLB":  ("GitLab",         "gitlab-inc-stock",            "NASDAQ"),
    "DOCN":  ("DigitalOcean",   "digitalocean-holdings-stock", "NYSE"),
    "AI":    ("C3.ai",          "c3ai-inc-stock",              "NYSE"),
    "SMCI":  ("Super Micro",    "super-micro-computer-stock",  "NASDAQ"),

    # Everything else
    "PG":    ("Procter & Gamble", "procter-gamble-co-stock",       "NYSE"),
    "KO":    ("Coca-Cola",        "coca-cola-co-stock",            "NYSE"),
    "COST":  ("Costco",           "costco-wholesale-stock",        "NASDAQ"),
    "WMT":   ("Walmart",          "walmart-inc-stock",             "NYSE"),
    "MCD":   ("McDonald's",       "mcdonalds-co-stock",            "NYSE"),
    "CMG":   ("Chipotle",         "chipotle-mexican-grill-stock",  "NYSE"),
    "ABNB":  ("Airbnb",           "airbnb-inc-stock",              "NASDAQ"),
    "LLY":   ("Eli Lilly",        "eli-lilly-and-co-stock",        "NYSE"),
    "MRK":   ("Merck",            "merck-and-co-inc-stock",        "NYSE"),
    "VRTX":  ("Vertex Pharma",    "vertex-pharmaceuticals-stock",  "NASDAQ"),
    "UNH":   ("UnitedHealth",     "unitedhealth-group-stock",      "NYSE"),
    "JPM":   ("JPMorgan Chase",   "jpmorgan-chase-and-co-stock",   "NYSE"),
    "V":     ("Visa",             "visa-inc-stock",                "NYSE"),
    "XYZ":   ("Block",            "square-inc-stock",               "NYSE"),
}

# Filter reports to earnings calls that occurred in this date range.
# Jan–Dec 2025 captures all four requested quarters:
#   2024 Q4 results → reported Jan–Mar 2025
#   2025 Q1 results → reported Apr–Jun 2025
#   2025 Q2 results → reported Jul–Sep 2025
#   2025 Q3 results → reported Oct–Dec 2025
DATE_RANGE_START = datetime(2025, 1, 1)
DATE_RANGE_END   = datetime(2025, 12, 31)

OUTPUT_DIR = Path("./40_downloads")
OUTPUT_DIR.mkdir(exist_ok=True)

HEADLESS = False  # set True to hide browser window

# How long (seconds) to wait for a page to load before giving up
PAGE_LOAD_TIMEOUT = 60

# ── Browser setup + window guard ──────────────────────────────────────────────

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
    driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
    return driver


def ensure_main_window(driver: webdriver.Chrome):
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
    for attempt in range(1, retries + 2):
        try:
            driver.get(url)
            break
        except TimeoutException:
            print(f"  Page load timed out (attempt {attempt}), stopping load and continuing...")
            try:
                driver.execute_script("window.stop();")
            except Exception:
                pass
            break
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

# ── Pre-flight: validate ticker/exchange combos ───────────────────────────────

def validate_tickers(driver: webdriver.Chrome) -> list[str]:
    """
    Visit each ticker's earnings page and flag any that return a 404 or
    'not found' response — almost always caused by a wrong exchange.
    Returns a list of bad ticker strings so main() can decide whether to abort.
    """
    print("\n" + "="*55)
    print("  PRE-FLIGHT: validating all ticker/exchange URLs")
    print("="*55)

    bad_tickers = []

    for ticker, (name, slug, exchange) in TICKERS.items():
        url = f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/"
        try:
            safe_get(driver, url)
            time.sleep(0.8)
            source_lower = driver.page_source.lower()
            title_lower  = driver.title.lower()

            is_bad = (
                "404" in title_lower
                or "page not found" in source_lower
                or "we couldn't find" in source_lower
                or "no results" in source_lower
            )

            status = "FAIL" if is_bad else "OK  "
            print(f"  [{status}] {ticker:5s} ({exchange}) — {name}")

            if is_bad:
                bad_tickers.append(ticker)

        except Exception as e:
            print(f"  [ERR ] {ticker:5s} ({exchange}) — {name}: {e}")
            bad_tickers.append(ticker)

        human_pause(0.5, 1.2)

    print()
    if bad_tickers:
        print(f"  ⚠️  {len(bad_tickers)} ticker(s) need attention: {', '.join(bad_tickers)}")
        print("  Fix the exchange or slug in TICKERS before running the full scrape.")
    else:
        print(f"  ✅  All {len(TICKERS)} tickers validated OK.")

    print("="*55 + "\n")
    return bad_tickers

# ── Step 1: Find N most recent earnings report page URLs ──────────────────────

def autodiscover_slug(driver: webdriver.Chrome, ticker: str, exchange: str) -> str | None:
    soup = BeautifulSoup(driver.page_source, "html.parser")
    pattern = re.compile(r"/earnings/reports/\d{4}-\d{1,2}-\d{1,2}-([a-z0-9-]+)/?")

    for a in soup.find_all("a", href=True):
        m = pattern.search(a["href"])
        if m:
            discovered = m.group(1)
            print(f"  Auto-discovered slug for {ticker}: '{discovered}'")
            return discovered

    return None


def scroll_to_load_all(driver: webdriver.Chrome):
    """Repeatedly scroll to the bottom until no new content appears, then look
    for an explicit 'next page' link and follow it, collecting all page sources."""
    sources = []
    last_height = 0
    for _ in range(8):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1.2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        sources.append(driver.page_source)
        if new_height == last_height:
            break
        last_height = new_height

    # Follow any explicit pagination links (e.g. "older", "next", page numbers)
    visited_pages = {driver.current_url}
    for _ in range(5):
        soup = BeautifulSoup(driver.page_source, "html.parser")
        next_link = None
        for a in soup.find_all("a", href=True):
            text = a.get_text(strip=True).lower()
            href = str(a["href"])
            if any(kw in text for kw in ("older", "next", "previous earnings", "more earnings")):
                full = urljoin("https://www.marketbeat.com", href)
                if full not in visited_pages:
                    next_link = full
                    break
        if not next_link:
            break
        print(f"  Following pagination: {next_link}")
        safe_get(driver, next_link)
        human_pause(1.5, 2.5)
        visited_pages.add(next_link)
        for _ in range(4):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.0)
        sources.append(driver.page_source)

    return sources


def extract_calls_from_sources(sources: list[str], slug: str, ticker: str, seen_urls: set) -> list[dict]:
    pattern = re.compile(
        r"/earnings/reports/(\d{4}-\d{1,2}-\d{1,2})-" + re.escape(slug) + r"/?"
    )
    calls = []
    for source in sources:
        soup = BeautifulSoup(source, "html.parser")
        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            m = pattern.search(href)
            if not m:
                continue
            full_url = urljoin("https://www.marketbeat.com", href)
            clean_url, _ = urldefrag(full_url)
            if clean_url in seen_urls:
                continue
            seen_urls.add(clean_url)
            calls.append({
                "ticker": ticker,
                "date": m.group(1),
                "label": a.get_text(strip=True) or m.group(1),
                "url": clean_url,
            })
    return calls


def get_call_report_urls(
    driver: webdriver.Chrome, ticker: str, slug: str, exchange: str
) -> list[dict]:
    index_url = f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/"
    print(f"[{ticker}] Navigating to: {index_url}")

    safe_get(driver, index_url)
    human_pause(2, 3)

    seen_urls: set = set()
    sources = scroll_to_load_all(driver)
    calls = extract_calls_from_sources(sources, slug, ticker, seen_urls)

    if not calls:
        print(f"  Slug '{slug}' matched nothing — attempting auto-discovery...")
        discovered_slug = autodiscover_slug(driver, ticker, exchange)
        if discovered_slug and discovered_slug != slug:
            calls = extract_calls_from_sources(sources, discovered_slug, ticker, seen_urls)

    def parse_date(c):
        parts = c["date"].split("-")
        try:
            return datetime(int(parts[0]), int(parts[1]), int(parts[2]))
        except Exception:
            return datetime.min

    calls.sort(key=parse_date, reverse=True)
    in_range = [c for c in calls if DATE_RANGE_START <= parse_date(c) <= DATE_RANGE_END]

    print(f"  Found {len(calls)} unique report pages, {len(in_range)} in target date range:")
    for c in in_range:
        print(f"    {c['date']} -> {c['url']}")

    return in_range

# ── Step 2: Extract audio URL from report page ────────────────────────────────

PLAY_BUTTON_SELECTORS = [
    (By.ID,           "playBtn"),
    (By.CSS_SELECTOR, "[id*='play']"),
    (By.CSS_SELECTOR, "button[aria-label*='lay']"),
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

# ── Step 3: Download audio ────────────────────────────────────────────────────

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

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = []

    print("Starting browser...")
    driver = make_driver()

    try:
        print("Warming up — visiting MarketBeat homepage...")
        safe_get(driver, "https://www.marketbeat.com/")
        human_pause(2, 3)
        dismiss_overlays(driver)

        # Pre-flight: catch bad exchange/ticker combos before the main loop
        bad_tickers = validate_tickers(driver)
        if bad_tickers:
            print(
                f"⚠️  Pre-flight found {len(bad_tickers)} problematic ticker(s). "
                "They will be skipped. Fix TICKERS and rerun to include them.\n"
            )
            # Log skipped tickers to manifest immediately so they're not silently lost
            for ticker in bad_tickers:
                name, slug, exchange = TICKERS[ticker]
                results.append({
                    "ticker": ticker,
                    "date": None,
                    "label": None,
                    "url": f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/",
                    "audio_url": None,
                    "file": None,
                    "skip_reason": "pre-flight validation failed (likely wrong exchange or delisted)",
                })

        tickers_to_run = {
            k: v for k, v in TICKERS.items() if k not in bad_tickers
        }

        for ticker, (name, slug, exchange) in tickers_to_run.items():
            print(f"\n{'='*55}")
            print(f"  {name} ({ticker})")
            print(f"{'='*55}")

            try:
                calls = get_call_report_urls(driver, ticker, slug, exchange)

                if not calls:
                    print(f"  No report pages found even after auto-discovery.")
                    print(f"  Manual check: https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/")
                    results.append({
                        "ticker": ticker,
                        "date": None,
                        "label": None,
                        "url": f"https://www.marketbeat.com/stocks/{exchange}/{ticker}/earnings/",
                        "audio_url": None,
                        "file": None,
                        "skip_reason": "no report pages found after auto-discovery",
                    })
                    continue

                for call in calls:
                    try:
                        audio_url = extract_audio_url(driver, call)
                        entry = {**call, "audio_url": audio_url, "file": None, "skip_reason": None}

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
                        results.append({
                            **call,
                            "audio_url": None,
                            "file": None,
                            "skip_reason": f"browser crash: {e}",
                        })

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

    found    = sum(1 for r in results if r.get("audio_url"))
    skipped  = sum(1 for r in results if r.get("skip_reason"))
    attempted = len(results) - skipped
    print(f"\nDone! {found}/{attempted} audio files found ({skipped} skipped).")
    print(f"Manifest: {manifest_path}")

    write_gaps_report()


def write_gaps_report():
    """
    Scan audio_downloads for each ticker and flag which quarter windows
    have no downloaded file. Writes audio_downloads/gaps.json.

    Quarter windows (by earnings call date):
      2024 Q4 → Jan  1 – Mar 31, 2025
      2025 Q1 → Apr  1 – Jun 30, 2025
      2025 Q2 → Jul  1 – Sep 30, 2025
      2025 Q3 → Oct  1 – Dec 31, 2025
    """
    QUARTERS = [
        ("2024 Q4", datetime(2025,  1,  1), datetime(2025,  3, 31)),
        ("2025 Q1", datetime(2025,  4,  1), datetime(2025,  6, 30)),
        ("2025 Q2", datetime(2025,  7,  1), datetime(2025,  9, 30)),
        ("2025 Q3", datetime(2025, 10,  1), datetime(2025, 12, 31)),
    ]

    # Parse existing filenames → {ticker: [datetime, ...]}
    file_pattern = re.compile(r"^([A-Z]+)_(\d{4})_(\d{1,2})_(\d{1,2})_earnings_call\.mp3$")
    ticker_dates: dict[str, list[datetime]] = {t: [] for t in TICKERS}

    for f in OUTPUT_DIR.iterdir():
        m = file_pattern.match(f.name)
        if not m:
            continue
        tkr, yr, mo, dy = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if tkr in ticker_dates:
            try:
                ticker_dates[tkr].append(datetime(yr, mo, dy))
            except ValueError:
                pass

    gaps = {}
    for ticker, dates in ticker_dates.items():
        missing = []
        covered = []
        for label, start, end in QUARTERS:
            if any(start <= d <= end for d in dates):
                covered.append(label)
            else:
                missing.append(label)
        if missing:
            gaps[ticker] = {
                "missing_quarters": missing,
                "covered_quarters": covered,
                "files_found": len(dates),
            }

    gaps_path = OUTPUT_DIR / "gaps.json"
    with open(gaps_path, "w") as f:
        json.dump(gaps, f, indent=2, sort_keys=True)

    if gaps:
        print(f"\n⚠️  {len(gaps)} ticker(s) have missing quarters → {gaps_path}")
        for ticker, info in sorted(gaps.items()):
            print(f"  {ticker:5s}  missing: {', '.join(info['missing_quarters'])}")
    else:
        print(f"\n✅  All {len(TICKERS)} tickers have complete coverage → {gaps_path}")


if __name__ == "__main__":
    main()