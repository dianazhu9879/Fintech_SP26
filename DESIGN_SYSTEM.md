# AlphaSights — Design System & Codex Guidelines

## Overview

This is the source of truth for building the AlphaSights Earnings Intelligence Dashboard. Follow these guidelines exactly when implementing components, pages, or features in Codex.

---

## App Architecture

```
/
├── index.html          ← Sector/ticker browser (home screen)
├── ticker.html         ← Individual ticker detail page (receives ?ticker=MSFT)
├── css/
│   ├── tokens.css      ← All design tokens (colors, spacing, type)
│   ├── base.css        ← Reset, body, global styles
│   ├── components.css  ← Reusable UI components
│   └── layout.css      ← Page-level layout structures
├── js/
│   ├── data.js         ← All ticker/earnings data
│   ├── home.js         ← Sector browser logic
│   └── ticker.js       ← Ticker detail page logic
└── assets/
    └── fonts/          ← Self-hosted fonts (optional fallback)
```

---

## Design Philosophy

**Aesthetic:** Dark, editorial, data-dense. Inspired by Bloomberg Terminal and expert-network research workflows. Clean grid structure with high information density. Every pixel earns its place.

**Core principle:** Data hierarchy first. The story flows: Bias → Result → Reaction → Language → History. Users should always know WHERE they are and WHAT it means.

**Feel:** Serious financial intelligence tool. Not a consumer app. Precision over decoration.

---

## Typography

### Font Stack
```css
--font-display: 'Syne', sans-serif;          /* Headers, ticker names, labels */
--font-mono: 'IBM Plex Mono', monospace;     /* Numbers, data values, codes */
--font-body: 'Syne', sans-serif;             /* Body copy, descriptions */
```

### Google Fonts Import
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Syne:wght@400;500;600;700&display=swap" rel="stylesheet">
```

### Type Scale
```css
--text-2xs:  9px;    /* Section labels, legend keys — UPPERCASE + letter-spacing */
--text-xs:   10px;   /* Sub-labels, secondary metadata */
--text-sm:   11px;   /* Data row labels, minor values */
--text-base: 13px;   /* Body copy, descriptions */
--text-md:   15px;   /* Panel titles, key labels */
--text-lg:   18px;   /* Stat values, important numbers */
--text-xl:   24px;   /* Large KPI numbers */
--text-2xl:  32px;   /* Ticker symbol display */
--text-3xl:  48px;   /* Hero numbers */
```

### Type Rules
- Section labels: UPPERCASE, `letter-spacing: 0.14em`, `--text-2xs`, `--color-text-muted`
- Numbers/values: always `font-family: var(--font-mono)`
- Ticker symbols: `font-family: var(--font-display)`, `font-weight: 700`
- Never use font-weight above 700
- Body copy line-height: 1.6

---

## Color Tokens

### Base Palette
```css
:root {
  /* Backgrounds */
  --color-bg-base:        #0a0a0b;   /* Page background */
  --color-bg-surface:     #111113;   /* Cards, panels */
  --color-bg-elevated:    #17171a;   /* Hover states, nested panels */
  --color-bg-overlay:     #1e1e22;   /* Dropdowns, tooltips */

  /* Borders */
  --color-border-subtle:  rgba(255,255,255,0.06);   /* Default card borders */
  --color-border-default: rgba(255,255,255,0.10);   /* Hover, active borders */
  --color-border-strong:  rgba(255,255,255,0.18);   /* Focused inputs, emphasis */

  /* Text */
  --color-text-primary:   #e8e6e0;                  /* Main readable text */
  --color-text-secondary: rgba(232,230,224,0.60);   /* Labels, secondary info */
  --color-text-muted:     rgba(232,230,224,0.35);   /* Hints, disabled, placeholders */
  --color-text-disabled:  rgba(232,230,224,0.20);   /* Truly inactive */

  /* Semantic — Bullish */
  --color-bull:           #4ecb71;
  --color-bull-bg:        rgba(78,203,113,0.12);
  --color-bull-border:    rgba(78,203,113,0.25);
  --color-bull-text:      #4ecb71;

  /* Semantic — Bearish */
  --color-bear:           #ff5757;
  --color-bear-bg:        rgba(255,87,87,0.12);
  --color-bear-border:    rgba(255,87,87,0.25);
  --color-bear-text:      #ff5757;

  /* Semantic — Neutral/Warning */
  --color-warn:           #f0a030;
  --color-warn-bg:        rgba(240,160,48,0.12);
  --color-warn-border:    rgba(240,160,48,0.25);
  --color-warn-text:      #f0a030;

  /* Semantic — Info/Blue */
  --color-info:           #5a8aff;
  --color-info-bg:        rgba(90,138,255,0.12);
  --color-info-border:    rgba(90,138,255,0.25);
  --color-info-text:      #5a8aff;

  /* Semantic — Purple (accent) */
  --color-accent:         #a37aff;
  --color-accent-bg:      rgba(163,122,255,0.12);
  --color-accent-border:  rgba(163,122,255,0.25);

  /* Sector Colors */
  --color-sector-tech:    #5a8aff;
  --color-sector-ai:      #a37aff;
  --color-sector-ev:      #4ecb71;
  --color-sector-ecomm:   #f0a030;
}
```

### Color Usage Rules
1. **Backgrounds** always use `--color-bg-*` tokens, never hardcoded hex
2. **Positive/bullish values** → `--color-bull` (returns, beats, positive sentiment)
3. **Negative/bearish values** → `--color-bear` (misses, negative returns, risk flags)
4. **Uncertain/caution** → `--color-warn` (mixed signals, capex concerns, watch items)
5. **Data/informational** → `--color-info` (analyst pressure, Q&A highlights)
6. **Accent/special** → `--color-accent` (defensive language, special metrics)

---

## Spacing Scale

```css
--space-1:   4px;
--space-2:   8px;
--space-3:   12px;
--space-4:   16px;
--space-5:   20px;
--space-6:   24px;
--space-8:   32px;
--space-10:  40px;
--space-12:  48px;
--space-16:  64px;
```

### Spacing Rules
- Panel inner padding: `var(--space-4)` (16px)
- Card gap in grids: `var(--space-3)` (12px)
- Section-to-section: `var(--space-4)` (16px)
- Topbar height: 48px
- Sidebar width: 280px

---

## Border Radius

```css
--radius-sm:   4px;    /* Chips, badges, small pills */
--radius-md:   6px;    /* Buttons, inputs, small cards */
--radius-lg:   10px;   /* Panels, main cards */
--radius-xl:   14px;   /* Large modals, feature cards */
--radius-full: 9999px; /* Fully rounded pills */
```

---

## Component Specifications

### 1. Topbar
```
Height: 48px
Background: var(--color-bg-base)
Border-bottom: 1px solid var(--color-border-subtle)
Position: sticky, top: 0, z-index: 200
Padding: 0 24px
Layout: flex, align-items: center, justify-content: space-between

Left: Logo + Sector Filter Tabs
Right: Date label + Live badge
```

### 2. Sector Filter Tab
```
Padding: 6px 14px
Border-radius: var(--radius-md)
Font: var(--font-display), 11px, weight 600, letter-spacing 0.08em, UPPERCASE
Default state: color var(--color-text-muted), background transparent
Hover state: color var(--color-text-primary), background var(--color-bg-elevated)
Active state: color = sector color, background = sector color with 0.12 opacity
Transition: all 150ms ease
```

### 3. Ticker Card (Home Screen)
```
Layout: vertical card, ~200px wide, ~240px tall
Background: var(--color-bg-surface)
Border: 1px solid var(--color-border-subtle)
Border-radius: var(--radius-lg)
Padding: 16px
Cursor: pointer

Hover state:
  - Border: 1px solid var(--color-border-default)
  - Background: var(--color-bg-elevated)
  - Transform: translateY(-2px)
  - Transition: all 200ms ease

Click: navigate to ticker.html?ticker=MSFT

Contents (top to bottom):
  1. Header row: ticker symbol (left) + bias pill (right)
  2. Company name (muted, 11px)
  3. Divider line
  4. EPS surprise row
  5. 1W return (large mono number)
  6. Mini sparkline (svg, 4 datapoints)
  7. Bottom row: sector tag + confidence bar
```

### 4. Bias Pill
```
Padding: 3px 9px
Border-radius: var(--radius-full)
Font: 10px, weight 700, letter-spacing 0.08em, UPPERCASE, var(--font-mono)

Variants:
  .pill-bull   → bg: var(--color-bull-bg),  border: var(--color-bull-border),  color: var(--color-bull)
  .pill-bear   → bg: var(--color-bear-bg),  border: var(--color-bear-border),  color: var(--color-bear)
  .pill-neutral → bg: rgba(255,255,255,0.06), border: var(--color-border-default), color: var(--color-text-secondary)
```

### 5. Panel (detail page)
```
Background: var(--color-bg-surface)
Border: 1px solid var(--color-border-subtle)
Border-radius: var(--radius-lg)
Padding: 16px

Panel header:
  - Title: 11px, weight 600, letter-spacing 0.04em, UPPERCASE, var(--color-text-primary)
  - Subtitle: 10px, var(--color-text-muted)
  - Badge (right): 9px, var(--font-mono), padding 2px 7px, radius var(--radius-sm), bg rgba(255,255,255,0.06)
```

### 6. Confidence Bar
```
Track: height 4px, background rgba(255,255,255,0.08), border-radius 2px
Fill: height 4px, border-radius 2px
  Bull variant: background #4ecb71
  Bear variant: background #ff5757
  Neutral: background rgba(232,230,224,0.3)
Transition: width 400ms ease
```

### 7. Sentiment Meter Row
```
Layout: flex, align-items center, gap 8px
Key label: 10px, var(--color-text-secondary), fixed width 90px
Track: flex 1, height 5px, border-radius 3px, overflow hidden, bg rgba(255,255,255,0.07)
Fill: height 5px, border-radius 3px
Value: 10px, var(--font-mono), var(--color-text-secondary), width 28px, text-align right

Fill colors by metric:
  Positive lang    → var(--color-bull)
  Negative lang    → var(--color-bear)
  Risk language    → var(--color-warn)
  Uncertainty      → var(--color-warn)
  Analyst pressure → var(--color-info)
  Defensive lang   → var(--color-accent)
  Guidance strength→ var(--color-bull)
```

### 8. Topic Chip
```
Padding: 4px 9px
Border-radius: var(--radius-sm)
Font: 10px, weight 500, letter-spacing 0.03em
Border: 1px solid

Variants:
  .topic-pos  → bg var(--color-bull-bg),   border var(--color-bull-border),  color var(--color-bull)
  .topic-neg  → bg var(--color-bear-bg),   border var(--color-bear-border),  color var(--color-bear)
  .topic-warn → bg var(--color-warn-bg),   border var(--color-warn-border),  color var(--color-warn)
  .topic-neut → bg rgba(255,255,255,0.04), border var(--color-border-subtle), color var(--color-text-muted)

Hover: opacity 0.85, cursor pointer
Click: opens topic drill-down or tooltip with detail
```

### 9. Audio Block Visualizer
```
Layout: flex row, gap 2px, flex 1
Each block:
  Width: flex 1
  Height: 16px
  Border-radius: 2px
  Background by segment type:
    Prepared remarks: var(--color-bull) at varying opacity
    Q&A segments: var(--color-info) at varying opacity
    Elevated/stressed: var(--color-warn) at varying opacity
  Opacity: 0.3–1.0 based on signal intensity
```

### 10. History Row
```
Layout: flex, align-items center, gap 8px
Padding: 7px 10px
Border-radius: 6px
Background: rgba(255,255,255,0.025)
Border: 1px solid rgba(255,255,255,0.05)
Cursor: pointer
Hover: background rgba(255,255,255,0.05)

Contents:
  Quarter label: 11px, var(--font-mono), var(--color-text-muted), width 40px
  Tone dots: 7×7px circles, colors per tone signal
  Return value: 11px, var(--font-mono), weight 700, colored bull/bear
  Horizon tag: 9px, var(--color-text-disabled)
```

### 11. Sector Card Group (Home Screen)
```
Section header:
  Sector name: 11px, UPPERCASE, letter-spacing 0.14em, var(--color-text-muted)
  Sector color dot: 8px circle
  Ticker count badge: 9px pill

Ticker cards grid:
  display: grid
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr))
  gap: var(--space-3)
```

### 12. Ticker Detail — Sidebar
```
Width: 280px
Border-right: 1px solid var(--color-border-subtle)
Padding: 20px 16px
Display: flex, flex-direction column, gap 20px
Overflow-y: auto
Position: sticky or fixed based on scroll behavior
```

### 13. Five Questions Component
```
List of 5 Q&A rows
Each row:
  Number: 9px, var(--font-mono), var(--color-text-disabled), width 14px
  Question text: 11px, var(--color-text-secondary)
  Answer: 11px, weight 600, colored by sentiment
    Positive answers → var(--color-bull)
    Negative answers → var(--color-bear)
    Mixed/caution    → var(--color-warn)
    Neutral          → var(--color-text-primary)
Border-bottom: 1px solid rgba(255,255,255,0.04) between rows
```

---

## Page Layouts

### Home Page (index.html)
```
┌─────────────────────────────────────────────────────┐
│ TOPBAR: Logo | Sector Tabs | Date + Live             │
├─────────────────────────────────────────────────────┤
│ HERO STRIP: Market summary bar (optional)            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  SECTOR: AI / Cloud                                 │
│  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐     │
│  │ MSFT │ │ NVDA │ │ GOOGL│ │ META │ │ AMZN │     │
│  └──────┘ └──────┘ └──────┘ └──────┘ └──────┘     │
│                                                     │
│  SECTOR: Consumer Hardware                          │
│  ┌──────┐ ┌──────┐                                  │
│  │ AAPL │ │ TSLA │                                  │
│  └──────┘ └──────┘                                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Ticker Detail Page (ticker.html)
```
┌─────────────────────────────────────────────────────┐
│ TOPBAR: ← Back | Ticker Switcher | Date + Live       │
├──────────────┬──────────────────────────────────────┤
│              │                                      │
│   SIDEBAR    │   CONTENT AREA                       │
│   (280px)    │                                      │
│              │   [Summary Banner]                   │
│  Bias Signal │                                      │
│              │   [Stock Reaction] [Transcript Tone] │
│  Earnings    │                                      │
│  Result      │   [Audio Signals]  [Topic Analysis]  │
│              │                                      │
│  Five        │   [Historical Comparison]            │
│  Questions   │                                      │
│              │                                      │
└──────────────┴──────────────────────────────────────┘
```

---

## Navigation & Routing

### Home → Ticker
```javascript
// On ticker card click
function navigateToTicker(symbol) {
  window.location.href = `ticker.html?ticker=${symbol}`;
}

// On ticker detail page load
const params = new URLSearchParams(window.location.search);
const ticker = params.get('ticker') || 'MSFT';
renderDashboard(ticker);
```

### Ticker Switcher (detail page topbar)
- Shows all 7 Mag7 tickers as small clickable buttons
- Active ticker highlighted with `--color-info` accent
- Clicking updates URL param and re-renders dashboard without full reload:
```javascript
function switchTicker(symbol) {
  const url = new URL(window.location);
  url.searchParams.set('ticker', symbol);
  window.history.pushState({}, '', url);
  renderDashboard(symbol);
}
```

### Back Button
```javascript
// Back to home
document.getElementById('backBtn').onclick = () => {
  window.location.href = 'index.html';
};
```

---

## Sector Groupings

```javascript
const SECTORS = [
  {
    id: 'ai-cloud',
    label: 'AI & Cloud',
    color: '#5a8aff',
    tickers: ['MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN']
  },
  {
    id: 'consumer-hardware',
    label: 'Consumer & Hardware',
    color: '#4ecb71',
    tickers: ['AAPL', 'TSLA']
  }
];
```

*Extend this array as more tickers are added.*

---

## Animations & Transitions

### Page Load Stagger
```css
/* Apply to ticker cards on home */
.ticker-card {
  opacity: 0;
  transform: translateY(8px);
  animation: fadeUp 300ms ease forwards;
}
.ticker-card:nth-child(1) { animation-delay: 0ms; }
.ticker-card:nth-child(2) { animation-delay: 60ms; }
.ticker-card:nth-child(3) { animation-delay: 120ms; }
.ticker-card:nth-child(4) { animation-delay: 180ms; }
.ticker-card:nth-child(5) { animation-delay: 240ms; }

@keyframes fadeUp {
  to { opacity: 1; transform: translateY(0); }
}
```

### Confidence Bar Fill (on load)
```css
.conf-fill {
  width: 0%;
  animation: fillBar 600ms ease 200ms forwards;
}
@keyframes fillBar {
  to { width: var(--target-width); }
}
/* Set --target-width via inline style in JS */
```

### Live Dot Pulse
```css
.live-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--color-bull);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.25; }
}
```

### Panel hover
```css
.panel:hover {
  border-color: var(--color-border-default);
  transition: border-color 150ms ease;
}
```

---

## Data Shape (data.js)

Each ticker object follows this exact schema:

```javascript
const TICKER_DATA = {
  MSFT: {
    // Identity
    symbol:       'MSFT',
    name:         'Microsoft Corp.',
    sector:       'ai-cloud',

    // Bias
    bias:         'BULLISH',      // 'BULLISH' | 'BEARISH' | 'NEUTRAL' | 'STRONG BULL' | 'STRONG BEAR'
    confidence:   74,             // 0–100
    probBull:     68,             // 0–100
    probNeut:     21,
    probBear:     11,
    horizons: [
      { period: 'Open', direction: 'up' },     // 'up' | 'down' | 'flat'
      { period: '1D',   direction: 'up' },
      { period: '1W',   direction: 'up' },
      { period: '1M',   direction: 'flat' },
    ],

    // Earnings
    eps:     { actual: 3.46, estimate: 3.22, surprisePct: 7.5 },
    revenue: { actual: '61.9B', estimate: '60.8B', surprisePct: 1.8 },
    guide:   { actual: '64.0B', consensus: '62.5B', surprisePct: 2.4 },
    netMargin:  35.4,   // percent
    fcf:        '21.1B',
    revGrowthPct: 17.6, // YoY percent (negative = decline)

    // Returns
    returns: {
      open: 4.2,   // percent
      d1:   3.5,
      w1:   2.1,
      m1:   0.4,
      excessVsBenchmark: 3.1,
    },

    // Transcript sentiment (0–100 scores)
    sentiment: {
      prepared: {
        positiveLang:    78,
        negativeLang:    18,
        riskLanguage:    34,
        uncertainty:     28,
        analystPressure: 45,
        defensiveLang:   22,
        guidanceStrength:82,
      },
      qa: {
        positiveLang:    62,
        negativeLang:    38,
        riskLanguage:    58,
        uncertainty:     49,
        analystPressure: 72,
        defensiveLang:   41,
        guidanceStrength:68,
      }
    },

    // Audio analysis (0–100 scores)
    audio: {
      confidence: 81,
      vocalStress: 29,
      instability: 21,
      paceControl: 74,
      clarity:     87,
      segmentCount: 16,
    },

    // Topics (open-ended array)
    topics: [
      { label: 'Azure',        sentiment: 'pos' },   // 'pos' | 'neg' | 'warn' | 'neut'
      { label: 'Copilot',      sentiment: 'pos' },
      { label: 'AI Infra',     sentiment: 'warn' },
      { label: 'Capex',        sentiment: 'warn' },
      { label: 'Cloud Margins',sentiment: 'neg' },
      { label: 'M365',         sentiment: 'pos' },
    ],

    // Historical quarters
    history: [
      {
        quarter: 'Q4 24',
        toneDots: ['pos','pos','pos','warn','pos'],  // 5 tone signals
        return1W: 3.8,
      },
      // ... 5 more quarters
    ],

    // Summary text (supports basic HTML: <strong>)
    summary: 'Microsoft beat on EPS and revenue, guided above consensus, and Azure growth re-accelerated. Management sounded confident in prepared remarks but became <strong>notably cautious on capex scale</strong> during Q&A.',
  },
  // ... other tickers
};
```

---

## Scrollbar Styling

```css
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.22); }
```

---

## Accessibility

- All interactive elements get `:focus-visible` ring: `outline: 2px solid var(--color-info); outline-offset: 2px`
- Bias pills and sentiment bars have `aria-label` describing their value
- Charts have `role="img"` with descriptive `aria-label`
- Color is never the sole differentiator — bias pills also have text, bars also have numeric values
- Minimum text contrast: 4.5:1 against panel backgrounds

---

## DO / DON'T

### DO
- Use CSS variables for every color and spacing value
- Use `var(--font-mono)` for all numeric data
- Add `transition` to all interactive hover states
- Keep panel titles UPPERCASE with letter-spacing
- Use semantic color (bull/bear/warn) consistently — if it's a positive return, it's always `--color-bull`
- Animate bars and confidence fills on page load

### DON'T
- Don't use `Inter`, `Roboto`, `Arial`, or `system-ui` — use `Syne` + `IBM Plex Mono`
- Don't hardcode hex values — always use tokens
- Don't add gradients or glow effects — flat surfaces only
- Don't center-align data tables or metric rows
- Don't show modals with `display:none` toggling (use CSS classes + transitions)
- Don't nest scrollable containers inside scrollable containers
