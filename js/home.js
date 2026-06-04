(function () {
  const tabs = document.getElementById('sectorTabs');
  const list = document.getElementById('tickerList');
  const statTickerCount = document.getElementById('statTickerCount');
  let activeFilter = 'all';

  const fmtPct = (value, digits = 1) => value === null || value === undefined ? 'n/a' : `${value > 0 ? '+' : ''}${Number(value).toFixed(digits)}%`;
  const fmtScore = (value) => value === null || value === undefined ? 'n/a' : Number(value).toFixed(1);
  const textClass = (value) => value > 0 ? 'text-bull' : value < 0 ? 'text-bear' : 'text-sec';
  const pillClass = (bias) => bias.includes('BEAR') ? 'pill-bear' : bias.includes('BULL') ? 'pill-bull' : 'pill-neutral';
  const displayTilt = (bias) => bias.includes('BEAR') ? 'Cautious' : bias.includes('BULL') ? 'Constructive' : 'Balanced';

  function allSymbols() {
    return Object.keys(window.TICKER_DATA).sort((a, b) => {
      const amag = window.TICKER_DATA[a].hasCallAnalysis ? 0 : 1;
      const bmag = window.TICKER_DATA[b].hasCallAnalysis ? 0 : 1;
      if (amag !== bmag) return amag - bmag;
      return a.localeCompare(b);
    });
  }

  function filters() {
    return [
      { id: 'call-analysis', label: 'Mag7' },
      ...window.SECTIONS.filter((section) => section.id !== 'call-analysis').map((section) => ({
        id: section.id,
        label: section.label.replace('Software, AI & Cloud', 'Software').replace('Consumer & Retail', 'Consumer'),
      })),
    ];
  }

  function renderTabs() {
    tabs.innerHTML = filters().map((filter) => `
      <button class="sector-tab ${activeFilter === filter.id ? 'active' : ''}" data-sector="all" data-filter="${filter.id}" type="button">${filter.label}</button>
    `).join('');
    tabs.querySelectorAll('button').forEach((button) => {
      button.addEventListener('click', () => {
        activeFilter = button.dataset.filter;
        renderList();
        renderTabs();
      });
    });
  }

  function visibleSymbols() {
    if (activeFilter === 'all') return allSymbols();
    return allSymbols().filter((symbol) => window.TICKER_DATA[symbol].sector === activeFilter);
  }

  function tickerRow(symbol, index) {
    const ticker = window.TICKER_DATA[symbol];
    const inputs = ticker.decisionInputs;
    const topTopics = (ticker.topics || []).slice(0, 4);
    return `
      <a class="ticker-list-row" href="ticker.html?ticker=${symbol}" style="animation-delay:${Math.min(index, 12) * 35}ms">
        <div class="ticker-row-id">
          <div class="compact-symbol">${symbol}</div>
          <div class="compact-name">${ticker.name}</div>
        </div>
        <div class="ticker-row-bias">
          <span class="bias-pill ${pillClass(ticker.bias)}">${displayTilt(ticker.bias)}</span>
          <span class="row-sub mono">${ticker.confidence}/100 conf.</span>
        </div>
        <div class="ticker-row-metric">
          <span class="row-key">Final score</span>
          <strong class="${textClass(ticker.finalScore || 0)}">${fmtScore(ticker.finalScore)}</strong>
        </div>
        <div class="ticker-row-metric">
          <span class="row-key">EPS</span>
          <strong class="${textClass(inputs.epsSurprisePct || 0)}">${fmtPct(inputs.epsSurprisePct)}</strong>
        </div>
        <div class="ticker-row-metric">
          <span class="row-key">Revenue</span>
          <strong class="${textClass(inputs.revenueSurprisePct || 0)}">${fmtPct(inputs.revenueSurprisePct)}</strong>
        </div>
        <div class="ticker-row-topics">
          ${topTopics.map((topic) => `<span class="topic-chip topic-${topic.sentiment} ${topic.sharedWith?.length ? 'shared' : ''}" data-shared="${topic.sharedWith?.length ? `+${topic.sharedWith.length}` : ''}">${topic.label}</span>`).join('')}
        </div>
        <div class="ticker-row-arrow">→</div>
      </a>
    `;
  }

  function renderList() {
    const symbols = visibleSymbols();
    list.innerHTML = `
      <div class="section-tools">
        <div>
          <div class="sector-header" style="margin-bottom:4px">
            <span class="sector-dot" style="background:${activeFilter === 'call-analysis' ? 'var(--color-info)' : 'var(--color-accent)'}"></span>
            <span class="sector-name">${activeFilter === 'all' ? 'Ticker universe' : filters().find((f) => f.id === activeFilter)?.label}</span>
            <span class="sector-count">${symbols.length}</span>
          </div>
          <div class="section-copy">Mag7 names stay at the top because they have call text/audio analysis. Other tickers still include earnings statistics, model tilt, fundamentals, and mapped business topics.</div>
        </div>
      </div>
      <div class="ticker-list">
        ${symbols.map(tickerRow).join('')}
      </div>
    `;
  }

  function render() {
    statTickerCount.textContent = Object.values(window.TICKER_DATA).filter((ticker) => ticker.finalScore !== null && ticker.finalScore !== undefined).length;
    renderTabs();
    renderList();
  }

  render();
})();
