(function () {
  const sidebar = document.getElementById('sidebar');
  const content = document.getElementById('contentArea');
  const dateLabel = document.getElementById('dateLabel');
  const topicCorrelationHorizon = '5';
  let topicPage = 0;
  const topicPageSize = 4;

  const fmtPct = (value, digits = 2) => value === null || value === undefined ? 'n/a' : `${value > 0 ? '+' : ''}${Number(value).toFixed(digits)}%`;
  const fmtScore = (value) => value === null || value === undefined ? 'n/a' : Number(value).toFixed(1);
  const fmtNum = (value) => value === null || value === undefined ? 'n/a' : Number(value).toFixed(2);
  const textClass = (value) => value > 0 ? 'text-bull' : value < 0 ? 'text-bear' : 'text-sec';
  const corrClass = (value) => {
    if (value === null || value === undefined || Math.abs(Number(value)) < 0.1) return 'corr-flat';
    return Number(value) > 0 ? 'corr-pos' : 'corr-neg';
  };
  const pillClass = (bias) => bias.includes('BEAR') ? 'pill-bear' : bias.includes('BULL') ? 'pill-bull' : 'pill-neutral';
  const displayTilt = (bias) => bias.includes('BEAR') ? 'Cautious tilt' : bias.includes('BULL') ? 'Constructive tilt' : 'Balanced';
  const confClass = (bias) => bias.includes('BEAR') ? 'bear' : bias.includes('BULL') ? 'bull' : 'neut';
  const dirClass = (dir) => dir === 'up' ? 'up' : dir === 'down' ? 'down' : 'flat';
  const dirArrow = (dir) => dir === 'up' ? 'UP' : dir === 'down' ? 'DOWN' : 'FLAT';

  document.getElementById('backBtn').addEventListener('click', () => {
    window.location.href = 'index.html';
  });

  function activeSymbol() {
    const params = new URLSearchParams(window.location.search);
    const requested = (params.get('ticker') || 'MSFT').toUpperCase();
    return window.TICKER_DATA[requested] ? requested : 'MSFT';
  }

  function confidenceBar(ticker) {
    return `
      <div class="conf-bar-wrap">
        <div class="conf-label"><span>Confidence</span><span>${ticker.confidence}/100</span></div>
        <div class="conf-track"><div class="conf-fill ${confClass(ticker.bias)}" style="--target-width:${ticker.confidence}%"></div></div>
      </div>
    `;
  }

  function horizons(ticker) {
    return `
      <div class="horizon-row">
        ${ticker.horizons.map((horizon) => `
          <div class="horizon-chip">
            <span class="hc-period">${horizon.period}</span>
            <span class="hc-dir ${dirClass(horizon.direction)}">${dirArrow(horizon.direction)}</span>
          </div>
        `).join('')}
      </div>
    `;
  }

  function decisionCells(ticker) {
    const inputs = ticker.decisionInputs;
    const cells = [
      ['Final score', fmtScore(ticker.finalScore), ticker.finalScore],
      ['EPS surprise', fmtPct(inputs.epsSurprisePct), inputs.epsSurprisePct],
      ['Revenue surprise', fmtPct(inputs.revenueSurprisePct), inputs.revenueSurprisePct],
      ['Guidance rev.', fmtPct(inputs.guidanceRevenueSurprisePct), inputs.guidanceRevenueSurprisePct],
      ['Model up prob.', `${ticker.probBull}%`, ticker.probBull - 50],
    ];
    return `
      <div class="decision-list">
        ${cells.map(([key, value, signal]) => `
          <div class="decision-row">
            <div class="decision-key">${key}</div>
            <div class="decision-val ${textClass(signal || 0)}">${value}</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  function earningsRows(ticker) {
    const inputs = ticker.decisionInputs;
    const rows = [
      ['EPS', fmtNum(ticker.fundamentals.epsActual), fmtNum(ticker.fundamentals.epsEstimate), inputs.epsSurprisePct],
      ['Revenue', ticker.fundamentals.revenueActual, ticker.fundamentals.revenueEstimate, inputs.revenueSurprisePct],
      ['Guidance rev.', ticker.fundamentals.guidanceRevenueMid, ticker.fundamentals.guidanceRevenueConsensus, inputs.guidanceRevenueSurprisePct],
      ['Net margin', fmtPct(inputs.netMarginPct), 'latest', inputs.netMarginPct],
      ['FCF margin', fmtPct(inputs.fcfMarginPct), ticker.fundamentals.freeCashFlow, inputs.fcfMarginPct],
      ['Revenue growth', fmtPct(inputs.revenueGrowthPct), 'QoQ', inputs.revenueGrowthPct],
    ];
    return rows.map(([key, actual, estimate, surprise]) => `
      <div class="eps-row">
        <div class="eps-key">${key}</div>
        <div class="eps-pair">
          <span class="eps-actual">${actual}</span>
          <span class="eps-est">${estimate}</span>
          ${surprise === null || surprise === undefined ? '' : `<span class="eps-surp ${surprise > 0 ? 'surp-pos' : surprise < 0 ? 'surp-neg' : 'surp-flat'}">${fmtPct(surprise, 1)}</span>`}
        </div>
      </div>
    `).join('');
  }

  function meterRows(metrics) {
    const config = [
      ['Positive', 'positiveLang', 'fill-pos'],
      ['Negative', 'negativeLang', 'fill-neg'],
      ['Risk', 'riskLanguage', 'fill-warn'],
      ['Uncertainty', 'uncertainty', 'fill-warn'],
      ['Pressure', 'analystPressure', 'fill-info'],
      ['Defensive', 'defensiveLang', 'fill-accent'],
      ['Guidance', 'guidanceStrength', 'fill-pos'],
    ];
    return `<div class="sentiment-meters">${config.map(([label, key, cls]) => `
      <div class="sent-row">
        <div class="sent-key">${label}</div>
        <div class="sent-track"><div class="sent-fill ${cls}" style="width:${metrics[key]}%"></div></div>
        <div class="sent-val">${metrics[key]}</div>
      </div>
    `).join('')}</div>`;
  }

  function overallCallMetrics(call) {
    const items = [
      ['Sentiment', call.overall.sentiment, 'text-bull'],
      ['Risk', call.overall.risk, 'text-warn'],
      ['Uncertainty', call.overall.uncertainty, 'text-warn'],
      ['Defensive', call.overall.defensiveness, 'text-info'],
      ['Analyst pressure', call.overall.analystPressure, 'text-info'],
      ['Neg/mixed turns', call.overall.negativeMixed, 'text-bear'],
    ];
    return `
      <div class="mini-stat-grid">
        ${items.map(([label, value, cls]) => `
          <div class="mini-stat">
            <div class="ms-label">${label}</div>
            <div class="ms-val ${cls}">${value}</div>
            <div class="ms-sub">0-100</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  function audioPanel(call) {
    const audio = call.audio;
    if (!audio.available) {
      return `<div class="no-audio">${audio.source}. Transcript and topic data still appear where available.</div>`;
    }
    const items = [
      ['Confidence', audio.confidence, 'var(--color-bull)'],
      ['Stress', audio.vocalStress, 'var(--color-warn)'],
      ['Instability', audio.instability, 'var(--color-bear)'],
      ['Pace', audio.paceControl, 'var(--color-info)'],
      ['Clarity', audio.clarity, 'var(--color-accent)'],
    ];
    return `
      <div class="audio-stack">
        ${items.map(([label, value, color]) => {
          const blocks = Array.from({ length: 12 }, (_, index) => {
            const opacity = Math.max(0.18, Math.min(1, (value / 100) * (0.45 + index / 14)));
            return `<span class="audio-block" style="background:${color}; opacity:${opacity.toFixed(2)}"></span>`;
          }).join('');
          return `
            <div class="audio-row">
              <div class="audio-key">${label}</div>
              <div class="audio-seg">${blocks}</div>
              <div class="audio-score">${value}</div>
            </div>
          `;
        }).join('')}
      </div>
      <div class="audio-source">${audio.source} · ${audio.segmentCount} Q&amp;A segments</div>
    `;
  }

  function sentimentHorizonPanel(ticker) {
    const series = ticker.sentimentHorizon || [];
    if (!series.length) {
      return `
        <div class="chart-tooltip-row">
          <span>Tone/return correlation by horizon</span>
          <span class="info-dot" title="Shows whether more positive call tone historically moved with later benchmark-relative returns. Positive rho means tone and excess return moved together; negative rho means they moved opposite.">?</span>
        </div>
        <div class="empty-state">Sentiment-by-horizon correlation is not available for this ticker in the current call-analysis snapshot.</div>
      `;
    }

    const width = 360;
    const height = 150;
    const pad = { left: 38, right: 16, top: 16, bottom: 30 };
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;
    const minY = -1;
    const maxY = 1;
    const xFor = (index) => pad.left + (innerW * index) / Math.max(1, series.length - 1);
    const yFor = (rho) => pad.top + ((maxY - rho) / (maxY - minY)) * innerH;
    const points = series.map((d, index) => `${xFor(index)},${yFor(d.rho ?? 0)}`).join(' ');
    const zeroY = yFor(0);

    return `
      <div class="chart-tooltip-row">
        <span>Tone vs later excess return</span>
        <span class="info-dot" title="Positive rho means more positive call tone historically moved with later benchmark-relative outperformance at that horizon. Negative rho means the relationship moved the other way. Small-sample context only.">?</span>
      </div>
      <div class="horizon-chart-wrap">
        <svg class="horizon-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Sentiment correlation by horizon">
          <line x1="${pad.left}" x2="${width - pad.right}" y1="${zeroY}" y2="${zeroY}" class="chart-zero"></line>
          <line x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" class="chart-axis"></line>
          <polyline points="${points}" class="chart-line"></polyline>
          ${series.map((d, index) => {
            const x = xFor(index);
            const y = yFor(d.rho ?? 0);
            return `
              <g>
                <circle cx="${x}" cy="${y}" r="4.5" class="chart-dot ${d.rho >= 0 ? 'pos' : 'neg'}"></circle>
                <text x="${x}" y="${height - 12}" text-anchor="middle" class="chart-label">${d.horizonDays}d</text>
                <text x="${x}" y="${y - 10}" text-anchor="middle" class="chart-value">${d.rho === null ? 'n/a' : d.rho.toFixed(2)}</text>
              </g>
            `;
          }).join('')}
          <text x="8" y="${pad.top + 5}" class="chart-y-label">+1</text>
          <text x="12" y="${zeroY + 4}" class="chart-y-label">0</text>
          <text x="8" y="${height - pad.bottom}" class="chart-y-label">-1</text>
        </svg>
      </div>
      <div class="chart-meta">n=${series[0]?.nEvents ?? 'n/a'} earnings-call events · Pearson rho · excess returns by horizon</div>
    `;
  }

  function topicChips(topics) {
    return `
      <div class="topic-grid">
        ${topics.map((topic) => `<button class="topic-chip topic-${topic.sentiment} ${topic.sharedWith?.length ? 'shared' : ''}" type="button" data-topic-open="${topic.label}" data-shared="${topic.sharedWith?.length ? `+${topic.sharedWith.length}` : ''}" title="${topic.sharedWith?.length ? `Also: ${topic.sharedWith.join(', ')}` : ''}">${topic.label}</button>`).join('')}
      </div>
    `;
  }

  function pastReactions(ticker) {
    const rows = ticker.pastReactions || [];
    if (!rows.length) return '<div class="empty-state">No past-quarter reaction history available for this ticker.</div>';
    const chartRows = [...rows].reverse();
    const width = 720;
    const height = 230;
    const pad = { left: 44, right: 20, top: 18, bottom: 42 };
    const vals = chartRows.flatMap((row) => [row.openReturnPct, row.oneDayReturnPct, row.oneWeekReturnPct, row.oneMonthReturnPct]).filter((v) => v !== null && v !== undefined);
    const maxAbs = Math.max(5, ...vals.map((v) => Math.abs(v)));
    const xFor = (index) => pad.left + ((width - pad.left - pad.right) * index) / Math.max(1, chartRows.length - 1);
    const yFor = (value) => pad.top + ((maxAbs - value) / (maxAbs * 2)) * (height - pad.top - pad.bottom);
    const zeroY = yFor(0);
    const series = [
      ['Open', 'openReturnPct', 'rgba(126,157,229,0.78)'],
      ['1D', 'oneDayReturnPct', 'rgba(210,162,86,0.78)'],
      ['1W', 'oneWeekReturnPct', 'rgba(166,214,180,0.78)'],
      ['1M', 'oneMonthReturnPct', 'rgba(230,164,164,0.72)'],
    ];
    return `
      <div class="chart-reading-note">Prior-quarter actual reaction windows are shown for calibration. The latest event’s forward reaction is not plotted here.</div>
      <div class="horizon-chart-wrap">
        <svg class="horizon-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Past quarter reaction chart">
          <line x1="${pad.left}" x2="${width - pad.right}" y1="${zeroY}" y2="${zeroY}" class="chart-zero"></line>
          <line x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" class="chart-axis"></line>
          ${series.map(([label, key, color]) => {
            const pts = chartRows.map((row, index) => `${xFor(index)},${yFor(row[key] ?? 0)}`).join(' ');
            return `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></polyline>`;
          }).join('')}
          ${chartRows.map((row, index) => `<text x="${xFor(index)}" y="${height - 14}" text-anchor="middle" class="chart-label">${row.quarter || row.reportDate}</text>`).join('')}
        </svg>
      </div>
      <div class="legend-row">
        ${series.map(([label,,color]) => `<span class="legend-item"><span class="hist-dot" style="background:${color}"></span>${label}</span>`).join('')}
      </div>
    `;
  }

  function callHistory(call) {
    if (!call?.history?.length) return '';
    const rows = call.history;
    const width = 720;
    const height = 230;
    const pad = { left: 44, right: 20, top: 18, bottom: 42 };
    const returns = rows.map((row) => row.excessReturn5d || 0);
    const maxAbs = Math.max(5, ...returns.map((v) => Math.abs(v)));
    const xFor = (index) => pad.left + ((width - pad.left - pad.right) * index) / Math.max(1, rows.length - 1);
    const yRet = (value) => pad.top + ((maxAbs - value) / (maxAbs * 2)) * (height - pad.top - pad.bottom);
    const yScore = (value) => pad.top + ((100 - value) / 100) * (height - pad.top - pad.bottom);
    const zeroY = yRet(0);
    const sentimentPts = rows.map((row, index) => `${xFor(index)},${yScore(row.sentiment)}`).join(' ');
    const riskPts = rows.map((row, index) => `${xFor(index)},${yScore(row.risk)}`).join(' ');
    return `
      <div class="chart-reading-note">Historical call tone context compares prior call sentiment/risk with prior 5D excess return. Bars are past excess return; lines are 0-100 call-tone scores.</div>
      <div class="horizon-chart-wrap">
        <svg class="horizon-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Past call tone versus actual outcome">
          <line x1="${pad.left}" x2="${width - pad.right}" y1="${zeroY}" y2="${zeroY}" class="chart-zero"></line>
          <line x1="${pad.left}" x2="${pad.left}" y1="${pad.top}" y2="${height - pad.bottom}" class="chart-axis"></line>
          ${rows.map((row, index) => {
            const x = xFor(index) - 9;
            const y = Math.min(yRet(row.excessReturn5d || 0), zeroY);
            const h = Math.abs(yRet(row.excessReturn5d || 0) - zeroY);
            return `<rect x="${x}" y="${y}" width="18" height="${Math.max(2, h)}" rx="2" fill="${(row.excessReturn5d || 0) >= 0 ? 'rgba(166,214,180,0.55)' : 'rgba(230,164,164,0.55)'}"></rect>`;
          }).join('')}
          <polyline points="${sentimentPts}" class="chart-line"></polyline>
          <polyline points="${riskPts}" fill="none" stroke="rgba(210,162,86,0.76)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"></polyline>
          ${rows.map((row, index) => `<text x="${xFor(index)}" y="${height - 14}" text-anchor="middle" class="chart-label">${row.quarter.replace('_20', '’')}</text>`).join('')}
        </svg>
      </div>
      <div class="legend-row">
        <span class="legend-item"><span class="hist-dot" style="background:rgba(166,214,180,0.55)"></span>5D excess return</span>
        <span class="legend-item"><span class="hist-dot" style="background:rgba(126,157,229,0.76)"></span>Sentiment</span>
        <span class="legend-item"><span class="hist-dot" style="background:rgba(210,162,86,0.76)"></span>Risk</span>
      </div>
    `;
  }

  function topicDetails(ticker) {
    const sourceRows = ticker.topicDetails || [];
    const rows = [...sourceRows].sort((a, b) => {
      const mentionDelta = (b.mentions || 0) - (a.mentions || 0);
      if (mentionDelta) return mentionDelta;
      const aH = a.horizons?.[topicCorrelationHorizon]?.sentimentCorrelation ?? a.sentimentCorrelation5d ?? a.sentimentScore ?? 0;
      const bH = b.horizons?.[topicCorrelationHorizon]?.sentimentCorrelation ?? b.sentimentCorrelation5d ?? b.sentimentScore ?? 0;
      return Math.abs(bH) - Math.abs(aH);
    });
    if (!rows.length) return '<div class="empty-state">No topic detail available for this ticker.</div>';
    const totalPages = Math.max(1, Math.ceil(rows.length / topicPageSize));
    topicPage = Math.min(topicPage, totalPages - 1);
    const pageRows = rows.slice(topicPage * topicPageSize, topicPage * topicPageSize + topicPageSize);
    return `
      <div class="topic-controls topic-controls-simple">
        <input class="topic-search" id="topicSearch" type="search" placeholder="Search topics..." aria-label="Search topics">
        <select class="topic-select" aria-label="Topic filter">
          <option>All topics</option>
        </select>
      </div>
      <div class="topic-explainer">
        Topic rows combine the latest earnings-report mentions with sentiment, risk, and quality scores. For Mag7 names, the correlation columns use historical per-topic tone against 5-day excess return; for the rest of the universe, topics are mapped business themes for analyst comparison.
      </div>
      <div class="topic-table-frame">
        <table class="audit-table topic-correlation-table">
          <thead><tr><th>Topic</th><th>N</th><th>Mentions</th><th>P Sentiment</th><th>P Risk</th><th>Quality</th></tr></thead>
          <tbody>
            ${pageRows.map((topic) => {
              const h = topic.horizons?.[topicCorrelationHorizon] || {};
              const pSentiment = h.sentimentCorrelation ?? topic.sentimentCorrelation5d ?? topic.sentimentScore;
              const pRisk = h.riskCorrelation ?? topic.riskCorrelation5d ?? topic.riskScore;
              const n = h.nEvents ?? topic.nEvents;
              const mentions = typeof topic.mentions === 'number' ? topic.mentions : null;
              const quality = mentions !== null && mentions > 6 ? 'reliable' : 'indicative';
              return `
              <tr data-topic-row="${topic.label.toLowerCase()}">
                <td>
                  <button class="topic-name-cell topic-open-button" type="button" data-topic-open="${topic.label}">
                    <span>${topic.label}</span>
                  </button>
                </td>
                <td class="mono">${n ?? 'n/a'}</td>
                <td class="mono">${topic.mentions ?? 'mapped'}</td>
                <td class="${corrClass(pSentiment)} mono">${pSentiment === null || pSentiment === undefined ? 'n/a' : Number(pSentiment).toFixed(h.sentimentCorrelation === undefined ? 2 : 3)}</td>
                <td class="${corrClass(pRisk)} mono">${pRisk === null || pRisk === undefined ? 'n/a' : Number(pRisk).toFixed(h.riskCorrelation === undefined ? 2 : 3)}</td>
                <td class="quality-text ${quality}">${quality}</td>
              </tr>
            `}).join('')}
          </tbody>
        </table>
      </div>
      <div class="table-pager">
        <button class="pager-btn" type="button" data-topic-page="prev" ${topicPage === 0 ? 'disabled' : ''}>Prev</button>
        <span class="pager-state mono">Page ${topicPage + 1} / ${totalPages}</span>
        <button class="pager-btn" type="button" data-topic-page="next" ${topicPage >= totalPages - 1 ? 'disabled' : ''}>Next</button>
      </div>
    `;
  }

  function escapeHtml(value) {
    return String(value || '').replace(/[&<>"']/g, (ch) => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;',
    }[ch]));
  }

  function topicTerms(label) {
    const stop = new Set(['and', 'the', 'for', 'with', 'from', 'into', 'outlook', 'quality', 'growth', 'performance', 'initiatives', 'solutions']);
    return label
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter((term) => term.length > 2 && !stop.has(term));
  }

  function toneTerms() {
    return [
      'accelerating', 'better', 'bullish', 'confidence', 'confident', 'demand', 'differentiated',
      'encouraged', 'encouraging', 'excited', 'exciting', 'expansionary', 'favorable', 'good',
      'great', 'impressed', 'improved', 'improving', 'momentum', 'opportunity', 'pleased',
      'positive', 'powerful', 'record', 'resonating', 'satisfaction', 'scaling', 'strong',
      'substantial', 'up significantly', 'vibrant',
      'challenge', 'challenging', 'constrained', 'constraints', 'decline', 'difficult', 'down',
      'headwind', 'issue', 'limited', 'negative', 'pressure', 'risk', 'slower', 'uncertain',
      'uncertainty', 'volatile', 'weakness'
    ];
  }

  function splitSentences(text) {
    return String(text || '').match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];
  }

  function highlightTranscriptText(text, topicMatches, toneMatches) {
    const topicSet = new Set(topicMatches.map((term) => term.toLowerCase()));
    const toneSet = new Set(toneMatches.map((term) => term.toLowerCase()));
    const terms = [...topicSet, ...toneSet]
      .filter(Boolean)
      .sort((a, b) => b.length - a.length)
      .map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));

    if (!terms.length) return escapeHtml(text);

    const pattern = new RegExp(`\\b(${terms.join('|')})\\b`, 'gi');
    let cursor = 0;
    let html = '';
    for (const match of String(text || '').matchAll(pattern)) {
      const value = match[0];
      const start = match.index ?? 0;
      const lower = value.toLowerCase();
      html += escapeHtml(String(text).slice(cursor, start));
      html += `<mark class="${toneSet.has(lower) ? 'tone-highlight' : 'topic-highlight'}">${escapeHtml(value)}</mark>`;
      cursor = start + value.length;
    }
    return html + escapeHtml(String(text).slice(cursor));
  }

  function transcriptModal() {
    return `
      <div class="topic-modal" id="topicModal" hidden>
        <div class="topic-modal-backdrop" data-topic-close></div>
        <div class="topic-modal-panel" role="dialog" aria-modal="true" aria-labelledby="topicModalTitle">
          <div class="topic-modal-head">
            <div>
              <div class="section-label">Topic transcript</div>
              <h2 id="topicModalTitle">Topic</h2>
              <p id="topicModalSub"></p>
            </div>
            <button class="modal-close" type="button" data-topic-close>Close</button>
          </div>
          <div class="topic-modal-body" id="topicModalBody"></div>
        </div>
      </div>
    `;
  }

  function openTopicModal(ticker, label) {
    const modal = document.getElementById('topicModal');
    const title = document.getElementById('topicModalTitle');
    const sub = document.getElementById('topicModalSub');
    const body = document.getElementById('topicModalBody');
    if (!modal || !title || !sub || !body) return;

    const transcript = ticker.transcript;
    title.textContent = label;
    sub.textContent = transcript?.exchanges?.length
      ? `${ticker.symbol} · ${transcript.date} · relevant Q&A transcript passages`
      : `${ticker.symbol} · transcript text is not available for this ticker`;

    if (!transcript?.exchanges?.length) {
      body.innerHTML = '<div class="empty-state">No transcript text is available in this repo snapshot for this ticker/topic.</div>';
      modal.hidden = false;
      return;
    }

    const terms = topicTerms(label);
    const tone = toneTerms();
    const passages = [];
    for (const exchange of transcript.exchanges) {
      const sentences = splitSentences(exchange.text);
      const matched = sentences.filter((sentence) => terms.some((term) => sentence.toLowerCase().includes(term)));
      if (matched.length) {
        passages.push({ exchange, sentences: matched.slice(0, 4) });
      }
    }

    if (!passages.length) {
      body.innerHTML = `
        <div class="empty-state">No exact topic-term matches were found in the available Q&A transcript. Try a broader mapped topic or related ticker.</div>
      `;
      modal.hidden = false;
      return;
    }

    body.innerHTML = passages.slice(0, 12).map(({ sentences }) => `
      <article class="transcript-hit">
        <p>${sentences.map((sentence) => highlightTranscriptText(sentence.trim(), terms, tone)).join(' ')}</p>
      </article>
    `).join('');
    modal.hidden = false;
  }

  function renderSidebar(ticker) {
    sidebar.innerHTML = `
      <div class="ticker-identity">
        <div class="ticker-symbol-large display">${ticker.symbol}</div>
        <div class="ticker-company">${ticker.name}</div>
        <div class="bias-pill ${pillClass(ticker.bias)}" style="width:max-content">${displayTilt(ticker.bias)}</div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div><div class="panel-title">Model Tilt</div><div class="panel-sub">${ticker.latestQuarter || 'latest event'}</div></div>
          <div class="panel-badge">${ticker.reportDate || 'latest'}</div>
        </div>
        ${confidenceBar(ticker)}
        <div class="prob-row">
          <div class="prob-item"><div class="prob-val bull">${ticker.probBull}%</div><div class="prob-key">Up</div></div>
          <div class="prob-item"><div class="prob-val bear">${ticker.probBear}%</div><div class="prob-key">Down</div></div>
          <div class="prob-item"><div class="prob-val neut">${fmtScore(ticker.mlScore)}</div><div class="prob-key">ML score</div></div>
        </div>
        ${horizons(ticker)}
      </div>
      <div class="panel">
        <div class="panel-header">
          <div><div class="panel-title">Mapped Topics</div><div class="panel-sub">Shared themes marked +N</div></div>
        </div>
        ${topicChips(ticker.topics || [])}
      </div>
    `;
  }

  function renderContent(ticker) {
    const call = ticker.callAnalysis;
    content.innerHTML = `
      <div class="summary-banner">
        <div class="banner-label">Setup</div>
        <div class="summary-text"><strong>${ticker.symbol}</strong> is shown with the latest earnings setup, model tilt, topic context, and call-analysis signals where available. The intent is to give analysts enough structured evidence to form their own view.</div>
      </div>

      <div class="panel-grid">
        <section class="panel span-6">
          <div class="panel-header">
            <div><div class="panel-title">Decision Inputs</div><div class="panel-sub">Current event data available to analyst before forming a view</div></div>
            <div class="panel-badge">setup</div>
          </div>
          ${decisionCells(ticker)}
        </section>

        <section class="panel span-6">
          <div class="panel-header">
            <div><div class="panel-title">Earnings Inputs</div><div class="panel-sub">Financial setup and consensus deltas</div></div>
            <div class="panel-badge">${ticker.latestQuarter || 'latest'}</div>
          </div>
          ${earningsRows(ticker)}
        </section>

        <section class="panel span-12 topic-focus-panel">
          <div class="panel-header">
            <div><div class="panel-title">Topic Correlations</div><div class="panel-sub">Per-topic sentiment/risk, mentions, quality, and related tickers</div></div>
            <div class="panel-badge">${(ticker.topicDetails || []).length}</div>
          </div>
          ${topicDetails(ticker)}
        </section>

        ${call ? `
          <section class="panel span-12 tone-emphasis-panel">
            <div class="panel-header">
              <div><div class="panel-title">Prepared vs Q&amp;A</div><div class="panel-sub">Sentiment, risk, pressure, defensiveness</div></div>
              <div class="panel-badge">FinBERT + LLM</div>
            </div>
            <div class="tone-split">
              <div><div class="subhead">Prepared remarks</div>${meterRows(call.prepared)}</div>
              <div><div class="subhead">Analyst Q&amp;A</div>${meterRows(call.qa)}</div>
            </div>
          </section>

          <section class="panel span-7">
            <div class="panel-header">
              <div><div class="panel-title">Sentiment By Horizon</div><div class="panel-sub">Whole-call tone correlation with later excess return</div></div>
              <div class="panel-badge">${(ticker.sentimentHorizon || []).length ? 'available' : 'placeholder'}</div>
            </div>
            ${sentimentHorizonPanel(ticker)}
          </section>

          <section class="panel span-5">
            <div class="panel-header">
              <div><div class="panel-title">Overall Call Tone</div><div class="panel-sub">${call.turnCount} labeled turns · ${call.callDate || 'latest call'}</div></div>
              <div class="panel-badge">text</div>
            </div>
            ${overallCallMetrics(call)}
          </section>
        ` : `
          <section class="panel span-7">
            <div class="panel-header">
              <div><div class="panel-title">Sentiment By Horizon</div><div class="panel-sub">Whole-call tone correlation with later excess return</div></div>
              <div class="panel-badge">${(ticker.sentimentHorizon || []).length ? 'available' : 'placeholder'}</div>
            </div>
            ${sentimentHorizonPanel(ticker)}
          </section>

          <section class="panel span-5">
            <div class="panel-header">
              <div><div class="panel-title">Call Analysis</div><div class="panel-sub">Not available for this ticker in current snapshot</div></div>
              <div class="panel-badge">earnings stats only</div>
            </div>
            <div class="no-audio">This ticker has earnings statistics, model scores, fundamentals, and mapped topics. Audio/transcript analysis is currently available only for the Mag7 call-analysis set.</div>
          </section>
        `}

        <section class="panel span-12">
          <div class="panel-header">
            <div><div class="panel-title">Past Quarter Actual Reactions</div><div class="panel-sub">Historical reaction windows for prior quarters</div></div>
            <div class="panel-badge">past actuals</div>
          </div>
          ${pastReactions(ticker)}
        </section>

        ${call ? `
          <section class="panel span-12">
            <div class="panel-header">
              <div><div class="panel-title">Past Call Tone vs Actual Outcome</div><div class="panel-sub">Historical tone context against past 5D excess return</div></div>
              <div class="panel-badge">past only</div>
            </div>
            ${callHistory(call)}
          </section>
        ` : ''}
      </div>
      ${transcriptModal()}
    `;
  }

  function renderDashboard(symbol) {
    const ticker = window.TICKER_DATA[symbol];
    document.title = `${symbol} · AlphaSights Earnings Intelligence`;
    dateLabel.textContent = ticker.reportDate || 'EARNINGS INTEL';
    renderSidebar(ticker);
    renderContent(ticker);
    attachTickerInteractions(ticker);
  }

  function attachTickerInteractions(ticker) {
    const input = document.getElementById('topicSearch');
    if (input) {
      input.addEventListener('input', () => {
        const query = input.value.trim().toLowerCase();
        document.querySelectorAll('[data-topic-row]').forEach((row) => {
          row.style.display = row.dataset.topicRow.includes(query) ? '' : 'none';
        });
      });
    }

    document.querySelectorAll('[data-topic-page]').forEach((button) => {
      button.addEventListener('click', () => {
        topicPage += button.dataset.topicPage === 'next' ? 1 : -1;
        renderDashboard(ticker.symbol);
      });
    });

    document.querySelectorAll('[data-topic-open]').forEach((button) => {
      button.addEventListener('click', () => openTopicModal(ticker, button.dataset.topicOpen));
    });

    document.querySelectorAll('[data-topic-close]').forEach((button) => {
      button.addEventListener('click', () => {
        const modal = document.getElementById('topicModal');
        if (modal) modal.hidden = true;
      });
    });
  }

  window.addEventListener('popstate', () => renderDashboard(activeSymbol()));
  renderDashboard(activeSymbol());
})();
