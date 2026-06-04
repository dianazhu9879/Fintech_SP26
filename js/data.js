window.SECTIONS = [
  {
    "id": "call-analysis",
    "label": "Mag7 Call Analysis",
    "color": "#5a8aff",
    "tickers": [
      "MSFT",
      "NVDA",
      "GOOGL",
      "META",
      "AMZN",
      "AAPL",
      "TSLA"
    ]
  },
  {
    "id": "software-cloud",
    "label": "Software, AI & Cloud",
    "color": "#a37aff",
    "tickers": [
      "PLTR",
      "GTLB",
      "DDOG",
      "DOCN",
      "CRWD",
      "NET",
      "APP",
      "ORCL",
      "CRM",
      "NOW",
      "SNOW",
      "ZS",
      "AI"
    ]
  },
  {
    "id": "semis-hardware",
    "label": "Semis & Hardware",
    "color": "#5a8aff",
    "tickers": [
      "MU",
      "AMD",
      "AVGO",
      "QCOM",
      "AMAT",
      "SMCI"
    ]
  },
  {
    "id": "consumer-retail",
    "label": "Consumer & Retail",
    "color": "#4ecb71",
    "tickers": [
      "KO",
      "WMT",
      "CMG",
      "COST",
      "PG",
      "MCD",
      "ABNB"
    ]
  },
  {
    "id": "financials",
    "label": "Financials",
    "color": "#f0a030",
    "tickers": [
      "V",
      "JPM",
      "XYZ"
    ]
  },
  {
    "id": "healthcare",
    "label": "Healthcare",
    "color": "#ff5757",
    "tickers": [
      "LLY",
      "VRTX",
      "UNH",
      "MRK"
    ]
  }
];

window.TICKER_DATA = {
  "MU": {
    "symbol": "MU",
    "name": "MICRON TECHNOLOGY INC",
    "sector": "semis-hardware",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-03-18",
    "bias": "STRONG BULLISH",
    "finalScore": 93.46,
    "overallScore": 93.46,
    "confidence": 95,
    "probBull": 31,
    "probBear": 69,
    "mlScore": -46.15,
    "decisionInputs": {
      "epsSurprisePct": 39.81,
      "revenueSurprisePct": 26.24,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 74.89,
      "netMarginPct": 57.77,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 12.08,
      "epsEstimate": 8.64,
      "revenueActual": "23.9B",
      "revenueEstimate": "18.9B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Semiconductor Cycle",
        "sentiment": "pos",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "QCOM",
          "AMAT",
          "SMCI"
        ],
        "sharedCount": 7
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-17",
        "openReturnPct": 13.75,
        "oneDayReturnPct": 10.21,
        "oneWeekReturnPct": 26.28,
        "oneMonthReturnPct": 72.54
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-23",
        "openReturnPct": -79.92,
        "oneDayReturnPct": -2.82,
        "oneWeekReturnPct": 9.46,
        "oneMonthReturnPct": 24.22
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-06-25",
        "openReturnPct": 1.79,
        "oneDayReturnPct": -98.23,
        "oneWeekReturnPct": -3.9,
        "oneMonthReturnPct": -12.57
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-03-20",
        "openReturnPct": null,
        "oneDayReturnPct": null,
        "oneWeekReturnPct": null,
        "oneMonthReturnPct": null
      }
    ],
    "topicDetails": [
      {
        "label": "Semiconductor Cycle",
        "sentiment": "pos",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "QCOM",
          "AMAT",
          "SMCI"
        ],
        "sharedCount": 7,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 100,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 100,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "PANW": {
    "symbol": "PANW",
    "name": "Palo Alto Networks Inc",
    "sector": "other",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-17",
    "bias": "STRONG BULLISH",
    "finalScore": 81.27,
    "overallScore": 81.27,
    "confidence": 95,
    "probBull": 59,
    "probBear": 41,
    "mlScore": 10.4,
    "decisionInputs": {
      "epsSurprisePct": 38.78,
      "revenueSurprisePct": null,
      "guidanceRevenueSurprisePct": 11.54,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.68,
      "epsEstimate": 0.49,
      "revenueActual": "n/a",
      "revenueEstimate": "2.6B",
      "guidanceRevenueMid": "2.9B",
      "guidanceRevenueConsensus": "2.6B",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Cybersecurity Demand",
        "sentiment": "pos",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "CRWD",
          "ZS",
          "NET"
        ],
        "sharedCount": 4
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-11-19",
        "openReturnPct": -88.79,
        "oneDayReturnPct": -7.42,
        "oneWeekReturnPct": -4.89,
        "oneMonthReturnPct": -5.21
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-08-18",
        "openReturnPct": 6.61,
        "oneDayReturnPct": 3.06,
        "oneWeekReturnPct": 4.58,
        "oneMonthReturnPct": 16.75
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-21",
        "openReturnPct": 96.55,
        "oneDayReturnPct": 2.69,
        "oneWeekReturnPct": 6.16,
        "oneMonthReturnPct": 11.27
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-20",
        "openReturnPct": -5.0,
        "oneDayReturnPct": -6.8,
        "oneWeekReturnPct": -4.45,
        "oneMonthReturnPct": 4.55
      }
    ],
    "topicDetails": [
      {
        "label": "Cybersecurity Demand",
        "sentiment": "pos",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "CRWD",
          "ZS",
          "NET"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 96,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 96,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 96,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "LLY": {
    "symbol": "LLY",
    "name": "ELI LILLY & Co",
    "sector": "healthcare",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-30",
    "bias": "STRONG BULLISH",
    "finalScore": 72.19,
    "overallScore": 72.19,
    "confidence": 92,
    "probBull": 62,
    "probBear": 38,
    "mlScore": 16.97,
    "decisionInputs": {
      "epsSurprisePct": 21.1,
      "revenueSurprisePct": 11.11,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 2.63,
      "netMarginPct": 37.36,
      "fcfMarginPct": 15.19,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 8.55,
      "epsEstimate": 7.06,
      "revenueActual": "19.8B",
      "revenueEstimate": "17.8B",
      "guidanceRevenueMid": "83.5B",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "3.0B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "pos",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "VRTX",
          "MRK",
          "UNH"
        ],
        "sharedCount": 4
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-04",
        "openReturnPct": -3.76,
        "oneDayReturnPct": -7.79,
        "oneWeekReturnPct": -6.22,
        "oneMonthReturnPct": -8.92
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-30",
        "openReturnPct": -63.23,
        "oneDayReturnPct": 2.17,
        "oneWeekReturnPct": 9.46,
        "oneMonthReturnPct": 23.87
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-07",
        "openReturnPct": 2.21,
        "oneDayReturnPct": -2.37,
        "oneWeekReturnPct": 9.42,
        "oneMonthReturnPct": 17.13
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-01",
        "openReturnPct": 3.91,
        "oneDayReturnPct": 3.72,
        "oneWeekReturnPct": -7.5,
        "oneMonthReturnPct": -5.46
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 92,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "pos",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "VRTX",
          "MRK",
          "UNH"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 92,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "PLTR": {
    "symbol": "PLTR",
    "name": "Palantir Technologies Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-04",
    "bias": "STRONG BULLISH",
    "finalScore": 70.94,
    "overallScore": 70.94,
    "confidence": 91,
    "probBull": 48,
    "probBear": 52,
    "mlScore": -12.48,
    "decisionInputs": {
      "epsSurprisePct": 13.64,
      "revenueSurprisePct": 6.01,
      "guidanceRevenueSurprisePct": 5.88,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 16.05,
      "netMarginPct": 53.32,
      "fcfMarginPct": 54.62,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.25,
      "epsEstimate": 0.22,
      "revenueActual": "1.6B",
      "revenueEstimate": "1.5B",
      "guidanceRevenueMid": "1.8B",
      "guidanceRevenueConsensus": "1.7B",
      "freeCashFlow": "891.8M",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Generative AI Products",
        "sentiment": "pos",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "CRM",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "SNOW",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-02",
        "openReturnPct": 11.7,
        "oneDayReturnPct": 6.85,
        "oneWeekReturnPct": -5.58,
        "oneMonthReturnPct": 3.32
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-03",
        "openReturnPct": -7.29,
        "oneDayReturnPct": -7.94,
        "oneWeekReturnPct": -7.83,
        "oneMonthReturnPct": -14.12
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-04",
        "openReturnPct": 6.94,
        "oneDayReturnPct": 7.85,
        "oneWeekReturnPct": 16.38,
        "oneMonthReturnPct": -2.81
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-05",
        "openReturnPct": -8.94,
        "oneDayReturnPct": -12.05,
        "oneWeekReturnPct": 3.5,
        "oneMonthReturnPct": -3.12
      }
    ],
    "topicDetails": [
      {
        "label": "Generative AI Products",
        "sentiment": "pos",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "CRM",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 91,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "SNOW",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 91,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 91,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "GTLB": {
    "symbol": "GTLB",
    "name": "Gitlab Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-03-03",
    "bias": "BULLISH",
    "finalScore": 59.78,
    "overallScore": 59.78,
    "confidence": 87,
    "probBull": 36,
    "probBear": 64,
    "mlScore": -35.81,
    "decisionInputs": {
      "epsSurprisePct": 400,
      "revenueSurprisePct": 3.21,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.06,
      "epsEstimate": -0.02,
      "revenueActual": "260.4M",
      "revenueEstimate": "252.3M",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "DDOG",
          "SNOW",
          "PLTR",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-02",
        "openReturnPct": -10.56,
        "oneDayReturnPct": -12.77,
        "oneWeekReturnPct": -6.62,
        "oneMonthReturnPct": -13.51
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-03",
        "openReturnPct": -7.17,
        "oneDayReturnPct": -7.35,
        "oneWeekReturnPct": 6.03,
        "oneMonthReturnPct": 0.0
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-06-11",
        "openReturnPct": 53.03,
        "oneDayReturnPct": 78.4,
        "oneWeekReturnPct": -6.02,
        "oneMonthReturnPct": -1.08
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-06-10",
        "openReturnPct": -13.46,
        "oneDayReturnPct": -10.6,
        "oneWeekReturnPct": -13.34,
        "oneMonthReturnPct": -12.24
      }
    ],
    "topicDetails": [
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "DDOG",
          "SNOW",
          "PLTR",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 85,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 85,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 85,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "DDOG": {
    "symbol": "DDOG",
    "name": "Datadog, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-07",
    "bias": "BULLISH",
    "finalScore": 56.73,
    "overallScore": 56.73,
    "confidence": 85,
    "probBull": 57,
    "probBear": 43,
    "mlScore": 7.23,
    "decisionInputs": {
      "epsSurprisePct": 183.33,
      "revenueSurprisePct": 4.82,
      "guidanceRevenueSurprisePct": 10.85,
      "guidanceEpsSurprisePct": 41.46,
      "revenueGrowthPct": 5.58,
      "netMarginPct": 5.22,
      "fcfMarginPct": 32.12,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.17,
      "epsEstimate": 0.06,
      "revenueActual": "1.0B",
      "revenueEstimate": "960.1M",
      "guidanceRevenueMid": "1.1B",
      "guidanceRevenueConsensus": "992.3M",
      "freeCashFlow": "323.3M",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Cloud Growth",
        "sentiment": "pos",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "NET",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8
      },
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "SNOW",
          "PLTR",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-10",
        "openReturnPct": 1.15,
        "oneDayReturnPct": -1.8,
        "oneWeekReturnPct": -6.99,
        "oneMonthReturnPct": -3.97
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-06",
        "openReturnPct": -3.42,
        "oneDayReturnPct": 22.01,
        "oneWeekReturnPct": -3.04,
        "oneMonthReturnPct": -20.05
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-07",
        "openReturnPct": 15.4,
        "oneDayReturnPct": -4.01,
        "oneWeekReturnPct": -6.69,
        "oneMonthReturnPct": 2.99
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-06",
        "openReturnPct": -33.0,
        "oneDayReturnPct": -3.77,
        "oneWeekReturnPct": 12.29,
        "oneMonthReturnPct": 15.18
      }
    ],
    "topicDetails": [
      {
        "label": "Cloud Growth",
        "sentiment": "pos",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "NET",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 83,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "SNOW",
          "PLTR",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 83,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 83,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 83,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "GOOGL": {
    "symbol": "GOOGL",
    "name": "Alphabet Inc.",
    "sector": "call-analysis",
    "hasCallAnalysis": true,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-29",
    "bias": "BULLISH",
    "finalScore": 55.79,
    "overallScore": 55.79,
    "confidence": 85,
    "probBull": 50,
    "probBear": 50,
    "mlScore": -7.24,
    "decisionInputs": {
      "epsSurprisePct": 93.56,
      "revenueSurprisePct": 2.73,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -3.45,
      "netMarginPct": 56.94,
      "fcfMarginPct": 9.21,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 5.11,
      "epsEstimate": 2.64,
      "revenueActual": "109.9B",
      "revenueEstimate": "107.0B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "10.1B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "callAnalysis": {
      "period": "Q4_2025",
      "callDate": "2026-02-04",
      "turnCount": 29,
      "overall": {
        "sentiment": 65,
        "confidence": 76,
        "risk": 15,
        "uncertainty": 26,
        "defensiveness": 13,
        "analystPressure": 15,
        "guidanceStrength": 44,
        "negativeMixed": 14
      },
      "prepared": {
        "positiveLang": 77,
        "negativeLang": 3,
        "riskLanguage": 7,
        "uncertainty": 18,
        "analystPressure": 5,
        "defensiveLang": 5,
        "guidanceStrength": 44
      },
      "qa": {
        "positiveLang": 27,
        "negativeLang": 2,
        "riskLanguage": 27,
        "uncertainty": 36,
        "analystPressure": 33,
        "defensiveLang": 25,
        "guidanceStrength": 34
      },
      "topics": [
        {
          "label": "Gemini Models",
          "sentiment": "pos",
          "sentimentScore": 0.36,
          "riskScore": 0.14,
          "negativeMixed": 0
        },
        {
          "label": "AI Infrastructure",
          "sentiment": "warn",
          "sentimentScore": 0.3,
          "riskScore": 0.28,
          "negativeMixed": 0.4,
          "sharedWith": [
            "MSFT",
            "NVDA",
            "META",
            "AMZN",
            "AMD",
            "AVGO",
            "ORCL",
            "SMCI"
          ],
          "sharedCount": 9,
          "description": "AI compute, data centers, networking, and infrastructure demand"
        },
        {
          "label": "Other",
          "sentiment": "neut",
          "sentimentScore": 0,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Search Innovations",
          "sentiment": "pos",
          "sentimentScore": 0.53,
          "riskScore": 0.1,
          "negativeMixed": 0
        },
        {
          "label": "Guidance Outlook",
          "sentiment": "warn",
          "sentimentScore": 0.27,
          "riskScore": 0.27,
          "negativeMixed": 0.33
        },
        {
          "label": "AI Agent Ecosystem",
          "sentiment": "pos",
          "sentimentScore": 0.43,
          "riskScore": 0.07,
          "negativeMixed": 0
        },
        {
          "label": "Youtube Growth",
          "sentiment": "neg",
          "sentimentScore": 0.15,
          "riskScore": 0.3,
          "negativeMixed": 0.5
        },
        {
          "label": "Capital Allocation",
          "sentiment": "neut",
          "sentimentScore": 0.3,
          "riskScore": 0.15,
          "negativeMixed": 0
        },
        {
          "label": "Macro Demand",
          "sentiment": "neut",
          "sentimentScore": 0.3,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "AI In Advertising",
          "sentiment": "pos",
          "sentimentScore": 0.6,
          "riskScore": 0.1,
          "negativeMixed": 0
        },
        {
          "label": "Generative AI Products",
          "sentiment": "pos",
          "description": "AI applications, copilots, automation, and productized model features",
          "sharedWith": [
            "MSFT",
            "META",
            "AMZN",
            "PLTR",
            "CRM",
            "NOW",
            "APP",
            "AI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Cloud Growth",
          "sentiment": "pos",
          "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
          "sharedWith": [
            "MSFT",
            "AMZN",
            "ORCL",
            "NET",
            "DDOG",
            "DOCN",
            "SNOW"
          ],
          "sharedCount": 8
        },
        {
          "label": "Advertising Demand",
          "sentiment": "pos",
          "description": "Ad pricing, conversion, monetization, and marketer demand",
          "sharedWith": [
            "META",
            "AMZN",
            "APP"
          ],
          "sharedCount": 4
        },
        {
          "label": "Data Center Capex",
          "sentiment": "pos",
          "description": "Capex intensity, server buildout, and capacity constraints",
          "sharedWith": [
            "MSFT",
            "NVDA",
            "META",
            "AMZN",
            "AVGO",
            "AMD",
            "SMCI"
          ],
          "sharedCount": 8
        }
      ],
      "audio": {
        "available": true,
        "source": "Q&A audio features",
        "confidence": 58,
        "vocalStress": 44,
        "instability": 53,
        "paceControl": 60,
        "clarity": 49,
        "segmentCount": 6
      },
      "history": [
        {
          "quarter": "Q4_2024",
          "sentiment": 68,
          "risk": 16,
          "uncertainty": 26,
          "negativeMixed": 15,
          "excessReturn5d": -4.31
        },
        {
          "quarter": "Q2_2025",
          "sentiment": 70,
          "risk": 16,
          "uncertainty": 27,
          "negativeMixed": 21,
          "excessReturn5d": -0.14
        },
        {
          "quarter": "Q3_2025",
          "sentiment": 69,
          "risk": 13,
          "uncertainty": 18,
          "negativeMixed": 0,
          "excessReturn5d": 3.46
        },
        {
          "quarter": "Q4_2025",
          "sentiment": 65,
          "risk": 15,
          "uncertainty": 26,
          "negativeMixed": 14,
          "excessReturn5d": -7.32
        }
      ]
    },
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "pos",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "META",
          "AMZN",
          "AMD",
          "AVGO",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Generative AI Products",
        "sentiment": "pos",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Cloud Growth",
        "sentiment": "pos",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "ORCL",
          "NET",
          "DDOG",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8
      },
      {
        "label": "Advertising Demand",
        "sentiment": "pos",
        "description": "Ad pricing, conversion, monetization, and marketer demand",
        "sharedWith": [
          "META",
          "AMZN",
          "APP"
        ],
        "sharedCount": 4
      },
      {
        "label": "Data Center Capex",
        "sentiment": "pos",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "META",
          "AMZN",
          "AVGO",
          "AMD",
          "SMCI"
        ],
        "sharedCount": 8
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "JPM",
          "V",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-04",
        "openReturnPct": -6.25,
        "oneDayReturnPct": -53.75,
        "oneWeekReturnPct": -7.22,
        "oneMonthReturnPct": -8.01
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-29",
        "openReturnPct": 6.2,
        "oneDayReturnPct": 2.52,
        "oneWeekReturnPct": 3.71,
        "oneMonthReturnPct": 14.68
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-23",
        "openReturnPct": 3.57,
        "oneDayReturnPct": 1.02,
        "oneWeekReturnPct": 87.79,
        "oneMonthReturnPct": 8.34
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-25",
        "openReturnPct": 28.71,
        "oneDayReturnPct": -83.35,
        "oneWeekReturnPct": 1.39,
        "oneMonthReturnPct": 6.42
      }
    ],
    "topicDetails": [
      {
        "label": "AI Infrastructure",
        "mentions": 5,
        "sentimentScore": 0.3,
        "riskScore": 0.28,
        "uncertaintyScore": 0.34,
        "qualityScore": 64,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.8529630309847686,
        "riskCorrelation5d": -0.9755595308080114,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.46354997812066795,
            "riskCorrelation": -0.8191850543059231,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.4822932760260421,
            "riskCorrelation": -0.9454863050794091,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.8529630309847686,
            "riskCorrelation": -0.9755595308080114,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.7413552895164621,
            "riskCorrelation": -0.9976686371773927,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.6454394995624566,
            "riskCorrelation": -0.9698809954071741,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.8501496281472792,
            "riskCorrelation": -0.9788511262749585,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Gemini Models",
        "mentions": 5,
        "sentimentScore": 0.36,
        "riskScore": 0.14,
        "uncertaintyScore": 0.3,
        "qualityScore": 69,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.31234052074325647,
        "riskCorrelation5d": 0.6824422018853291,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.3549494526670395,
            "riskCorrelation": -0.0024972257305178893,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.22414952095589796,
            "riskCorrelation": 0.3875982952244693,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.31234052074325647,
            "riskCorrelation": 0.6824422018853291,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": -0.13843853703223521,
            "riskCorrelation": 0.6252218839261323,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.11216911459763337,
            "riskCorrelation": 0.6986077848834911,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.29579272901007125,
            "riskCorrelation": 0.6657716098124882,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Other",
        "mentions": 4,
        "sentimentScore": 0.0,
        "riskScore": 0.0,
        "uncertaintyScore": 0.0,
        "qualityScore": 60,
        "sentiment": "neut",
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": -0.3150952920795814,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": null,
            "riskCorrelation": 0.2829889364982312,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": null,
            "riskCorrelation": -0.19324637205907255,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": null,
            "riskCorrelation": -0.3150952920795814,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": null,
            "riskCorrelation": -0.3186232968446863,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": null,
            "riskCorrelation": -0.472336429756893,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": null,
            "riskCorrelation": -0.29636096029058373,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Guidance / Outlook",
        "mentions": 3,
        "sentimentScore": 0.267,
        "riskScore": 0.267,
        "uncertaintyScore": 0.367,
        "qualityScore": 59,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.7402847268569525,
        "riskCorrelation5d": 0.7402847268569526,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.9861571455907523,
            "riskCorrelation": 0.9861571455907523,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.8378973595370096,
            "riskCorrelation": 0.8378973595370097,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.7402847268569525,
            "riskCorrelation": 0.7402847268569526,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.7796916177487956,
            "riskCorrelation": 0.7796916177487956,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.6732509515005651,
            "riskCorrelation": 0.6732509515005651,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.75542321101772,
            "riskCorrelation": 0.7554232110177199,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Search Innovations",
        "mentions": 3,
        "sentimentScore": 0.533,
        "riskScore": 0.1,
        "uncertaintyScore": 0.233,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.6371134843304557,
        "riskCorrelation5d": 0.6367514957602327,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9913414187523346,
            "riskCorrelation": 0.34904390687174963,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.816126466330199,
            "riskCorrelation": 0.7422337459457592,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.6371134843304557,
            "riskCorrelation": 0.6367514957602327,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": -0.7010359792675094,
            "riskCorrelation": 0.7220203470835086,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.6000403075573622,
            "riskCorrelation": 0.8408148550156579,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.6545241085305091,
            "riskCorrelation": 0.6302679238928561,
            "nEvents": 4
          }
        }
      },
      {
        "label": "AI Agent Ecosystem",
        "mentions": 3,
        "sentimentScore": 0.433,
        "riskScore": 0.067,
        "uncertaintyScore": 0.233,
        "qualityScore": 69,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.0587845914224423,
        "riskCorrelation5d": 0.4388930569708996,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.42666635144379667,
            "riskCorrelation": 0.8077195176287811,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.0391961814667507,
            "riskCorrelation": 0.4641069556128339,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.0587845914224423,
            "riskCorrelation": 0.4388930569708996,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.09218723935765231,
            "riskCorrelation": 0.4291222928194765,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.2719010398909832,
            "riskCorrelation": 0.2604753054037003,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.04083402922484599,
            "riskCorrelation": 0.45685484825201456,
            "nEvents": 4
          }
        }
      },
      {
        "label": "YouTube Growth",
        "mentions": 2,
        "sentimentScore": 0.15,
        "riskScore": 0.3,
        "uncertaintyScore": 0.35,
        "qualityScore": 53,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.68698655685923,
        "riskCorrelation5d": -0.53654105118676,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.7876915902289728,
            "riskCorrelation": -0.803491659876349,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.5541563236608934,
            "riskCorrelation": -0.49364844671735014,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.68698655685923,
            "riskCorrelation": -0.53654105118676,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.6355509871772668,
            "riskCorrelation": -0.5074577802738993,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.4757311809974545,
            "riskCorrelation": -0.337865600092247,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.6987492355048665,
            "riskCorrelation": -0.5522733338403277,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Capital Allocation",
        "mentions": 2,
        "sentimentScore": 0.3,
        "riskScore": 0.15,
        "uncertaintyScore": 0.2,
        "qualityScore": 61,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.5817510919750826,
        "riskCorrelation5d": 0.9784860516294691,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.8968547040087766,
            "riskCorrelation": 0.959964373730085,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": -0.5836054337115334,
            "riskCorrelation": 0.9789542481023088,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.5817510919750826,
            "riskCorrelation": 0.9784860516294691,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.5406805299685971,
            "riskCorrelation": 0.9670450825761941,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.33674752279594977,
            "riskCorrelation": 0.8845701389330229,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.6042355990115034,
            "riskCorrelation": 0.9838653279320536,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Macro / Demand",
        "mentions": 1,
        "sentimentScore": 0.3,
        "riskScore": 0.0,
        "uncertaintyScore": 0.4,
        "qualityScore": 63,
        "sentiment": "neut",
        "sentimentCorrelation5d": 1.0,
        "riskCorrelation5d": null,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          }
        }
      },
      {
        "label": "AI in Advertising",
        "mentions": 1,
        "sentimentScore": 0.6,
        "riskScore": 0.1,
        "uncertaintyScore": 0.2,
        "qualityScore": 68,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.060942011195419706,
        "riskCorrelation5d": -0.07208773638836714,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.7069499962179541,
            "riskCorrelation": 0.608916898171252,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.31440213008705425,
            "riskCorrelation": 0.19488557080953367,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.060942011195419706,
            "riskCorrelation": -0.07208773638836714,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": -0.12012292350754222,
            "riskCorrelation": -0.01291042416268752,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.01576735818680819,
            "riskCorrelation": -0.14575908365309606,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.08343219352194596,
            "riskCorrelation": -0.049629894123560225,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Google Cloud Expansion",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.9991024954246946,
        "riskCorrelation5d": 0.9716546351497188,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.22479766406481902,
            "riskCorrelation": 0.05427284176214372,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.5972412695090965,
            "riskCorrelation": 0.7962627242763757,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.9991024954246946,
            "riskCorrelation": 0.9716546351497188,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": 0.9482718323141637,
            "riskCorrelation": 0.9991175294371382,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": 0.881912999022181,
            "riskCorrelation": 0.9780608219772118,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": 0.9990164786084627,
            "riskCorrelation": 0.9721218140833114,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Margins / Profitability",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.98910409138435,
        "riskCorrelation5d": 0.8440691215293513,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.036601599881015155,
            "riskCorrelation": -0.6818832202938773,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": -0.7380546856299876,
            "riskCorrelation": 0.1161952941904793,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.98910409138435,
            "riskCorrelation": 0.8440691215293513,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.9911786594645341,
            "riskCorrelation": 0.6624977481315972,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.9551099049232917,
            "riskCorrelation": 0.5280529611621414,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.989394264556984,
            "riskCorrelation": 0.84300342448538,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Waymo Developments",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      },
      {
        "label": "Subscription Services",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      }
    ],
    "sentimentHorizon": [
      {
        "horizonDays": 1,
        "rho": 0.96,
        "nEvents": 4,
        "indicativeOnly": true
      },
      {
        "horizonDays": 3,
        "rho": 0.976,
        "nEvents": 4,
        "indicativeOnly": true
      },
      {
        "horizonDays": 5,
        "rho": 0.847,
        "nEvents": 4,
        "indicativeOnly": true
      },
      {
        "horizonDays": 7,
        "rho": 0.914,
        "nEvents": 4,
        "indicativeOnly": true
      },
      {
        "horizonDays": 10,
        "rho": 0.868,
        "nEvents": 4,
        "indicativeOnly": true
      },
      {
        "horizonDays": 21,
        "rho": 0.857,
        "nEvents": 4,
        "indicativeOnly": true
      }
    ],
    "transcript": {
      "sourceFile": "GOOG_2025_10_30_earnings_call_qa_features.json",
      "date": "2025-10-30",
      "exchanges": [
        {
          "exchangeIdx": 0,
          "speaker": "SPEAKER_01",
          "startSec": 159.5,
          "endSec": 203.527,
          "wordCount": 102,
          "text": "And Brian, on Waymo, a great question was reflecting, I think, on the exact same topic I'm scheduled to meet with the team to do a review on it in a few weeks out. Look, it is an exciting time. Waymo clearly is scaling up, particularly in 2026. And I think the possibility, as you said, of German Eye, particularly with the multimodal experience, as well as services like YouTube, I think there's a real opportunity to make the in-car experience dramatically better. Definitely something we are excited about, and you'll see newer experiences in 2026 for sure. Great. Thank you both."
        },
        {
          "exchangeIdx": 1,
          "speaker": "SPEAKER_01",
          "startSec": 507.817,
          "endSec": 579.991,
          "wordCount": 160,
          "text": "So overall, I would say we are seeing substantial demand for our AI infrastructure products, including TPU-based and GPU-based solutions. It is one of the key drivers of our growth over the past year. And I think on a going forward basis, I think we continue to see a very strong demand. And we are investing to meet that. I do think a big part of what differentiates Google Cloud, effectively, we have taken a deep, full stack approach to AI. And that really plays out. We are the only hyperscaler. who's really building offerings on our own models. And we are also highly differentiated on our own technology. So to your question, I think that does give us the opportunity to continue driving growth and operating margins in cloud as we have done in the past. And also I think from a revenue sets the infrastructure portion of our business to be a growth driver looking ahead as well."
        },
        {
          "exchangeIdx": 2,
          "speaker": "SPEAKER_01",
          "startSec": 711.582,
          "endSec": 774.728,
          "wordCount": 173,
          "text": "Mark, look, I think, obviously, AI overviews are a natural part of the Google experience. And so, you know, engagement is very, very high. I would say AI mode, you have a varied cohorts that are people who are casual users who are checking it out. But there's a core group which really likes AI more and is passionate about it. And so you see the early adopters, the product is resonating very strongly and they are seeking it out. So I think that's how I would highlight the difference. With Gemini, again, a set of engaged user base who are seeking out the product and so on. But across the board, I think the trajectory has been we are definitely seeing in each of those use cases a set of early adopters and then more people coming in and the people who are using it continue to use it more over time and report high user satisfaction. So I would say the underlying product metrics are pretty encouraging to see as well."
        },
        {
          "exchangeIdx": 3,
          "speaker": "SPEAKER_01",
          "startSec": 1088.08,
          "endSec": 1116.278,
          "wordCount": 75,
          "text": "Yeah, and the only thing I would add is just stepping back broadly, I think, AI overviews and AI mode are, you know, dramatically improving search. We can see it in user satisfaction, user quality, all our metrics. And they're universal in the nature. They apply across the universality of human needs. So I think we are seeing it in breadth. And so, you know, naturally over time, that'll apply to commercial categories as well."
        },
        {
          "exchangeIdx": 4,
          "speaker": "SPEAKER_01",
          "startSec": 1191.355,
          "endSec": 1294.765,
          "wordCount": 264,
          "text": "Ken, thanks. Look, I think it's a dynamic moment. And I think we are meeting people in the moment with what they are trying to do. Obviously, search is evolving. You know between AI overviews and AI mode. I think we are if we are able to kind of give that range of experience For people in this moment over time you will expect us to you can expect us To make the experience is simpler in a way that just like we did universal search many, many years ago. We may have done tech search, image search, video search, et cetera. And then we kind of brought it together as universal search. So you will see evolutions like that. But I think we want to be sensitive to making sure we are meeting the users in terms of what they are looking for. I think Gemini allows us to build a more personal, proactive, powerful AI assistant for that moment. And I think having the two-surfaces search in Gemini allows us to really serve users across the breadth of their needs. But over time, we will thoughtfully look for opportunities to make the experience better for users. And to the first part, I would broadly say as As I do think we've been consistently saying for a while now, this is an expansionary moment and we are seeing people engage more. And I think when they do that, naturally a portion of that information for users, those journeys are commercial in nature. So we expect that to play out over time as well."
        },
        {
          "exchangeIdx": 5,
          "speaker": "SPEAKER_01",
          "startSec": 1333.426,
          "endSec": 1432.567,
          "wordCount": 229,
          "text": "Thanks, Justin. The first on the pace of frontier model research and development. Look, I think two things are both simultaneously true. I'm incredibly impressed by the pace at which the teams are executing and the pace at which we are improving these models. But it also is true at the same time that each of the prior model you're trying to get better over is now getting more and more capable. So I think both the pace is increasing, but sometimes we are taking the time to put out a notably improved model. So I think that may take slightly longer. But I do think the underlying pace is phenomenal to see. And I'm excited about our Gemini 3.0 release later this year. On cloud, I would point out as a sign of the momentum, I think the number of deals greater than $1 billion that we signed in the first three quarters of this year are greater than the two years prior. So we are definitely seeing strong momentum and we are executing at pace. And in terms of long-term economics, I would say that, again, us being a full-stack AI player and the fact that we are developing highly differentiated products on our own technology, I think will help us drive a good trajectory here, as you've seen over the past few years. Great, thank you."
        }
      ]
    }
  },
  "DOCN": {
    "symbol": "DOCN",
    "name": "DigitalOcean Holdings, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-05",
    "bias": "BULLISH",
    "finalScore": 47.44,
    "overallScore": 47.44,
    "confidence": 81,
    "probBull": 44,
    "probBear": 56,
    "mlScore": -19.51,
    "decisionInputs": {
      "epsSurprisePct": 83.33,
      "revenueSurprisePct": 3.26,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 6.4,
      "netMarginPct": 6.12,
      "fcfMarginPct": 2.69,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.22,
      "epsEstimate": 0.12,
      "revenueActual": "257.9M",
      "revenueEstimate": "249.8M",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "6.9M",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Cloud Growth",
        "sentiment": "pos",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "NET",
          "DDOG",
          "SNOW"
        ],
        "sharedCount": 8
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-24",
        "openReturnPct": 23.91,
        "oneDayReturnPct": -5.53,
        "oneWeekReturnPct": -16.7,
        "oneMonthReturnPct": 36.47
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-05",
        "openReturnPct": 4.84,
        "oneDayReturnPct": 2.77,
        "oneWeekReturnPct": -1.35,
        "oneMonthReturnPct": 8.03
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-05",
        "openReturnPct": 2.18,
        "oneDayReturnPct": 4.77,
        "oneWeekReturnPct": -8.79,
        "oneMonthReturnPct": -5.89
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-06",
        "openReturnPct": 21.24,
        "oneDayReturnPct": 1.31,
        "oneWeekReturnPct": 10.51,
        "oneMonthReturnPct": 1.35
      }
    ],
    "topicDetails": [
      {
        "label": "Cloud Growth",
        "sentiment": "pos",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "NET",
          "DDOG",
          "SNOW"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 78,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 78,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "V": {
    "symbol": "V",
    "name": "VISA INC.",
    "sector": "financials",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-28",
    "bias": "BULLISH",
    "finalScore": 44.68,
    "overallScore": 44.68,
    "confidence": 80,
    "probBull": 49,
    "probBear": 51,
    "mlScore": -9.63,
    "decisionInputs": {
      "epsSurprisePct": 7.12,
      "revenueSurprisePct": 4.47,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 3.02,
      "netMarginPct": 53.62,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 3.31,
      "epsEstimate": 3.09,
      "revenueActual": "11.2B",
      "revenueEstimate": "10.8B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "pos",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "JPM",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-29",
        "openReturnPct": 12.66,
        "oneDayReturnPct": -3.0,
        "oneWeekReturnPct": -6.63,
        "oneMonthReturnPct": -3.31
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-28",
        "openReturnPct": -25.94,
        "oneDayReturnPct": -1.62,
        "oneWeekReturnPct": -1.98,
        "oneMonthReturnPct": -3.59
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-29",
        "openReturnPct": 11.67,
        "oneDayReturnPct": -10.82,
        "oneWeekReturnPct": -3.29,
        "oneMonthReturnPct": -40.71
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-29",
        "openReturnPct": -2.07,
        "oneDayReturnPct": 1.17,
        "oneWeekReturnPct": 2.44,
        "oneMonthReturnPct": 6.93
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 77,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "pos",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 77,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "JPM",
          "UNH"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 77,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "JPM": {
    "symbol": "JPM",
    "name": "JPMORGAN CHASE & CO",
    "sector": "financials",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-14",
    "bias": "BULLISH",
    "finalScore": 41.75,
    "overallScore": 41.75,
    "confidence": 79,
    "probBull": 48,
    "probBear": 52,
    "mlScore": -11.46,
    "decisionInputs": {
      "epsSurprisePct": 8.2,
      "revenueSurprisePct": 3.18,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 8.82,
      "netMarginPct": 33.1,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 5.94,
      "epsEstimate": 5.49,
      "revenueActual": "49.8B",
      "revenueEstimate": "48.3B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "pos",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "V",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-13",
        "openReturnPct": -86.84,
        "oneDayReturnPct": -97.46,
        "oneWeekReturnPct": -2.34,
        "oneMonthReturnPct": -2.69
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-14",
        "openReturnPct": 1.43,
        "oneDayReturnPct": 1.2,
        "oneWeekReturnPct": -2.64,
        "oneMonthReturnPct": 2.45
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-15",
        "openReturnPct": 64.56,
        "oneDayReturnPct": -25.48,
        "oneWeekReturnPct": 3.56,
        "oneMonthReturnPct": 2.66
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-11",
        "openReturnPct": 38.1,
        "oneDayReturnPct": -62.66,
        "oneWeekReturnPct": -25.83,
        "oneMonthReturnPct": 12.46
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "pos",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "V",
          "UNH"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "CRWD": {
    "symbol": "CRWD",
    "name": "CrowdStrike Holdings, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-03-03",
    "bias": "BULLISH",
    "finalScore": 40.99,
    "overallScore": 40.99,
    "confidence": 79,
    "probBull": 50,
    "probBear": 50,
    "mlScore": -6.61,
    "decisionInputs": {
      "epsSurprisePct": 15,
      "revenueSurprisePct": 0.41,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 5.76,
      "netMarginPct": 4.55,
      "fcfMarginPct": 30.29,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.23,
      "epsEstimate": 0.2,
      "revenueActual": "1.3B",
      "revenueEstimate": "1.3B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "395.4M",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Cybersecurity Demand",
        "sentiment": "pos",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "PANW",
          "ZS",
          "NET"
        ],
        "sharedCount": 4
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "ZS"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-02",
        "openReturnPct": -3.59,
        "oneDayReturnPct": 1.48,
        "oneWeekReturnPct": 57.88,
        "oneMonthReturnPct": -11.62
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-08-27",
        "openReturnPct": -2.98,
        "oneDayReturnPct": 4.59,
        "oneWeekReturnPct": -1.18,
        "oneMonthReturnPct": 15.58
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-06-04",
        "openReturnPct": 38.32,
        "oneDayReturnPct": 51.68,
        "oneWeekReturnPct": 4.6,
        "oneMonthReturnPct": 10.24
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-06-03",
        "openReturnPct": -6.96,
        "oneDayReturnPct": -5.77,
        "oneWeekReturnPct": -2.41,
        "oneMonthReturnPct": 3.42
      }
    ],
    "topicDetails": [
      {
        "label": "Cybersecurity Demand",
        "sentiment": "pos",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "PANW",
          "ZS",
          "NET"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "ZS"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "NET": {
    "symbol": "NET",
    "name": "Cloudflare, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-07",
    "bias": "BULLISH",
    "finalScore": 40.41,
    "overallScore": 40.41,
    "confidence": 79,
    "probBull": 52,
    "probBear": 48,
    "mlScore": -4.45,
    "decisionInputs": {
      "epsSurprisePct": 16.67,
      "revenueSurprisePct": 3.05,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 4.11,
      "netMarginPct": -3.58,
      "fcfMarginPct": 14.55,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 5
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": -0.05,
      "epsEstimate": -0.06,
      "revenueActual": "639.8M",
      "revenueEstimate": "620.8M",
      "guidanceRevenueMid": "664.5M",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "93.1M",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Cloud Growth",
        "sentiment": "pos",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "DDOG",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8
      },
      {
        "label": "Cybersecurity Demand",
        "sentiment": "pos",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 4
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-10",
        "openReturnPct": 12.82,
        "oneDayReturnPct": 5.24,
        "oneWeekReturnPct": 7.03,
        "oneMonthReturnPct": 18.04
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-30",
        "openReturnPct": 6.29,
        "oneDayReturnPct": 13.84,
        "oneWeekReturnPct": 4.63,
        "oneMonthReturnPct": -9.55
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-31",
        "openReturnPct": -4.34,
        "oneDayReturnPct": -3.65,
        "oneWeekReturnPct": -1.48,
        "oneMonthReturnPct": 17.82
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-08",
        "openReturnPct": 7.49,
        "oneDayReturnPct": 6.46,
        "oneWeekReturnPct": 26.45,
        "oneMonthReturnPct": 44.27
      }
    ],
    "topicDetails": [
      {
        "label": "Cloud Growth",
        "sentiment": "pos",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "DDOG",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Cybersecurity Demand",
        "sentiment": "pos",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 75,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "APP": {
    "symbol": "APP",
    "name": "AppLovin Corp",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-06",
    "bias": "BULLISH",
    "finalScore": 39.59,
    "overallScore": 39.59,
    "confidence": 78,
    "probBull": 48,
    "probBear": 52,
    "mlScore": -12.16,
    "decisionInputs": {
      "epsSurprisePct": 4.71,
      "revenueSurprisePct": 4.09,
      "guidanceRevenueSurprisePct": 0,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 11.13,
      "netMarginPct": 65.44,
      "fcfMarginPct": null,
      "quartersWithConsensus": 4,
      "quartersWithGuidance": 3
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.67,
      "epsEstimate": null,
      "revenueActual": "1.2B",
      "revenueEstimate": "n/a",
      "guidanceRevenueMid": "1.9B",
      "guidanceRevenueConsensus": "1.9B",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Generative AI Products",
        "sentiment": "warn",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "NOW",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Advertising Demand",
        "sentiment": "warn",
        "description": "Ad pricing, conversion, monetization, and marketer demand",
        "sharedWith": [
          "GOOGL",
          "META",
          "AMZN"
        ],
        "sharedCount": 4
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Travel & Marketplace",
        "sentiment": "warn",
        "description": "Marketplace liquidity, take rate, bookings, and platform engagement",
        "sharedWith": [
          "ABNB",
          "XYZ"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q1",
        "reportDate": "",
        "openReturnPct": null,
        "oneDayReturnPct": null,
        "oneWeekReturnPct": null,
        "oneMonthReturnPct": null
      },
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-11",
        "openReturnPct": -11.56,
        "oneDayReturnPct": -19.68,
        "oneWeekReturnPct": -8.35,
        "oneMonthReturnPct": -76.84
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-05",
        "openReturnPct": 5.58,
        "oneDayReturnPct": 69.85,
        "oneWeekReturnPct": -9.87,
        "oneMonthReturnPct": 11.78
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-06",
        "openReturnPct": 1.71,
        "oneDayReturnPct": 11.97,
        "oneWeekReturnPct": 10.95,
        "oneMonthReturnPct": 40.06
      }
    ],
    "topicDetails": [
      {
        "label": "Generative AI Products",
        "sentiment": "warn",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "NOW",
          "AI"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 74,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Advertising Demand",
        "sentiment": "warn",
        "description": "Ad pricing, conversion, monetization, and marketer demand",
        "sharedWith": [
          "GOOGL",
          "META",
          "AMZN"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 74,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 74,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Travel & Marketplace",
        "sentiment": "warn",
        "description": "Marketplace liquidity, take rate, bookings, and platform engagement",
        "sharedWith": [
          "ABNB",
          "XYZ"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 74,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "ORCL": {
    "symbol": "ORCL",
    "name": "ORACLE CORP",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-03-10",
    "bias": "BULLISH",
    "finalScore": 32.03,
    "overallScore": 32.03,
    "confidence": 75,
    "probBull": 45,
    "probBear": 55,
    "mlScore": -17.36,
    "decisionInputs": {
      "epsSurprisePct": 6.72,
      "revenueSurprisePct": 1.66,
      "guidanceRevenueSurprisePct": -0.26,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 7.05,
      "netMarginPct": 21.65,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.43,
      "epsEstimate": 1.34,
      "revenueActual": "17.2B",
      "revenueEstimate": "16.9B",
      "guidanceRevenueMid": "19.1B",
      "guidanceRevenueConsensus": "19.1B",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "AVGO",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Cloud Growth",
        "sentiment": "warn",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "NET",
          "DDOG",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-10",
        "openReturnPct": -14.52,
        "oneDayReturnPct": -10.83,
        "oneWeekReturnPct": -19.27,
        "oneMonthReturnPct": -9.29
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-09",
        "openReturnPct": 32.16,
        "oneDayReturnPct": 35.95,
        "oneWeekReturnPct": 24.8,
        "oneMonthReturnPct": 22.96
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-06-11",
        "openReturnPct": 7.7,
        "oneDayReturnPct": 13.31,
        "oneWeekReturnPct": 16.32,
        "oneMonthReturnPct": 33.21
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-03-10",
        "openReturnPct": null,
        "oneDayReturnPct": null,
        "oneWeekReturnPct": null,
        "oneMonthReturnPct": null
      }
    ],
    "topicDetails": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "AVGO",
          "SMCI"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Cloud Growth",
        "sentiment": "warn",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "NET",
          "DDOG",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "AMD": {
    "symbol": "AMD",
    "name": "ADVANCED MICRO DEVICES INC",
    "sector": "semis-hardware",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-05",
    "bias": "BULLISH",
    "finalScore": 32.01,
    "overallScore": 32.01,
    "confidence": 75,
    "probBull": 49,
    "probBear": 51,
    "mlScore": -9.94,
    "decisionInputs": {
      "epsSurprisePct": 4.72,
      "revenueSurprisePct": 3.57,
      "guidanceRevenueSurprisePct": 6.67,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -0.17,
      "netMarginPct": 13.49,
      "fcfMarginPct": 25.03,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.11,
      "epsEstimate": 1.06,
      "revenueActual": "10.3B",
      "revenueEstimate": "9.9B",
      "guidanceRevenueMid": "11.2B",
      "guidanceRevenueConsensus": "10.5B",
      "freeCashFlow": "2.6B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "warn",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AVGO",
          "QCOM",
          "AMAT",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "SMCI"
        ],
        "sharedCount": 8
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-03",
        "openReturnPct": -11.2,
        "oneDayReturnPct": -17.31,
        "oneWeekReturnPct": -11.78,
        "oneMonthReturnPct": -20.52
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-04",
        "openReturnPct": -2.72,
        "oneDayReturnPct": 2.51,
        "oneWeekReturnPct": 3.54,
        "oneMonthReturnPct": -12.83
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-05",
        "openReturnPct": -5.31,
        "oneDayReturnPct": -6.42,
        "oneWeekReturnPct": 5.8,
        "oneMonthReturnPct": -13.29
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-06",
        "openReturnPct": 2.16,
        "oneDayReturnPct": 1.76,
        "oneWeekReturnPct": 19.37,
        "oneMonthReturnPct": 17.82
      }
    ],
    "topicDetails": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "warn",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AVGO",
          "QCOM",
          "AMAT",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "SMCI"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "XYZ": {
    "symbol": "XYZ",
    "name": "Block, Inc.",
    "sector": "financials",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-07",
    "bias": "BULLISH",
    "finalScore": 31.34,
    "overallScore": 31.34,
    "confidence": 75,
    "probBull": 39,
    "probBear": 61,
    "mlScore": -29.98,
    "decisionInputs": {
      "epsSurprisePct": 136.67,
      "revenueSurprisePct": null,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -3.13,
      "netMarginPct": -5.1,
      "fcfMarginPct": 15.44,
      "quartersWithConsensus": 4,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": null,
      "epsEstimate": null,
      "revenueActual": "n/a",
      "revenueEstimate": "n/a",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": ""
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "pos",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "WMT",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8
      },
      {
        "label": "Travel & Marketplace",
        "sentiment": "pos",
        "description": "Marketplace liquidity, take rate, bookings, and platform engagement",
        "sharedWith": [
          "ABNB",
          "APP"
        ],
        "sharedCount": 3
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "pos",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "WMT",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Travel & Marketplace",
        "sentiment": "pos",
        "description": "Marketplace liquidity, take rate, bookings, and platform engagement",
        "sharedWith": [
          "ABNB",
          "APP"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 70,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "META": {
    "symbol": "META",
    "name": "Meta Platforms, Inc.",
    "sector": "call-analysis",
    "hasCallAnalysis": true,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-29",
    "bias": "BULLISH",
    "finalScore": 29.38,
    "overallScore": 29.38,
    "confidence": 74,
    "probBull": 46,
    "probBear": 54,
    "mlScore": -15.97,
    "decisionInputs": {
      "epsSurprisePct": 8.94,
      "revenueSurprisePct": 1.35,
      "guidanceRevenueSurprisePct": -0.17,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -5.98,
      "netMarginPct": 47.54,
      "fcfMarginPct": 23.49,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 4
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 7.31,
      "epsEstimate": 6.71,
      "revenueActual": "56.3B",
      "revenueEstimate": "55.6B",
      "guidanceRevenueMid": "59.5B",
      "guidanceRevenueConsensus": "59.6B",
      "freeCashFlow": "13.2B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "callAnalysis": {
      "period": "Q1_2026",
      "callDate": "2026-04-23",
      "turnCount": 33,
      "overall": {
        "sentiment": 58,
        "confidence": 72,
        "risk": 23,
        "uncertainty": 37,
        "defensiveness": 23,
        "analystPressure": 24,
        "guidanceStrength": 34,
        "negativeMixed": 12
      },
      "prepared": {
        "positiveLang": 47,
        "negativeLang": 48,
        "riskLanguage": 15,
        "uncertainty": 29,
        "analystPressure": 14,
        "defensiveLang": 15,
        "guidanceStrength": 34
      },
      "qa": {
        "positiveLang": 34,
        "negativeLang": 14,
        "riskLanguage": 35,
        "uncertainty": 47,
        "analystPressure": 42,
        "defensiveLang": 35,
        "guidanceStrength": 24
      },
      "topics": [
        {
          "label": "Macro Demand",
          "sentiment": "warn",
          "sentimentScore": 0.12,
          "riskScore": 0.28,
          "negativeMixed": 0.08
        },
        {
          "label": "Guidance Outlook",
          "sentiment": "neut",
          "sentimentScore": 0.17,
          "riskScore": 0.22,
          "negativeMixed": 0.17
        },
        {
          "label": "Margins Profitability",
          "sentiment": "warn",
          "sentimentScore": 0.03,
          "riskScore": 0.3,
          "negativeMixed": 0.33
        },
        {
          "label": "Regulatory And Compliance Efforts",
          "sentiment": "neut",
          "sentimentScore": 0,
          "riskScore": 0.2,
          "negativeMixed": 0
        },
        {
          "label": "Other",
          "sentiment": "neut",
          "sentimentScore": 0,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Infrastructure Investments",
          "sentiment": "warn",
          "sentimentScore": 0.3,
          "riskScore": 0.25,
          "negativeMixed": 0
        },
        {
          "label": "Advertising Performance",
          "sentiment": "neg",
          "sentimentScore": 0.2,
          "riskScore": 0.4,
          "negativeMixed": 0.5
        },
        {
          "label": "Market Positioning",
          "sentiment": "pos",
          "sentimentScore": 0.6,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Consumer Electronics Trends",
          "sentiment": "pos",
          "sentimentScore": 0.6,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Capital Allocation",
          "sentiment": "pos",
          "sentimentScore": 0.6,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "AI Infrastructure",
          "sentiment": "warn",
          "description": "AI compute, data centers, networking, and infrastructure demand",
          "sharedWith": [
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "AMD",
            "AVGO",
            "ORCL",
            "SMCI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Generative AI Products",
          "sentiment": "warn",
          "description": "AI applications, copilots, automation, and productized model features",
          "sharedWith": [
            "MSFT",
            "GOOGL",
            "AMZN",
            "PLTR",
            "CRM",
            "NOW",
            "APP",
            "AI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Advertising Demand",
          "sentiment": "warn",
          "description": "Ad pricing, conversion, monetization, and marketer demand",
          "sharedWith": [
            "GOOGL",
            "AMZN",
            "APP"
          ],
          "sharedCount": 4
        },
        {
          "label": "Data Center Capex",
          "sentiment": "warn",
          "description": "Capex intensity, server buildout, and capacity constraints",
          "sharedWith": [
            "MSFT",
            "NVDA",
            "GOOGL",
            "AMZN",
            "AVGO",
            "AMD",
            "SMCI"
          ],
          "sharedCount": 8
        }
      ],
      "audio": {
        "available": false,
        "source": "No extracted audio feature file in this snapshot",
        "confidence": null,
        "vocalStress": null,
        "instability": null,
        "paceControl": null,
        "clarity": null,
        "segmentCount": 0
      },
      "history": [
        {
          "quarter": "Q3_2024",
          "sentiment": 67,
          "risk": 22,
          "uncertainty": 34,
          "negativeMixed": 4,
          "excessReturn5d": -1.89
        },
        {
          "quarter": "Q4_2024",
          "sentiment": 66,
          "risk": 24,
          "uncertainty": 34,
          "negativeMixed": 16,
          "excessReturn5d": 2.39
        },
        {
          "quarter": "Q3_2025",
          "sentiment": 59,
          "risk": 17,
          "uncertainty": 27,
          "negativeMixed": 11,
          "excessReturn5d": -4.83
        },
        {
          "quarter": "Q4_2025",
          "sentiment": 64,
          "risk": 16,
          "uncertainty": 22,
          "negativeMixed": 11,
          "excessReturn5d": -4.08
        },
        {
          "quarter": "Q1_2026",
          "sentiment": 58,
          "risk": 23,
          "uncertainty": 37,
          "negativeMixed": 12,
          "excessReturn5d": -9.67
        }
      ]
    },
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "AMZN",
          "AMD",
          "AVGO",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Generative AI Products",
        "sentiment": "warn",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "AMZN",
          "PLTR",
          "CRM",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Advertising Demand",
        "sentiment": "warn",
        "description": "Ad pricing, conversion, monetization, and marketer demand",
        "sharedWith": [
          "GOOGL",
          "AMZN",
          "APP"
        ],
        "sharedCount": 4
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "AMZN",
          "AVGO",
          "AMD",
          "SMCI"
        ],
        "sharedCount": 8
      },
      {
        "label": "Margins & Cost Control",
        "sentiment": "warn",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "warn",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "JPM",
          "V",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-28",
        "openReturnPct": 10.27,
        "oneDayReturnPct": 10.4,
        "oneWeekReturnPct": 22.13,
        "oneMonthReturnPct": -2.27
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-29",
        "openReturnPct": -10.98,
        "oneDayReturnPct": -11.33,
        "oneWeekReturnPct": -17.66,
        "oneMonthReturnPct": -14.74
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-30",
        "openReturnPct": 11.51,
        "oneDayReturnPct": 11.25,
        "oneWeekReturnPct": 9.58,
        "oneMonthReturnPct": 6.26
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-30",
        "openReturnPct": 7.85,
        "oneDayReturnPct": 4.23,
        "oneWeekReturnPct": 8.93,
        "oneMonthReturnPct": 22.2
      }
    ],
    "topicDetails": [
      {
        "label": "Macro / Demand",
        "mentions": 13,
        "sentimentScore": 0.115,
        "riskScore": 0.285,
        "uncertaintyScore": 0.446,
        "qualityScore": 69,
        "sentiment": "warn",
        "horizons": {}
      },
      {
        "label": "Guidance / Outlook",
        "mentions": 6,
        "sentimentScore": 0.167,
        "riskScore": 0.217,
        "uncertaintyScore": 0.383,
        "qualityScore": 64,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.28537932222603163,
        "riskCorrelation5d": 0.8397245985422664,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9469645826830576,
            "riskCorrelation": -0.5099239355385923,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.9969432379522333,
            "riskCorrelation": -0.3274963972261632,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.28537932222603163,
            "riskCorrelation": 0.8397245985422664,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": -0.4231193207044744,
            "riskCorrelation": 0.7660063333732653,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.41793557532207426,
            "riskCorrelation": 0.7605115151040416,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.17995860369424005,
            "riskCorrelation": 0.8327570661521906,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Margins / Profitability",
        "mentions": 3,
        "sentimentScore": 0.033,
        "riskScore": 0.3,
        "uncertaintyScore": 0.367,
        "qualityScore": 52,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.8962946064826471,
        "riskCorrelation5d": -0.11334713782485785,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.5354942050659318,
            "riskCorrelation": -0.9999683767555959,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.6825635172296342,
            "riskCorrelation": -0.9841609823596326,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.8962946064826471,
            "riskCorrelation": -0.11334713782485785,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": 0.88345510421241,
            "riskCorrelation": -0.08533164303659703,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": 0.8727997850773248,
            "riskCorrelation": -0.06311686470747163,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": 0.578588717106682,
            "riskCorrelation": 0.37162003620394446,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Regulatory and Compliance Efforts",
        "mentions": 2,
        "sentimentScore": 0.0,
        "riskScore": 0.2,
        "uncertaintyScore": 0.5,
        "qualityScore": 52,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.9999999999999999,
        "riskCorrelation5d": -0.9999999999999999,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 0.9999999999999999,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": 0.9999999999999998,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": 0.9999999999999999,
            "riskCorrelation": -0.9999999999999999,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": 0.9999999999999999,
            "riskCorrelation": -0.9999999999999999,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Advertising Performance",
        "mentions": 2,
        "sentimentScore": 0.2,
        "riskScore": 0.4,
        "uncertaintyScore": 0.4,
        "qualityScore": 52,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.7781630495518193,
        "riskCorrelation5d": -0.4133727443920776,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.43615070434385744,
            "riskCorrelation": 0.666185022043856,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.36034561070856824,
            "riskCorrelation": 0.5915174209613107,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.7781630495518193,
            "riskCorrelation": -0.4133727443920776,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.7397509797316497,
            "riskCorrelation": -0.30810086314642454,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.7314997809925542,
            "riskCorrelation": -0.35522547164861656,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.8142539665326979,
            "riskCorrelation": -0.6931715794800172,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Infrastructure Investments",
        "mentions": 2,
        "sentimentScore": 0.3,
        "riskScore": 0.25,
        "uncertaintyScore": 0.35,
        "qualityScore": 58,
        "sentiment": "warn",
        "sentimentCorrelation5d": -0.1701618015116352,
        "riskCorrelation5d": 0.41780996163769146,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.2957058792264376,
            "riskCorrelation": 0.3039628810658701,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.15074843966245696,
            "riskCorrelation": 0.5612377883288647,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.1701618015116352,
            "riskCorrelation": 0.41780996163769146,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.3267411045687608,
            "riskCorrelation": 0.5790541469145627,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.28050683714020086,
            "riskCorrelation": 0.566008313810303,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.2508898896180255,
            "riskCorrelation": 0.38671334205162505,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Other",
        "mentions": 2,
        "sentimentScore": 0.0,
        "riskScore": 0.0,
        "uncertaintyScore": 0.0,
        "qualityScore": 56,
        "sentiment": "neut",
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Capital Allocation",
        "mentions": 1,
        "sentimentScore": 0.6,
        "riskScore": 0.0,
        "uncertaintyScore": 0.1,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.954231485079429,
        "riskCorrelation5d": 0.9017200924460886,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.19678565982254068,
            "riskCorrelation": -0.3348310824375474,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.011906313144400603,
            "riskCorrelation": -0.1546312106470266,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.954231485079429,
            "riskCorrelation": 0.9017200924460886,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.9622726691517313,
            "riskCorrelation": 0.9135333802050748,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.9680946110650066,
            "riskCorrelation": 0.9223673211291625,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.9818854733205578,
            "riskCorrelation": 0.9988824974049104,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Consumer Electronics Trends",
        "mentions": 1,
        "sentimentScore": 0.6,
        "riskScore": 0.0,
        "uncertaintyScore": 0.3,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.7393276684823665,
        "riskCorrelation5d": 0.7181580085713002,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.5916359564806326,
            "riskCorrelation": -0.6162767596277761,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.4321853581006441,
            "riskCorrelation": -0.45985718709019285,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.7393276684823665,
            "riskCorrelation": 0.7181580085713002,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.7579906686145734,
            "riskCorrelation": 0.7374637833031903,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.772331714341984,
            "riskCorrelation": 0.7523249452532719,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.9702874423753441,
            "riskCorrelation": 0.9623437020310966,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Market Positioning",
        "mentions": 1,
        "sentimentScore": 0.6,
        "riskScore": 0.0,
        "uncertaintyScore": 0.2,
        "qualityScore": 71,
        "sentiment": "pos",
        "horizons": {}
      },
      {
        "label": "User Growth Metrics",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.9992834428325845,
        "riskCorrelation5d": -0.41436933776065527,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9502405455059921,
            "riskCorrelation": -0.162027834704958,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.9848395471448169,
            "riskCorrelation": -0.4664495475351557,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.9992834428325845,
            "riskCorrelation": -0.41436933776065527,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": -0.9551474089833781,
            "riskCorrelation": -0.5497688057799368,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.9575997911279549,
            "riskCorrelation": -0.6008840955128942,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.6542865791733842,
            "riskCorrelation": -0.9492672313629972,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Monetization Strategies",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.4056592534001187,
        "riskCorrelation5d": -0.6704813120428127,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.18408946779564606,
            "riskCorrelation": -0.3838099439654644,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.4771553247903598,
            "riskCorrelation": -0.579892488083133,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.4056592534001187,
            "riskCorrelation": -0.6704813120428127,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.5157344303982104,
            "riskCorrelation": -0.839098965120214,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.5765314510079154,
            "riskCorrelation": -0.8082243965295273,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.9401941756961232,
            "riskCorrelation": -0.7223491544225168,
            "nEvents": 4
          }
        }
      },
      {
        "label": "AI Integration",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.10823743191150663,
        "riskCorrelation5d": 0.19253267369189062,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.257764883235779,
            "riskCorrelation": 0.46295194185514316,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.005793815420033391,
            "riskCorrelation": 0.3297094619834691,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.10823743191150663,
            "riskCorrelation": 0.19253267369189062,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.016247795548041837,
            "riskCorrelation": -0.0451736681004302,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.06149035135000925,
            "riskCorrelation": 0.02184355317835079,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.6154686604659371,
            "riskCorrelation": -0.02544079552168092,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Product Innovations",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      },
      {
        "label": "Content Recommendation Systems",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      },
      {
        "label": "Reality Labs Developments",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      }
    ],
    "sentimentHorizon": [
      {
        "horizonDays": 1,
        "rho": 0.23,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 3,
        "rho": 0.241,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 5,
        "rho": 0.817,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 7,
        "rho": 0.693,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 10,
        "rho": 0.715,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 21,
        "rho": 0.634,
        "nEvents": 5,
        "indicativeOnly": true
      }
    ]
  },
  "KO": {
    "symbol": "KO",
    "name": "COCA COLA CO",
    "sector": "consumer-retail",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-28",
    "bias": "BULLISH",
    "finalScore": 28.36,
    "overallScore": 28.36,
    "confidence": 74,
    "probBull": 52,
    "probBear": 48,
    "mlScore": -3.38,
    "decisionInputs": {
      "epsSurprisePct": 6.17,
      "revenueSurprisePct": 1.9,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 5.5,
      "netMarginPct": 31.46,
      "fcfMarginPct": 14.07,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.86,
      "epsEstimate": 0.81,
      "revenueActual": "12.5B",
      "revenueEstimate": "12.2B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "1.8B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "warn",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "PG"
        ],
        "sharedCount": 8
      },
      {
        "label": "Restaurants & Traffic",
        "sentiment": "warn",
        "description": "Restaurant traffic, pricing, volumes, and consumer staples demand",
        "sharedWith": [
          "CMG",
          "MCD"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-10",
        "openReturnPct": 9.11,
        "oneDayReturnPct": 2.33,
        "oneWeekReturnPct": 2.73,
        "oneMonthReturnPct": 69.0
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-21",
        "openReturnPct": 19.66,
        "oneDayReturnPct": -57.57,
        "oneWeekReturnPct": -4.03,
        "oneMonthReturnPct": -1.4
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-22",
        "openReturnPct": 0.0,
        "oneDayReturnPct": -71.78,
        "oneWeekReturnPct": -1.31,
        "oneMonthReturnPct": 1.44
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-29",
        "openReturnPct": 1.38,
        "oneDayReturnPct": 27.64,
        "oneWeekReturnPct": 6.91,
        "oneMonthReturnPct": -34.55
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 68,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "warn",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "PG"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 68,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Restaurants & Traffic",
        "sentiment": "warn",
        "description": "Restaurant traffic, pricing, volumes, and consumer staples demand",
        "sharedWith": [
          "CMG",
          "MCD"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 68,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "AVGO": {
    "symbol": "AVGO",
    "name": "Broadcom Inc.",
    "sector": "semis-hardware",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-03-04",
    "bias": "BULLISH",
    "finalScore": 26.61,
    "overallScore": 26.61,
    "confidence": 73,
    "probBull": 49,
    "probBear": 51,
    "mlScore": -9.61,
    "decisionInputs": {
      "epsSurprisePct": 5.39,
      "revenueSurprisePct": 1.1,
      "guidanceRevenueSurprisePct": 7.84,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.76,
      "epsEstimate": 1.67,
      "revenueActual": "19.3B",
      "revenueEstimate": "19.1B",
      "guidanceRevenueMid": "22.0B",
      "guidanceRevenueConsensus": "20.4B",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "warn",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "QCOM",
          "AMAT",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "SMCI"
        ],
        "sharedCount": 8
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-11",
        "openReturnPct": -6.5,
        "oneDayReturnPct": -11.43,
        "oneWeekReturnPct": -16.24,
        "oneMonthReturnPct": -16.36
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-04",
        "openReturnPct": 16.23,
        "oneDayReturnPct": 9.41,
        "oneWeekReturnPct": 17.57,
        "oneMonthReturnPct": 9.6
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-06-11",
        "openReturnPct": -1.08,
        "oneDayReturnPct": 1.25,
        "oneWeekReturnPct": -1.15,
        "oneMonthReturnPct": 11.08
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-06-05",
        "openReturnPct": -3.38,
        "oneDayReturnPct": -5.0,
        "oneWeekReturnPct": -4.32,
        "oneMonthReturnPct": 6.91
      }
    ],
    "topicDetails": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 67,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "warn",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "QCOM",
          "AMAT",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 67,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "SMCI"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 67,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 67,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": [],
    "transcript": {
      "sourceFile": "AVGO_2025_12_11_earnings_call_qa_features.json",
      "date": "2025-12-11",
      "exchanges": [
        {
          "exchangeIdx": 0,
          "speaker": "SPEAKER_08",
          "startSec": 64.325,
          "endSec": 105.753,
          "wordCount": 87,
          "text": "Well, to answer your first question, what we said is correct that as of now, we have $73 billion of backlog in place of XPUs, switches, PSPs, lasers for AI data centers that we anticipate shipping over the next 18 months. And obviously, this is as of now. I mean, we fully expect more bookings to come in over that period of time. And so don't take that 73 as that's the revenue we ship over the next 18 months. We're just saying we have that now."
        },
        {
          "exchangeIdx": 1,
          "speaker": "SPEAKER_08",
          "startSec": 106.023,
          "endSec": 178.282,
          "wordCount": 147,
          "text": "And then the bookings has been accelerating. And frankly, we see that bookings not just in XP use, but in switches, DSPs, all the other components that go into AI data center. We have never seen bookings of the nature of what we have seen over the past three months. Particularly with respect to TomHawk 6 switches. This is one of the fastest growing products in terms of deployment that we ever seen of any switch product that we put out there. It is pretty interesting and partly because it is the only one of its kind out there at this point at 102 terabits per second and that's the exact product needed to expand the clusters of the latest GPU and XP use out there. Oh, that's great. But as far as what is the future as XPU is your broader question, my answer to you"
        },
        {
          "exchangeIdx": 2,
          "speaker": "SPEAKER_08",
          "startSec": 178.906,
          "endSec": 287.125,
          "wordCount": 208,
          "text": "don't follow what you hear out there as gospel. It's a trajectory, it's a multi-year journey and many of the players and not too many players doing LLMs wants to do their own custom AI accelerator. For very good reasons. You can put in hardware, if you use a general purpose GPU, you can only do in software and kernels in software. You can achieve performance wise so much better in the custom purpose design hardware driven XPU. And we see that in the TPU, And we see that in all the accelerators we are doing for other customers, much, much better in areas of sparse call, training, inference, reasoning, all that stuff. Now, will that mean that over time they all want to go do it themselves? Not necessarily. And in fact, because the technology in silicon keeps updating, keeps evolving. And if you are an LLM player, where do you put your resources in order to compete in this space? Especially when you have to compete at the end of the day against merchant GPU who are not slowing down in the rate of evolution. So I see that as this concept of customer tooling is an overblown hypothesis, which frankly, I don't think will happen."
        },
        {
          "exchangeIdx": 3,
          "speaker": "SPEAKER_08",
          "startSec": 325.955,
          "endSec": 397.977,
          "wordCount": 132,
          "text": "And so that's a very good question, Ross. And what we see right now is the most obvious move it does is it goes, the people who use it, TPUs, The alternative is GPUs, merchant basis. That's the most common thing that happens. Because to do that substitution for another custom, that's different. To make an investment in custom accelerator is a multi-year journey. It's a strategic directional thing. It's not necessary. a very transactional or short-term move. Moving from GPU to TPU is a transactional move. Going into AI accelerator of your own is a long-term strategic move and nothing would deter you from there to continue to make that investment towards that end goal of successfully creating and deploying your own custom AI accelerator. So that's the motion we see."
        },
        {
          "exchangeIdx": 4,
          "speaker": "SPEAKER_08",
          "startSec": 483.533,
          "endSec": 545.296,
          "wordCount": 136,
          "text": "Thanks. Well, to answer your first simpler question, you're right. You can say that 73 billion is the backlog we have today to ship over the next six quarters. You might also say that I'm given our lead time. We expect more orders to be able to be absorbed into a backlog for shipments over the next six quarters. So take it that we expect revenue, a minimum revenue, one way to look at it, of 73 billion over the next six quarters. But we do expect much more as small orders come in for shipments within the next six quarters. Our lead time, depending on the particular product it is, can be anywhere from six months to a year. on with respect to supply chain is what you're asking critical supply chain on silicon and"
        },
        {
          "exchangeIdx": 5,
          "speaker": "SPEAKER_08",
          "startSec": 545.785,
          "endSec": 643.424,
          "wordCount": 175,
          "text": "packaging. Yeah, that's an interesting challenge that we are have been addressing over for constantly and continue to and with the strength of the demand and the need for more innovative packaging, advanced packaging, because you're talking about multi-chips, multi-chips in creating every customer accelerator now. The packaging becomes a very interesting and technical challenge. And building our Singapore Fab is to really talk about partially insourcing those advanced packaging. We believe that we have enough demand, we can literally insource not from the viewpoint of not just cost, but in the viewpoint of supply chain security and delivery. And we're building up fairly substantial facility for packaging, advanced packaging Singapore has indicated purely for that purpose to address the package, advanced packaging side. Silicon wise, now we go back to the same pressure source in Taiwan TSMC. And so we keep going for more and more capacity in two nanometers, three nanometers. And so far, we do not have that constraint. But again, time will tell as we progress and as our backlog builds"
        },
        {
          "exchangeIdx": 6,
          "speaker": "SPEAKER_08",
          "startSec": 690.995,
          "endSec": 749.213,
          "wordCount": 131,
          "text": "That's a very complicated question, Blaine. Let me tell you what it is. It's a system sale. How about that? It's a real system sale. We have so many components beyond XP used, custom accelerators in any system, in AI system, any AI system, used by hyperscalers that, yeah, we believe it begin to make sense to do it as a system sales and be responsible, but be fully responsible for the entire system or rank, as you call it. I think people understand it as a system sale better. And so on this customer number four, we are selling it as a system with our key components in it. And that's no different than selling a chip we certify and final ability to run as part of the whole selling"
        },
        {
          "exchangeIdx": 7,
          "speaker": "SPEAKER_08",
          "startSec": 815.195,
          "endSec": 878.678,
          "wordCount": 151,
          "text": "I'll let Kirsten give you the details, but enough for me to broadly high level explain to you, Stacy. Good question. Phenomenal. You don't see that impacting us right now, and we have already started that process. of some system sales. You don't see that in our numbers, but it will. And we have said that openly. The AI revenue has a lower gross margin than our, obviously, the rest of our business, including software, of course. But we expect the rate of growth of, as we do more and more AI revenue to be so, so much that we get the operating leverage on our of operating spending that operating margin won't deliver dollars that are still a high level of growth from what it has been. So we expect operating leverage to benefit us at the operating margin level even as gross margin will start to deteriorate, high level."
        },
        {
          "exchangeIdx": 8,
          "speaker": "SPEAKER_08",
          "startSec": 983.236,
          "endSec": 1077.685,
          "wordCount": 218,
          "text": "Wow, there's a lot of question here. Let me start off with 26. You know, our backlog is very dynamic these days, as I said, and it is continuing to ramp up. And you're right. We originally, six months ago, said maybe year on year, AI revenues would grow in 26, 60, 70 percent. Q1 would double. And Q1 26 today, we're saying it doubled. When we're looking at it, because all the thresholders keeps coming in, and we give you a milestone of where we are today, which is 73 billion back log to be shipped over the next 18 months. And we do fully expect, as I answered the earlier question, for that 73 billion over the 18 months to keep growing. Now, it's a moving number as we move in time, but it will grow. And it's hard for me to pinpoint what 26 is going to look like precisely. So I'd rather not give you guys any guidance. And that's why we don't give you guidance, but we do give it for Q1. Give it time. We'll give it Q2. And you're right. It's a net to us. Is it an accelerating trend? And my answer is it's likely to be an accelerating trend as we progress through 26. Hope that answers your question."
        },
        {
          "exchangeIdx": 9,
          "speaker": "SPEAKER_08",
          "startSec": 1129.66,
          "endSec": 1205.969,
          "wordCount": 137,
          "text": "You didn't hear that answer from my last caller, Jim's question is because I didn't answer it. I didn't answer it and I'm not answering it either. It's the fifth customer and it's a real customer and it will grow. They are on their multi-year journey to their own XP use and let's leave it at that. As far as the open AI view that you have, We appreciate the fact that it is a multi-year journey that will run through 29 as our press release with OpenAI showed 10 gigawatts between 26, more like 27, 28, 29. Ben, not 26. It's more like 27, 28, 29, 10 gigawatts. That was the OpenAI discussion. And that's, I call it an agreement, an alignment of where we're headed with respect to various respected and valued customer open AI."
        },
        {
          "exchangeIdx": 10,
          "speaker": "SPEAKER_08",
          "startSec": 1205.986,
          "endSec": 1208.332,
          "wordCount": 6,
          "text": "we do not expect much in"
        },
        {
          "exchangeIdx": 11,
          "speaker": "SPEAKER_08",
          "startSec": 1259.108,
          "endSec": 1378.533,
          "wordCount": 222,
          "text": "Thank you. No, you hit it right on. The nice thing about a customer accelerator is you try not to do one size fits all and do it generationally. Each of these five customers now can create their version of an XPU customer accelerator for training and inference. And it basically is almost two parallel tracks going on almost simultaneously for each of them. So I would have plenty of versions to deal with. I don't need to create any more versions. We've got plenty of different content out there just on the basis of creating these customer accelerators. And by the way, when you do customer accelerators, you tend to put more hardware in that are unique differentiated versus trying to make it work on software and creating kernels into software. I know that's very tricky too, but think about the difference where you can create in hardware, those sparse core data routers versus the dense matrix multipliers, all in one same chip. And that's just one example of what creating customer accelerators is letting us do. Over that matter, a variation in how much memory capacity or memory bandwidth for the same customer from chip to chip just because Even in inference, you want to do more reasoning versus decoding versus something else, like pre-fill. So you literally start to"
        },
        {
          "exchangeIdx": 12,
          "speaker": "SPEAKER_08",
          "startSec": 1378.634,
          "endSec": 1396.555,
          "wordCount": 39,
          "text": "create different hardware for different aspects of how you want to train our inference and run your workloads. It's a very fascinating area. And we are seeing a lot of variations and multiple chips for each of our custom"
        },
        {
          "exchangeIdx": 13,
          "speaker": "SPEAKER_08",
          "startSec": 1454.892,
          "endSec": 1568.005,
          "wordCount": 224,
          "text": "I like the way you address this question, the way that you address the question to me. It's almost hesitant. Thank you. I appreciate that. But on your first part, yeah, we are driving growth and we begin to feel like this thing never ends. And it's a real mix back of existing customers and on existing XP use. And I'll pick part of it as XP use that we're seeing. And that's not to slow down the fact that as I indicated in my remarks and commented on the demand for switches, not just among six, among five switches, the demand for our latest 1.6 terabit per second DSPs. that enables optical interconnects for scale out, particularly. It's just very, very strong. And by extension, demand for the optical components like lasers, pin dies, just going nuts. All that come together. Now, all that is more or relatively lesser dollars when it comes to XPUs as you probably guess. To give you a sense, maybe let me look at it on a backlog side. Of the 73 billion or AI revenue backlog over the next 18 months I talked about, maybe 20 billion of it is everything else. The rest is XPUs. Hope that gives you a sense of what the mix is. But the rest is still 20 billion. That's not small"
        },
        {
          "exchangeIdx": 14,
          "speaker": "SPEAKER_08",
          "startSec": 1568.343,
          "endSec": 1650.035,
          "wordCount": 173,
          "text": "by any means. So we value that. So when you talk about your next question of silicon photonics and as a means to create basically much better, more efficient lower power interconnects in not just scale out, but hopefully scale up. Yeah, I could see a point in time in the future when silicon photonics matters as the only way to do it. We're not quite there yet, but we have the technology and we continue to develop the technology. And even at each time, we develop it first for 400 gigabits bandwidth, going on to 900, 800 gigabit bandwidth, not ready for it yet. And even we have the product and we're now doing it for 1.6 terabit bandwidth to create silicon photonics, switches, silicon photonics interconnects, not even sure it will get fully deployed because engineers, our engineers, our peers, and the peers we have out there was somehow trying to find a way to still do, try to do scale up within a rack in copper as long as possible"
        },
        {
          "exchangeIdx": 15,
          "speaker": "SPEAKER_08",
          "startSec": 1650.524,
          "endSec": 1675.465,
          "wordCount": 54,
          "text": "and in scale out in no pluggable optics. The final, final straw is when you can't do it well in pluggable optics. And of course, when you can't do it even in copper, then you're right, you go to silicon photonics. And it will happen. And we're ready for it. Just saying, not anytime"
        },
        {
          "exchangeIdx": 16,
          "speaker": "SPEAKER_08",
          "startSec": 1739.0,
          "endSec": 1762.169,
          "wordCount": 32,
          "text": "It's across the board, typically. I mean, we are very fortunate in some ways that we have the product technology and the operating business lines to create multiple key leading edge components"
        },
        {
          "exchangeIdx": 17,
          "speaker": "SPEAKER_08",
          "startSec": 1762.49,
          "endSec": 1853.142,
          "wordCount": 204,
          "text": "that enables today's state of the art AI data centers. I mean, our DSP, as I said earlier, is now at 1.6 terabit per second. That's the leading edge connectivity for bandwidth for the top of the XPU and even GPU. And we intend to be that way. And we have the lasers, EMLs, Vixels, see the view with lasers that goes with it. So it's fortunate that we have all this and the key active components that go with it and we see it very quickly and we expand the capacity as we do the design to match it. And this is a long answer to what I'm trying to get at which is I think we are of any of this data center suppliers of the system racks, not counting the power, the power shell and all that. Now that starts to get beyond us on the power shell and the transformers and the gas turbines. If you just look at the racks, the systems on AI, we probably have a good handle on where the bottlenecks are because sometimes we are part of the bottlenecks, which we then want to get to resolve. So we feel pretty good about that through"
        },
        {
          "exchangeIdx": 18,
          "speaker": "SPEAKER_08",
          "startSec": 1911.412,
          "endSec": 2017.792,
          "wordCount": 198,
          "text": "Well, on the non-AI semiconductor, we see broadband literally recovering very well. And we don't see the others. No, we see stability. We don't see a sharp recovery that is sustainable yet. So I guess, given a couple more quarters, But we don't see any further deterioration in demand. And it's more, I think, maybe the AI is sucking the oxygen a lot out of enterprise spending elsewhere and hyperscalar spending elsewhere. We don't see getting any worse. We don't see it recovering very quickly with the exception of broadband. That's a simple summary of non-AI. With respect to open AI, Now, before diving into this, I'm just telling you what that 10 gigawatt announcement is all about. Separately, the journey with them on the customer accelerator progresses at a very advanced stage and will happen very, very quickly. And it will have a committed element to this whole thing. And it will. What we, I was articulating earlier was the 10 gigawatt announcement. And that 10 gigawatt announcement is an agreement to be aligned on developing 10 gigawatts for open AI over 27 to 29 time frame. That's different from the XPU program we were developing with"
        },
        {
          "exchangeIdx": 19,
          "speaker": "SPEAKER_08",
          "startSec": 2058.865,
          "endSec": 2102.234,
          "wordCount": 99,
          "text": "Well, it's an interesting question, and that question basically comes to how much compute capacity is needed by our customers over the next, as I say, over the period beyond 18 months? And your guess is probably as good as mine based on what we all know out there, which is really what they relate to. But if they need more, then you see that continuing even larger. If they don't need it, then probably it won't. But as of what we're trying to indicate is that's the demand we are seeing over that period of time right now."
        }
      ]
    }
  },
  "VRTX": {
    "symbol": "VRTX",
    "name": "VERTEX PHARMACEUTICALS INC / MA",
    "sector": "healthcare",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-04",
    "bias": "BULLISH",
    "finalScore": 26.22,
    "overallScore": 26.22,
    "confidence": 73,
    "probBull": 57,
    "probBear": 43,
    "mlScore": 7.08,
    "decisionInputs": {
      "epsSurprisePct": 7.05,
      "revenueSurprisePct": -0.1,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -6.37,
      "netMarginPct": 34.53,
      "fcfMarginPct": 43.35,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 3.95,
      "epsEstimate": 3.69,
      "revenueActual": "3.0B",
      "revenueEstimate": "3.0B",
      "guidanceRevenueMid": "13.0B",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "1.3B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "warn",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "LLY",
          "MRK",
          "UNH"
        ],
        "sharedCount": 4
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-12",
        "openReturnPct": 1.42,
        "oneDayReturnPct": 5.69,
        "oneWeekReturnPct": 3.44,
        "oneMonthReturnPct": -54.41
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-03",
        "openReturnPct": -1.75,
        "oneDayReturnPct": -1.02,
        "oneWeekReturnPct": 75.59,
        "oneMonthReturnPct": 7.36
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-04",
        "openReturnPct": -13.56,
        "oneDayReturnPct": -20.6,
        "oneWeekReturnPct": -17.89,
        "oneMonthReturnPct": -15.95
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-05",
        "openReturnPct": -6.92,
        "oneDayReturnPct": -10.03,
        "oneWeekReturnPct": -13.63,
        "oneMonthReturnPct": -11.27
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 67,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "warn",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "LLY",
          "MRK",
          "UNH"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 67,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "CRM": {
    "symbol": "CRM",
    "name": "Salesforce, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-25",
    "bias": "BULLISH",
    "finalScore": 25.1,
    "overallScore": 25.1,
    "confidence": 63,
    "probBull": 41,
    "probBear": 59,
    "mlScore": -25.96,
    "decisionInputs": {
      "epsSurprisePct": 33.18,
      "revenueSurprisePct": 0.18,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 2.85,
      "epsEstimate": 2.14,
      "revenueActual": "11.2B",
      "revenueEstimate": "11.2B",
      "guidanceRevenueMid": "46.00",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Generative AI Products",
        "sentiment": "pos",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "SNOW",
          "PLTR",
          "NOW"
        ],
        "sharedCount": 6
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-03",
        "openReturnPct": 2.08,
        "oneDayReturnPct": 3.66,
        "oneWeekReturnPct": 9.9,
        "oneMonthReturnPct": 10.13
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-03",
        "openReturnPct": -6.39,
        "oneDayReturnPct": -4.85,
        "oneWeekReturnPct": -3.97,
        "oneMonthReturnPct": -6.27
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-29",
        "openReturnPct": 43.46,
        "oneDayReturnPct": -58.07,
        "oneWeekReturnPct": 2.84,
        "oneMonthReturnPct": 1.87
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-28",
        "openReturnPct": -4.51,
        "oneDayReturnPct": -3.3,
        "oneWeekReturnPct": -3.22,
        "oneMonthReturnPct": -1.21
      }
    ],
    "topicDetails": [
      {
        "label": "Generative AI Products",
        "sentiment": "pos",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 62,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Developer Platforms",
        "sentiment": "pos",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "SNOW",
          "PLTR",
          "NOW"
        ],
        "sharedCount": 6,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 62,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 62,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "pos",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 62,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "MSFT": {
    "symbol": "MSFT",
    "name": "MICROSOFT CORP",
    "sector": "call-analysis",
    "hasCallAnalysis": true,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-29",
    "bias": "BULLISH",
    "finalScore": 24.49,
    "overallScore": 24.49,
    "confidence": 72,
    "probBull": 43,
    "probBear": 57,
    "mlScore": -21.78,
    "decisionInputs": {
      "epsSurprisePct": 4.91,
      "revenueSurprisePct": 1.78,
      "guidanceRevenueSurprisePct": -0.4,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 1.98,
      "netMarginPct": 38.34,
      "fcfMarginPct": 19.07,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 4.27,
      "epsEstimate": 4.07,
      "revenueActual": "82.9B",
      "revenueEstimate": "81.4B",
      "guidanceRevenueMid": "87.2B",
      "guidanceRevenueConsensus": "87.6B",
      "freeCashFlow": "15.8B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "callAnalysis": {
      "period": "Q2_2026",
      "callDate": "2026-01-28",
      "turnCount": 23,
      "overall": {
        "sentiment": 66,
        "confidence": 80,
        "risk": 16,
        "uncertainty": 26,
        "defensiveness": 15,
        "analystPressure": 18,
        "guidanceStrength": 42,
        "negativeMixed": 13
      },
      "prepared": {
        "positiveLang": 51,
        "negativeLang": 18,
        "riskLanguage": 8,
        "uncertainty": 18,
        "analystPressure": 8,
        "defensiveLang": 7,
        "guidanceStrength": 42
      },
      "qa": {
        "positiveLang": 14,
        "negativeLang": 3,
        "riskLanguage": 28,
        "uncertainty": 36,
        "analystPressure": 36,
        "defensiveLang": 27,
        "guidanceStrength": 32
      },
      "topics": [
        {
          "label": "Microsoft Cloud Growth",
          "sentiment": "pos",
          "sentimentScore": 0.54,
          "riskScore": 0.11,
          "negativeMixed": 0
        },
        {
          "label": "Azure Data Center Expansion",
          "sentiment": "warn",
          "sentimentScore": 0.12,
          "riskScore": 0.22,
          "negativeMixed": 0.25
        },
        {
          "label": "AI Integration And Development",
          "sentiment": "warn",
          "sentimentScore": 0.4,
          "riskScore": 0.25,
          "negativeMixed": 0.25
        },
        {
          "label": "Other",
          "sentiment": "neut",
          "sentimentScore": 0,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Capital Allocation",
          "sentiment": "neg",
          "sentimentScore": 0.15,
          "riskScore": 0.3,
          "negativeMixed": 0.5
        },
        {
          "label": "Microsoft 365 Copilot Adoption",
          "sentiment": "pos",
          "sentimentScore": 0.7,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Margins Profitability",
          "sentiment": "pos",
          "sentimentScore": 0.4,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Guidance Outlook",
          "sentiment": "warn",
          "sentimentScore": 0,
          "riskScore": 0.4,
          "negativeMixed": 0
        },
        {
          "label": "Azure",
          "sentiment": "neut",
          "sentimentScore": null,
          "riskScore": null,
          "negativeMixed": null
        },
        {
          "label": "Copilot",
          "sentiment": "neut",
          "sentimentScore": null,
          "riskScore": null,
          "negativeMixed": null
        },
        {
          "label": "AI Infrastructure",
          "sentiment": "warn",
          "description": "AI compute, data centers, networking, and infrastructure demand",
          "sharedWith": [
            "NVDA",
            "GOOGL",
            "META",
            "AMZN",
            "AMD",
            "AVGO",
            "ORCL",
            "SMCI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Generative AI Products",
          "sentiment": "warn",
          "description": "AI applications, copilots, automation, and productized model features",
          "sharedWith": [
            "GOOGL",
            "META",
            "AMZN",
            "PLTR",
            "CRM",
            "NOW",
            "APP",
            "AI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Cloud Growth",
          "sentiment": "warn",
          "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
          "sharedWith": [
            "AMZN",
            "GOOGL",
            "ORCL",
            "NET",
            "DDOG",
            "DOCN",
            "SNOW"
          ],
          "sharedCount": 8
        },
        {
          "label": "Data Center Capex",
          "sentiment": "warn",
          "description": "Capex intensity, server buildout, and capacity constraints",
          "sharedWith": [
            "NVDA",
            "GOOGL",
            "META",
            "AMZN",
            "AVGO",
            "AMD",
            "SMCI"
          ],
          "sharedCount": 8
        }
      ],
      "audio": {
        "available": false,
        "source": "No extracted audio feature file in this snapshot",
        "confidence": null,
        "vocalStress": null,
        "instability": null,
        "paceControl": null,
        "clarity": null,
        "segmentCount": 0
      },
      "history": [
        {
          "quarter": "Q1_2025",
          "sentiment": 67,
          "risk": 14,
          "uncertainty": 32,
          "negativeMixed": 17,
          "excessReturn5d": -1.44
        },
        {
          "quarter": "Q2_2025",
          "sentiment": 76,
          "risk": 13,
          "uncertainty": 22,
          "negativeMixed": 6,
          "excessReturn5d": -1.05
        },
        {
          "quarter": "Q4_2025",
          "sentiment": 77,
          "risk": 11,
          "uncertainty": 21,
          "negativeMixed": 0,
          "excessReturn5d": -3.12
        },
        {
          "quarter": "Q1_2026",
          "sentiment": 65,
          "risk": 21,
          "uncertainty": 31,
          "negativeMixed": 22,
          "excessReturn5d": -3.15
        },
        {
          "quarter": "Q2_2026",
          "sentiment": 66,
          "risk": 16,
          "uncertainty": 26,
          "negativeMixed": 13,
          "excessReturn5d": -4.04
        }
      ]
    },
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "AVGO",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Generative AI Products",
        "sentiment": "warn",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Cloud Growth",
        "sentiment": "warn",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "AMZN",
          "GOOGL",
          "ORCL",
          "NET",
          "DDOG",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "AMD",
          "SMCI"
        ],
        "sharedCount": 8
      },
      {
        "label": "Margins & Cost Control",
        "sentiment": "warn",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-28",
        "openReturnPct": -8.65,
        "oneDayReturnPct": -9.99,
        "oneWeekReturnPct": -18.26,
        "oneMonthReturnPct": -17.25
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-29",
        "openReturnPct": -2.04,
        "oneDayReturnPct": -2.92,
        "oneWeekReturnPct": -8.21,
        "oneMonthReturnPct": -10.12
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-30",
        "openReturnPct": 8.18,
        "oneDayReturnPct": 3.95,
        "oneWeekReturnPct": 1.48,
        "oneMonthReturnPct": -1.28
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-30",
        "openReturnPct": 9.07,
        "oneDayReturnPct": 7.63,
        "oneWeekReturnPct": 10.86,
        "oneMonthReturnPct": 16.88
      }
    ],
    "topicDetails": [
      {
        "label": "Microsoft Cloud Growth",
        "mentions": 7,
        "sentimentScore": 0.543,
        "riskScore": 0.114,
        "uncertaintyScore": 0.229,
        "qualityScore": 79,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.4282930569672308,
        "riskCorrelation5d": 0.922209073007696,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.5318179282954603,
            "riskCorrelation": 0.16105099803821527,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.08349827067324843,
            "riskCorrelation": 0.8493380623031315,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.4282930569672308,
            "riskCorrelation": 0.922209073007696,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.35106550240458534,
            "riskCorrelation": -0.16996239881439934,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.6098201501453593,
            "riskCorrelation": 0.12734788298933813,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.4007955656516554,
            "riskCorrelation": 0.6718944534471548,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Azure Data Center Expansion",
        "mentions": 4,
        "sentimentScore": 0.125,
        "riskScore": 0.225,
        "uncertaintyScore": 0.325,
        "qualityScore": 58,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.9608063521949533,
        "riskCorrelation5d": -0.7614285251609665,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.17598022816379474,
            "riskCorrelation": 0.5657016714143585,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": -0.9806522014118325,
            "riskCorrelation": -0.9742666753544263,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.9608063521949533,
            "riskCorrelation": -0.7614285251609665,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": 0.9177520820521392,
            "riskCorrelation": 0.9998724978978322,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.9926704342534707,
            "riskCorrelation": -0.8548516592396683,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.6531852192887813,
            "riskCorrelation": -0.28346074790790815,
            "nEvents": 3
          }
        }
      },
      {
        "label": "AI Integration and Development",
        "mentions": 4,
        "sentimentScore": 0.4,
        "riskScore": 0.25,
        "uncertaintyScore": 0.35,
        "qualityScore": 65,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.5446976201538271,
        "riskCorrelation5d": -0.6922638305918084,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.7932872629556889,
            "riskCorrelation": -0.71794004620011,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.4240328598613174,
            "riskCorrelation": -0.5672752321311519,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.5446976201538271,
            "riskCorrelation": -0.6922638305918084,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.16466263013781457,
            "riskCorrelation": 0.003307075913012579,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.4662958646107585,
            "riskCorrelation": 0.1708563899222678,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.48643343383529725,
            "riskCorrelation": -0.6539891345627445,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Other",
        "mentions": 3,
        "sentimentScore": 0.0,
        "riskScore": 0.0,
        "uncertaintyScore": 0.0,
        "qualityScore": 58,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.22761918471735237,
        "riskCorrelation5d": null,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.701374459412495,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.7453840858503413,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.22761918471735237,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.776397302329767,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.3643270215332984,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.13195303573457282,
            "riskCorrelation": null,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Capital Allocation",
        "mentions": 2,
        "sentimentScore": 0.15,
        "riskScore": 0.3,
        "uncertaintyScore": 0.4,
        "qualityScore": 53,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.3902160897289345,
        "riskCorrelation5d": -0.5999432320550848,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.5389896884271009,
            "riskCorrelation": 0.7255278356499955,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.12613841857717228,
            "riskCorrelation": -0.36071949117576785,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.3902160897289345,
            "riskCorrelation": -0.5999432320550848,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": 0.8337664618881524,
            "riskCorrelation": -0.6767439547676362,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.8357118484099987,
            "riskCorrelation": 0.6793407372295912,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": 0.05262558088059943,
            "riskCorrelation": -0.29094438478125084,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Margins / Profitability",
        "mentions": 1,
        "sentimentScore": 0.4,
        "riskScore": 0.0,
        "uncertaintyScore": 0.1,
        "qualityScore": 66,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.8499800233064586,
        "riskCorrelation5d": 0.5268149200430374,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9086974382465428,
            "riskCorrelation": 0.41745534578463667,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.6839023358718501,
            "riskCorrelation": 0.7295735706486544,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.8499800233064586,
            "riskCorrelation": 0.5268149200430374,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.7563845449151828,
            "riskCorrelation": -0.6541272202052532,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": 0.38993031292577096,
            "riskCorrelation": -0.9208443685344498,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.868181892963196,
            "riskCorrelation": -0.49624610903345273,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Microsoft 365 Copilot Adoption",
        "mentions": 1,
        "sentimentScore": 0.7,
        "riskScore": 0.0,
        "uncertaintyScore": 0.2,
        "qualityScore": 74,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.7537718058625865,
        "riskCorrelation5d": 0.8178453750976123,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.6548845234239994,
            "riskCorrelation": -0.39798523856510876,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.819125446368791,
            "riskCorrelation": 0.4488241105459079,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.7537718058625865,
            "riskCorrelation": 0.8178453750976123,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.5488622767234288,
            "riskCorrelation": -0.12000916831298584,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.5516653636108347,
            "riskCorrelation": 0.8148360875689824,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.6548718001076625,
            "riskCorrelation": 0.9391996597630639,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Guidance / Outlook",
        "mentions": 1,
        "sentimentScore": 0.0,
        "riskScore": 0.4,
        "uncertaintyScore": 0.5,
        "qualityScore": 44,
        "sentiment": "warn",
        "sentimentCorrelation5d": -0.11196041210243633,
        "riskCorrelation5d": -0.7722657077978877,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.2759871162806176,
            "riskCorrelation": -0.6298793546925368,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.5055381551826779,
            "riskCorrelation": -0.5993336149394054,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.11196041210243633,
            "riskCorrelation": -0.7722657077978877,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": -0.5178309446229378,
            "riskCorrelation": 0.44653080321926963,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.8345041491562682,
            "riskCorrelation": -0.12715791689059386,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.4368263293552168,
            "riskCorrelation": -0.7056164664894746,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Macro / Demand",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 1.0,
        "riskCorrelation5d": -1.0,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          }
        }
      }
    ],
    "sentimentHorizon": [
      {
        "horizonDays": 1,
        "rho": 0.436,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 3,
        "rho": 0.494,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 5,
        "rho": 0.345,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 7,
        "rho": 0.104,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 10,
        "rho": -0.803,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 21,
        "rho": 0.04,
        "nEvents": 5,
        "indicativeOnly": true
      }
    ]
  },
  "NVDA": {
    "symbol": "NVDA",
    "name": "NVIDIA CORP",
    "sector": "call-analysis",
    "hasCallAnalysis": true,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-25",
    "bias": "BULLISH",
    "finalScore": 22.77,
    "overallScore": 22.77,
    "confidence": 62,
    "probBull": 43,
    "probBear": 57,
    "mlScore": -22.11,
    "decisionInputs": {
      "epsSurprisePct": 8.28,
      "revenueSurprisePct": 3.92,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.57,
      "epsEstimate": 1.45,
      "revenueActual": "68.1B",
      "revenueEstimate": "65.6B",
      "guidanceRevenueMid": "78.0B",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "callAnalysis": {
      "period": "Q4_2026",
      "callDate": "2026-02-25",
      "turnCount": 29,
      "overall": {
        "sentiment": 71,
        "confidence": 82,
        "risk": 10,
        "uncertainty": 26,
        "defensiveness": 10,
        "analystPressure": 13,
        "guidanceStrength": 41,
        "negativeMixed": 10
      },
      "prepared": {
        "positiveLang": 91,
        "negativeLang": 1,
        "riskLanguage": 2,
        "uncertainty": 18,
        "analystPressure": 3,
        "defensiveLang": 2,
        "guidanceStrength": 41
      },
      "qa": {
        "positiveLang": 88,
        "negativeLang": 1,
        "riskLanguage": 22,
        "uncertainty": 36,
        "analystPressure": 31,
        "defensiveLang": 22,
        "guidanceStrength": 31
      },
      "topics": [
        {
          "label": "Data Center Revenue",
          "sentiment": "pos",
          "sentimentScore": 0.37,
          "riskScore": 0.13,
          "negativeMixed": 0.17
        },
        {
          "label": "AI Factory Deployments",
          "sentiment": "pos",
          "sentimentScore": 0.77,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Other",
          "sentiment": "neut",
          "sentimentScore": 0,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Capital Allocation",
          "sentiment": "warn",
          "sentimentScore": 0.43,
          "riskScore": 0.13,
          "negativeMixed": 0.33
        },
        {
          "label": "Blackwell Architecture",
          "sentiment": "neut",
          "sentimentScore": 0.33,
          "riskScore": 0.13,
          "negativeMixed": 0
        },
        {
          "label": "Networking Solutions",
          "sentiment": "pos",
          "sentimentScore": 0.7,
          "riskScore": 0.05,
          "negativeMixed": 0
        },
        {
          "label": "Margins Profitability",
          "sentiment": "pos",
          "sentimentScore": 0.45,
          "riskScore": 0.2,
          "negativeMixed": 0
        },
        {
          "label": "Macro Demand",
          "sentiment": "neg",
          "sentimentScore": 0.15,
          "riskScore": 0.25,
          "negativeMixed": 0.5
        },
        {
          "label": "Inference Performance",
          "sentiment": "pos",
          "sentimentScore": 0.4,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Generative AI Applications",
          "sentiment": "pos",
          "sentimentScore": 0.7,
          "riskScore": 0.15,
          "negativeMixed": 0
        },
        {
          "label": "AI Infrastructure",
          "sentiment": "pos",
          "description": "AI compute, data centers, networking, and infrastructure demand",
          "sharedWith": [
            "MSFT",
            "GOOGL",
            "META",
            "AMZN",
            "AMD",
            "AVGO",
            "ORCL",
            "SMCI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Semiconductor Cycle",
          "sentiment": "pos",
          "description": "Chip demand, supply, memory, equipment, and hardware cycle",
          "sharedWith": [
            "AMD",
            "AVGO",
            "QCOM",
            "AMAT",
            "MU",
            "SMCI"
          ],
          "sharedCount": 7
        },
        {
          "label": "Data Center Capex",
          "sentiment": "pos",
          "description": "Capex intensity, server buildout, and capacity constraints",
          "sharedWith": [
            "MSFT",
            "GOOGL",
            "META",
            "AMZN",
            "AVGO",
            "AMD",
            "SMCI"
          ],
          "sharedCount": 8
        },
        {
          "label": "Guidance Quality",
          "sentiment": "pos",
          "description": "Management guidance availability and surprise versus consensus",
          "sharedWith": [
            "MU",
            "PANW",
            "LLY",
            "PLTR",
            "GTLB",
            "DDOG",
            "GOOGL",
            "DOCN",
            "V",
            "JPM",
            "CRWD",
            "NET",
            "APP",
            "ORCL",
            "AMD",
            "XYZ",
            "META",
            "KO",
            "AVGO",
            "VRTX",
            "CRM",
            "MSFT",
            "WMT",
            "UNH",
            "AAPL",
            "CMG",
            "QCOM",
            "COST",
            "MRK",
            "AMAT",
            "AMZN",
            "PG",
            "MCD",
            "NOW",
            "SMCI",
            "SNOW",
            "ABNB",
            "ZS",
            "AI"
          ],
          "sharedCount": 40
        }
      ],
      "audio": {
        "available": false,
        "source": "No extracted audio feature file in this snapshot",
        "confidence": null,
        "vocalStress": null,
        "instability": null,
        "paceControl": null,
        "clarity": null,
        "segmentCount": 0
      },
      "history": [
        {
          "quarter": "Q3_2025",
          "sentiment": 66,
          "risk": 12,
          "uncertainty": 32,
          "negativeMixed": 24,
          "excessReturn5d": -5.43
        },
        {
          "quarter": "Q4_2025",
          "sentiment": 68,
          "risk": 17,
          "uncertainty": 31,
          "negativeMixed": 18,
          "excessReturn5d": -4.46
        },
        {
          "quarter": "Q1_2026",
          "sentiment": 60,
          "risk": 8,
          "uncertainty": 34,
          "negativeMixed": 39,
          "excessReturn5d": -2.68
        },
        {
          "quarter": "Q2_2026",
          "sentiment": 68,
          "risk": 16,
          "uncertainty": 32,
          "negativeMixed": 11,
          "excessReturn5d": -5.12
        },
        {
          "quarter": "Q3_2026",
          "sentiment": 68,
          "risk": 8,
          "uncertainty": 26,
          "negativeMixed": 9,
          "excessReturn5d": -12.7
        },
        {
          "quarter": "Q4_2026",
          "sentiment": 71,
          "risk": 10,
          "uncertainty": 26,
          "negativeMixed": 10,
          "excessReturn5d": 4.5
        }
      ]
    },
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "pos",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "AVGO",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "pos",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "AMD",
          "AVGO",
          "QCOM",
          "AMAT",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7
      },
      {
        "label": "Data Center Capex",
        "sentiment": "pos",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "AMD",
          "SMCI"
        ],
        "sharedCount": 8
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "QCOM",
          "JPM",
          "V",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-11-19",
        "openReturnPct": 5.06,
        "oneDayReturnPct": -3.15,
        "oneWeekReturnPct": -5.1,
        "oneMonthReturnPct": -1.52
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-08-27",
        "openReturnPct": -42.95,
        "oneDayReturnPct": -78.74,
        "oneWeekReturnPct": -8.03,
        "oneMonthReturnPct": 13.77
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-28",
        "openReturnPct": 5.52,
        "oneDayReturnPct": 3.25,
        "oneWeekReturnPct": 3.84,
        "oneMonthReturnPct": 17.19
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-28",
        "openReturnPct": 5.52,
        "oneDayReturnPct": 3.25,
        "oneWeekReturnPct": 3.84,
        "oneMonthReturnPct": 17.19
      }
    ],
    "topicDetails": [
      {
        "label": "Data Center Revenue",
        "mentions": 6,
        "sentimentScore": 0.367,
        "riskScore": 0.133,
        "uncertaintyScore": 0.35,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.09829088176421735,
        "riskCorrelation5d": -0.33952530541574394,
        "nEvents": 7,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.11467599676086704,
            "riskCorrelation": 0.18547926559039846,
            "nEvents": 7
          },
          "3": {
            "sentimentCorrelation": -0.4187337914545342,
            "riskCorrelation": -0.11904560570690778,
            "nEvents": 7
          },
          "5": {
            "sentimentCorrelation": -0.09829088176421735,
            "riskCorrelation": -0.33952530541574394,
            "nEvents": 7
          },
          "7": {
            "sentimentCorrelation": -0.017997851825624907,
            "riskCorrelation": -0.1972574228676932,
            "nEvents": 7
          },
          "10": {
            "sentimentCorrelation": 0.18415101058760358,
            "riskCorrelation": -0.1114234033801794,
            "nEvents": 7
          },
          "21": {
            "sentimentCorrelation": -0.12905979330010836,
            "riskCorrelation": -0.41176292308371804,
            "nEvents": 7
          }
        }
      },
      {
        "label": "AI Factory Deployments",
        "mentions": 4,
        "sentimentScore": 0.775,
        "riskScore": 0.0,
        "uncertaintyScore": 0.1,
        "qualityScore": 82,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.007907824397983983,
        "riskCorrelation5d": 0.09428846307820885,
        "nEvents": 7,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.13311297665976163,
            "riskCorrelation": 0.17420759587732138,
            "nEvents": 7
          },
          "3": {
            "sentimentCorrelation": 0.17930719837876583,
            "riskCorrelation": 0.12427321831702678,
            "nEvents": 7
          },
          "5": {
            "sentimentCorrelation": 0.007907824397983983,
            "riskCorrelation": 0.09428846307820885,
            "nEvents": 7
          },
          "7": {
            "sentimentCorrelation": -0.07206201140638467,
            "riskCorrelation": 0.19726730667604267,
            "nEvents": 7
          },
          "10": {
            "sentimentCorrelation": 0.040165919050596,
            "riskCorrelation": 0.10769685278246255,
            "nEvents": 7
          },
          "21": {
            "sentimentCorrelation": -0.28320055331164945,
            "riskCorrelation": 0.3426091762111526,
            "nEvents": 7
          }
        }
      },
      {
        "label": "Blackwell Architecture",
        "mentions": 3,
        "sentimentScore": 0.333,
        "riskScore": 0.133,
        "uncertaintyScore": 0.367,
        "qualityScore": 64,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.12302857802595772,
        "riskCorrelation5d": -0.31749916144034795,
        "nEvents": 7,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.6060231980468758,
            "riskCorrelation": -0.472132989579475,
            "nEvents": 7
          },
          "3": {
            "sentimentCorrelation": 0.43273253010507756,
            "riskCorrelation": -0.6274202190621583,
            "nEvents": 7
          },
          "5": {
            "sentimentCorrelation": 0.12302857802595772,
            "riskCorrelation": -0.31749916144034795,
            "nEvents": 7
          },
          "7": {
            "sentimentCorrelation": -0.008683468100754424,
            "riskCorrelation": -0.23655721415037942,
            "nEvents": 7
          },
          "10": {
            "sentimentCorrelation": 0.023932002633940568,
            "riskCorrelation": -0.18785456767330877,
            "nEvents": 7
          },
          "21": {
            "sentimentCorrelation": 0.3570928777327702,
            "riskCorrelation": -0.4405873677291083,
            "nEvents": 7
          }
        }
      },
      {
        "label": "Capital Allocation",
        "mentions": 3,
        "sentimentScore": 0.433,
        "riskScore": 0.133,
        "uncertaintyScore": 0.233,
        "qualityScore": 67,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.023002527099003196,
        "riskCorrelation5d": 0.3906871153600496,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9055937150008689,
            "riskCorrelation": 0.6506157264734785,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.3778325025672263,
            "riskCorrelation": 0.03691993134808469,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.023002527099003196,
            "riskCorrelation": 0.3906871153600496,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.08544388040599109,
            "riskCorrelation": 0.48811672955120533,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.2415609331423103,
            "riskCorrelation": 0.6196969930572556,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.46872112941251814,
            "riskCorrelation": 0.7908673284766724,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Other",
        "mentions": 3,
        "sentimentScore": 0.0,
        "riskScore": 0.0,
        "uncertaintyScore": 0.167,
        "qualityScore": 58,
        "sentiment": "neut",
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": 7,
        "horizons": {
          "1": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 7
          },
          "3": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 7
          },
          "5": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 7
          },
          "7": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 7
          },
          "10": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 7
          },
          "21": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 7
          }
        }
      },
      {
        "label": "Networking Solutions",
        "mentions": 2,
        "sentimentScore": 0.7,
        "riskScore": 0.05,
        "uncertaintyScore": 0.15,
        "qualityScore": 74,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.4681951750766579,
        "riskCorrelation5d": 0.42929224518405085,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.6543472927988215,
            "riskCorrelation": -0.9849371783227905,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.8587269156300412,
            "riskCorrelation": -0.07805630595303155,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.4681951750766579,
            "riskCorrelation": 0.42929224518405085,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.5350277064467431,
            "riskCorrelation": 0.3808814386629053,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.4289143434452487,
            "riskCorrelation": 0.4810380218782648,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.7129700550631244,
            "riskCorrelation": -0.06394074385967902,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Margins / Profitability",
        "mentions": 2,
        "sentimentScore": 0.45,
        "riskScore": 0.2,
        "uncertaintyScore": 0.35,
        "qualityScore": 64,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.40964438916573104,
        "riskCorrelation5d": -0.19630461083907794,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.8931634036219461,
            "riskCorrelation": 0.9547106164649839,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.32891962634381144,
            "riskCorrelation": 0.08255395879026863,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.40964438916573104,
            "riskCorrelation": -0.19630461083907794,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.4858594931569752,
            "riskCorrelation": -0.2395186998421467,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.1216858562379919,
            "riskCorrelation": 0.04725261464053539,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.08516051881307682,
            "riskCorrelation": 0.2894594148637336,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Generative AI Applications",
        "mentions": 2,
        "sentimentScore": 0.7,
        "riskScore": 0.15,
        "uncertaintyScore": 0.25,
        "qualityScore": 72,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.30970351916474304,
        "riskCorrelation5d": 0.5295017215669139,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.6673897949614116,
            "riskCorrelation": 0.036856169717085475,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": 0.5231156764225662,
            "riskCorrelation": 0.16295439955253951,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": 0.30970351916474304,
            "riskCorrelation": 0.5295017215669139,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": 0.3855116831441119,
            "riskCorrelation": 0.5887143002427547,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": 0.49764595202847817,
            "riskCorrelation": 0.6587196990157982,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": 0.6186852061202432,
            "riskCorrelation": 0.7983325239517046,
            "nEvents": 6
          }
        }
      },
      {
        "label": "Macro / Demand",
        "mentions": 2,
        "sentimentScore": 0.15,
        "riskScore": 0.25,
        "uncertaintyScore": 0.4,
        "qualityScore": 54,
        "sentiment": "warn",
        "sentimentCorrelation5d": -0.28178152326865674,
        "riskCorrelation5d": 0.34534395953970515,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.8570850293596693,
            "riskCorrelation": 0.6667107585511961,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": -0.2175454116783157,
            "riskCorrelation": 0.3122170767838255,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": -0.28178152326865674,
            "riskCorrelation": 0.34534395953970515,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": -0.1903487157560381,
            "riskCorrelation": 0.3134534258600454,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": -0.5399623299938828,
            "riskCorrelation": 0.6451179900039447,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": -0.637514551461844,
            "riskCorrelation": 0.4808509666417847,
            "nEvents": 6
          }
        }
      },
      {
        "label": "Inference Performance",
        "mentions": 2,
        "sentimentScore": 0.4,
        "riskScore": 0.0,
        "uncertaintyScore": 0.2,
        "qualityScore": 68,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.0943173357056423,
        "riskCorrelation5d": 0.13113051794647243,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.686543859021341,
            "riskCorrelation": 0.8585145934669863,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.10937880822562572,
            "riskCorrelation": 0.2620369616960241,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.0943173357056423,
            "riskCorrelation": 0.13113051794647243,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.004818155004090144,
            "riskCorrelation": 0.05494966516883062,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.02044358057314499,
            "riskCorrelation": -0.008791715655765421,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.4292213247464439,
            "riskCorrelation": 0.497188243528044,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Enterprise AI Adoption",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 1.0,
        "riskCorrelation5d": null,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": 0.9999999999999998,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": null,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": null,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Sovereign AI Initiatives",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 1.0,
        "riskCorrelation5d": -1.0,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -0.9999999999999999,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -0.9999999999999999,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Regulation / Geopolitics",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.8591987144130657,
        "riskCorrelation5d": -0.49083092123110733,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.8439331050218609,
            "riskCorrelation": 0.5565892748363487,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": -0.9347292135047173,
            "riskCorrelation": 0.37773760579572674,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.8591987144130657,
            "riskCorrelation": -0.49083092123110733,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.9044588835145323,
            "riskCorrelation": 0.44818904757685324,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.24125385409671843,
            "riskCorrelation": 0.9759833474944214,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.9205028168258699,
            "riskCorrelation": -0.36848546370405344,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Hopper Architecture",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.6490044914274113,
        "riskCorrelation5d": 0.1002155480871003,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.5443652065352685,
            "riskCorrelation": -0.9251852035763931,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": -0.5617531240861343,
            "riskCorrelation": -0.9329040703639623,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.6490044914274113,
            "riskCorrelation": 0.1002155480871003,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": 0.8678364428005001,
            "riskCorrelation": 0.43040367959797626,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": 0.24266910685569965,
            "riskCorrelation": -0.3529993196754322,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": 0.24667125117235733,
            "riskCorrelation": -0.34913444327294985,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Guidance / Outlook",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      }
    ],
    "sentimentHorizon": [
      {
        "horizonDays": 1,
        "rho": -0.082,
        "nEvents": 7,
        "indicativeOnly": false
      },
      {
        "horizonDays": 3,
        "rho": 0.047,
        "nEvents": 7,
        "indicativeOnly": false
      },
      {
        "horizonDays": 5,
        "rho": 0.166,
        "nEvents": 7,
        "indicativeOnly": false
      },
      {
        "horizonDays": 7,
        "rho": 0.304,
        "nEvents": 7,
        "indicativeOnly": false
      },
      {
        "horizonDays": 10,
        "rho": 0.395,
        "nEvents": 7,
        "indicativeOnly": false
      },
      {
        "horizonDays": 21,
        "rho": 0.047,
        "nEvents": 7,
        "indicativeOnly": false
      }
    ]
  },
  "WMT": {
    "symbol": "WMT",
    "name": "Walmart Inc.",
    "sector": "consumer-retail",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-19",
    "bias": "BULLISH",
    "finalScore": 21.32,
    "overallScore": 21.32,
    "confidence": 71,
    "probBull": 56,
    "probBear": 44,
    "mlScore": 4.25,
    "decisionInputs": {
      "epsSurprisePct": 1.37,
      "revenueSurprisePct": 1.22,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.74,
      "epsEstimate": 0.73,
      "revenueActual": "190.7B",
      "revenueEstimate": "188.4B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "neut",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8
      },
      {
        "label": "Retail Scale",
        "sentiment": "neut",
        "description": "Membership, inventory, traffic, and retail operating scale",
        "sharedWith": [
          "COST",
          "AMZN"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-11-20",
        "openReturnPct": 81.22,
        "oneDayReturnPct": -1.67,
        "oneWeekReturnPct": 4.13,
        "oneMonthReturnPct": 3.54
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-08-21",
        "openReturnPct": 10.72,
        "oneDayReturnPct": -1.15,
        "oneWeekReturnPct": -1.0,
        "oneMonthReturnPct": 4.65
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-06-06",
        "openReturnPct": -7.18,
        "oneDayReturnPct": -2.05,
        "oneWeekReturnPct": -3.26,
        "oneMonthReturnPct": -2.68
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-15",
        "openReturnPct": -21.8,
        "oneDayReturnPct": 1.96,
        "oneWeekReturnPct": -1.04,
        "oneMonthReturnPct": -2.18
      }
    ],
    "topicDetails": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "COST",
          "PG"
        ],
        "sharedCount": 10,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 64,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 64,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "neut",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "COST",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 64,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Retail Scale",
        "sentiment": "neut",
        "description": "Membership, inventory, traffic, and retail operating scale",
        "sharedWith": [
          "COST",
          "AMZN"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 64,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "UNH": {
    "symbol": "UNH",
    "name": "UNITEDHEALTH GROUP INC",
    "sector": "healthcare",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-21",
    "bias": "BULLISH",
    "finalScore": 17.4,
    "overallScore": 17.4,
    "confidence": 64,
    "probBull": 54,
    "probBear": 46,
    "mlScore": -0.5,
    "decisionInputs": {
      "epsSurprisePct": 11.92,
      "revenueSurprisePct": 1.71,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -1.32,
      "netMarginPct": 5.62,
      "fcfMarginPct": 7.29,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 7.23,
      "epsEstimate": 6.46,
      "revenueActual": "111.7B",
      "revenueEstimate": "109.8B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "8.1B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "pos",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "LLY",
          "VRTX",
          "MRK"
        ],
        "sharedCount": 4
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "JPM",
          "V"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-27",
        "openReturnPct": 36.08,
        "oneDayReturnPct": 4.0,
        "oneWeekReturnPct": -2.4,
        "oneMonthReturnPct": 3.74
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-28",
        "openReturnPct": -96.78,
        "oneDayReturnPct": -3.42,
        "oneWeekReturnPct": -10.9,
        "oneMonthReturnPct": -10.35
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-29",
        "openReturnPct": 1.43,
        "oneDayReturnPct": 1.9,
        "oneWeekReturnPct": -5.86,
        "oneMonthReturnPct": 15.79
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-17",
        "openReturnPct": -95.79,
        "oneDayReturnPct": -6.34,
        "oneWeekReturnPct": -7.51,
        "oneMonthReturnPct": -29.18
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 60,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "pos",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "LLY",
          "VRTX",
          "MRK"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 60,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "pos",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "QCOM",
          "JPM",
          "V"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 60,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "AAPL": {
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "sector": "call-analysis",
    "hasCallAnalysis": true,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-30",
    "bias": "BULLISH",
    "finalScore": 12.15,
    "overallScore": 12.15,
    "confidence": 67,
    "probBull": 58,
    "probBear": 42,
    "mlScore": 8.23,
    "decisionInputs": {
      "epsSurprisePct": 4.69,
      "revenueSurprisePct": 1.58,
      "guidanceRevenueSurprisePct": 5.64,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -22.66,
      "netMarginPct": 26.6,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 3
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 2.01,
      "epsEstimate": 1.92,
      "revenueActual": "111.2B",
      "revenueEstimate": "109.5B",
      "guidanceRevenueMid": "108.6B",
      "guidanceRevenueConsensus": "102.8B",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "callAnalysis": {
      "period": "Q1_2026",
      "callDate": "2026-01-29",
      "turnCount": 49,
      "overall": {
        "sentiment": 64,
        "confidence": 78,
        "risk": 14,
        "uncertainty": 32,
        "defensiveness": 15,
        "analystPressure": 16,
        "guidanceStrength": 23,
        "negativeMixed": 16
      },
      "prepared": {
        "positiveLang": 68,
        "negativeLang": 4,
        "riskLanguage": 6,
        "uncertainty": 24,
        "analystPressure": 6,
        "defensiveLang": 7,
        "guidanceStrength": 23
      },
      "qa": {
        "positiveLang": 28,
        "negativeLang": 2,
        "riskLanguage": 26,
        "uncertainty": 42,
        "analystPressure": 34,
        "defensiveLang": 27,
        "guidanceStrength": 13
      },
      "topics": [
        {
          "label": "Iphone Lineup",
          "sentiment": "pos",
          "sentimentScore": 0.42,
          "riskScore": 0.07,
          "negativeMixed": 0.08
        },
        {
          "label": "Apple Intelligence",
          "sentiment": "neut",
          "sentimentScore": 0.18,
          "riskScore": 0.14,
          "negativeMixed": 0.11
        },
        {
          "label": "Supply Chain And Manufacturing",
          "sentiment": "neg",
          "sentimentScore": -0.3,
          "riskScore": 0.32,
          "negativeMixed": 0.6
        },
        {
          "label": "Margins Profitability",
          "sentiment": "pos",
          "sentimentScore": 0.44,
          "riskScore": 0.14,
          "negativeMixed": 0
        },
        {
          "label": "Services Growth",
          "sentiment": "pos",
          "sentimentScore": 0.47,
          "riskScore": 0,
          "negativeMixed": 0.25
        },
        {
          "label": "Emerging Market Performance",
          "sentiment": "pos",
          "sentimentScore": 0.6,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Macro Demand",
          "sentiment": "neg",
          "sentimentScore": 0.25,
          "riskScore": 0.4,
          "negativeMixed": 0.5
        },
        {
          "label": "Mac Performance",
          "sentiment": "pos",
          "sentimentScore": 0.35,
          "riskScore": 0.1,
          "negativeMixed": 0
        },
        {
          "label": "Guidance Outlook",
          "sentiment": "neg",
          "sentimentScore": 0.15,
          "riskScore": 0.45,
          "negativeMixed": 0.5
        },
        {
          "label": "AI And Machine Learning In Products",
          "sentiment": "neut",
          "sentimentScore": 0.3,
          "riskScore": 0.1,
          "negativeMixed": 0
        },
        {
          "label": "Consumer Devices",
          "sentiment": "warn",
          "description": "Hardware cycles, devices, product refreshes, and consumer demand",
          "sharedWith": [
            "TSLA",
            "QCOM"
          ],
          "sharedCount": 3
        },
        {
          "label": "Margins & Cost Control",
          "sentiment": "warn",
          "description": "Operating leverage, efficiency, pricing, and cost discipline",
          "sharedWith": [
            "AMZN",
            "META",
            "MSFT",
            "TSLA",
            "CMG",
            "MCD",
            "WMT",
            "COST",
            "PG"
          ],
          "sharedCount": 10
        },
        {
          "label": "Guidance Quality",
          "sentiment": "warn",
          "description": "Management guidance availability and surprise versus consensus",
          "sharedWith": [
            "MU",
            "PANW",
            "LLY",
            "PLTR",
            "GTLB",
            "DDOG",
            "GOOGL",
            "DOCN",
            "V",
            "JPM",
            "CRWD",
            "NET",
            "APP",
            "ORCL",
            "AMD",
            "XYZ",
            "META",
            "KO",
            "AVGO",
            "VRTX",
            "CRM",
            "MSFT",
            "NVDA",
            "WMT",
            "UNH",
            "CMG",
            "QCOM",
            "COST",
            "MRK",
            "AMAT",
            "AMZN",
            "PG",
            "MCD",
            "NOW",
            "SMCI",
            "SNOW",
            "ABNB",
            "ZS",
            "AI"
          ],
          "sharedCount": 40
        },
        {
          "label": "Regulation & Geopolitics",
          "sentiment": "warn",
          "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
          "sharedWith": [
            "GOOGL",
            "META",
            "TSLA",
            "NVDA",
            "QCOM",
            "JPM",
            "V",
            "UNH"
          ],
          "sharedCount": 9
        }
      ],
      "audio": {
        "available": true,
        "source": "Q&A audio features",
        "confidence": 56,
        "vocalStress": 44,
        "instability": 54,
        "paceControl": 60,
        "clarity": 60,
        "segmentCount": 16
      },
      "history": [
        {
          "quarter": "Q4_2024",
          "sentiment": 65,
          "risk": 15,
          "uncertainty": 31,
          "negativeMixed": 11,
          "excessReturn5d": -3.55
        },
        {
          "quarter": "Q1_2025",
          "sentiment": 67,
          "risk": 14,
          "uncertainty": 30,
          "negativeMixed": 11,
          "excessReturn5d": -3.67
        },
        {
          "quarter": "Q3_2025",
          "sentiment": 60,
          "risk": 18,
          "uncertainty": 36,
          "negativeMixed": 20,
          "excessReturn5d": 9.59
        },
        {
          "quarter": "Q4_2025",
          "sentiment": 64,
          "risk": 14,
          "uncertainty": 33,
          "negativeMixed": 15,
          "excessReturn5d": 2.37
        },
        {
          "quarter": "Q1_2026",
          "sentiment": 64,
          "risk": 14,
          "uncertainty": 32,
          "negativeMixed": 16,
          "excessReturn5d": 9.15
        }
      ]
    },
    "topics": [
      {
        "label": "Consumer Devices",
        "sentiment": "warn",
        "description": "Hardware cycles, devices, product refreshes, and consumer demand",
        "sharedWith": [
          "TSLA",
          "QCOM"
        ],
        "sharedCount": 3
      },
      {
        "label": "Margins & Cost Control",
        "sentiment": "warn",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "warn",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "TSLA",
          "NVDA",
          "QCOM",
          "JPM",
          "V",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-29",
        "openReturnPct": -1.21,
        "oneDayReturnPct": 46.46,
        "oneWeekReturnPct": 7.68,
        "oneMonthReturnPct": 2.12
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-30",
        "openReturnPct": 2.06,
        "oneDayReturnPct": -37.95,
        "oneWeekReturnPct": -1.08,
        "oneMonthReturnPct": 5.45
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-31",
        "openReturnPct": 1.59,
        "oneDayReturnPct": -2.5,
        "oneWeekReturnPct": 10.49,
        "oneMonthReturnPct": 10.67
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-01",
        "openReturnPct": -3.39,
        "oneDayReturnPct": -3.74,
        "oneWeekReturnPct": -6.93,
        "oneMonthReturnPct": -4.71
      }
    ],
    "topicDetails": [
      {
        "label": "iPhone Lineup",
        "mentions": 12,
        "sentimentScore": 0.417,
        "riskScore": 0.067,
        "uncertaintyScore": 0.233,
        "qualityScore": 82,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.6974000902350335,
        "riskCorrelation5d": -0.8982679323701424,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.4185728499033537,
            "riskCorrelation": -0.3110405200674136,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.8559721548177911,
            "riskCorrelation": -0.8247762861093277,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.6974000902350335,
            "riskCorrelation": -0.8982679323701424,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.7468664107174342,
            "riskCorrelation": -0.9183270795699192,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.3266915853150365,
            "riskCorrelation": -0.6938594162867213,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.3456750800394719,
            "riskCorrelation": -0.6827835015983387,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Apple Intelligence",
        "mentions": 9,
        "sentimentScore": 0.178,
        "riskScore": 0.144,
        "uncertaintyScore": 0.378,
        "qualityScore": 72,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.6276169098123249,
        "riskCorrelation5d": 0.5150634971824255,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9428561965685569,
            "riskCorrelation": 0.8452260670316213,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.8731427817073903,
            "riskCorrelation": 0.7914851307514589,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.6276169098123249,
            "riskCorrelation": 0.5150634971824255,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.520958940728191,
            "riskCorrelation": 0.4673303128222689,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.0792430155579052,
            "riskCorrelation": -0.12256876557809815,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.1538952505791778,
            "riskCorrelation": -0.1795377026354813,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Margins / Profitability",
        "mentions": 5,
        "sentimentScore": 0.44,
        "riskScore": 0.14,
        "uncertaintyScore": 0.28,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.32866524639326833,
        "riskCorrelation5d": -0.21056487615197259,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.6811416904231387,
            "riskCorrelation": -0.5484186885673579,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.4470714966733461,
            "riskCorrelation": -0.35246914941694896,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.32866524639326833,
            "riskCorrelation": -0.21056487615197259,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.1579956026495488,
            "riskCorrelation": -0.05743675726149863,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.5891273584681987,
            "riskCorrelation": 0.6522287773685448,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.6728445480529014,
            "riskCorrelation": 0.6825469819591068,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Supply Chain and Manufacturing",
        "mentions": 5,
        "sentimentScore": -0.3,
        "riskScore": 0.32,
        "uncertaintyScore": 0.62,
        "qualityScore": 46,
        "sentiment": "neg",
        "sentimentCorrelation5d": 0.09362298464336247,
        "riskCorrelation5d": 0.34356486866273583,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.5784253240970846,
            "riskCorrelation": 0.8340226642185357,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.3446819891223033,
            "riskCorrelation": 0.693801373799439,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.09362298464336247,
            "riskCorrelation": 0.34356486866273583,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.14642122791483578,
            "riskCorrelation": 0.28799877791711914,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.6694316032930157,
            "riskCorrelation": -0.3805638861164639,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.7355920885819947,
            "riskCorrelation": -0.4912874005891395,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Services Growth",
        "mentions": 4,
        "sentimentScore": 0.475,
        "riskScore": 0.0,
        "uncertaintyScore": 0.25,
        "qualityScore": 74,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.5091475938196253,
        "riskCorrelation5d": 0.03277997867856526,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.2695609889448231,
            "riskCorrelation": -0.6964478340079083,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.07168443103125997,
            "riskCorrelation": -0.3848760517881898,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.5091475938196253,
            "riskCorrelation": 0.03277997867856526,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.5207295169336571,
            "riskCorrelation": 0.08746824721123922,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.8648320845443198,
            "riskCorrelation": 0.6349288179187058,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.8321930501361214,
            "riskCorrelation": 0.6621769857458285,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Emerging Market Performance",
        "mentions": 4,
        "sentimentScore": 0.6,
        "riskScore": 0.0,
        "uncertaintyScore": 0.15,
        "qualityScore": 77,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.25744768193243106,
        "riskCorrelation5d": -0.03260296440075648,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.2083163739542535,
            "riskCorrelation": -0.6808088915070912,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.11887683315280927,
            "riskCorrelation": -0.30059592981505256,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.25744768193243106,
            "riskCorrelation": -0.03260296440075648,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.14328230548802826,
            "riskCorrelation": 0.03694066315614514,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.17422803822743516,
            "riskCorrelation": 0.46421092731946073,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.1116995553701407,
            "riskCorrelation": 0.5316003682211772,
            "nEvents": 5
          }
        }
      },
      {
        "label": "AI and Machine Learning in Products",
        "mentions": 2,
        "sentimentScore": 0.3,
        "riskScore": 0.1,
        "uncertaintyScore": 0.25,
        "qualityScore": 62,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.9863371438012359,
        "riskCorrelation5d": 0.6600984924472475,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.40613340912241824,
            "riskCorrelation": 0.2600990384705162,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.82443314423002,
            "riskCorrelation": 0.3641499445776721,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.9863371438012359,
            "riskCorrelation": 0.6600984924472475,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.9909530276307638,
            "riskCorrelation": 0.6296951800308686,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.7958552241956475,
            "riskCorrelation": 0.6680334344795004,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.740647336440466,
            "riskCorrelation": 0.5715743994175821,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Mac Performance",
        "mentions": 2,
        "sentimentScore": 0.35,
        "riskScore": 0.1,
        "uncertaintyScore": 0.2,
        "qualityScore": 64,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.5004899198392172,
        "riskCorrelation5d": 0.528626516206073,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.2953414112801283,
            "riskCorrelation": 0.704226748905665,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": -0.3020597601317932,
            "riskCorrelation": 0.6992112902391917,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.5004899198392172,
            "riskCorrelation": 0.528626516206073,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.596281860337597,
            "riskCorrelation": 0.4280055224377957,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.9777754536406214,
            "riskCorrelation": -0.6448843918097678,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.8000095852951351,
            "riskCorrelation": -0.9057453960520477,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Guidance / Outlook",
        "mentions": 2,
        "sentimentScore": 0.15,
        "riskScore": 0.45,
        "uncertaintyScore": 0.55,
        "qualityScore": 49,
        "sentiment": "warn",
        "sentimentCorrelation5d": -0.4829229943066691,
        "riskCorrelation5d": 0.1619028780414156,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.4140968170692928,
            "riskCorrelation": 0.40030325606108624,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.42706164264529906,
            "riskCorrelation": 0.48094537848721813,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.4829229943066691,
            "riskCorrelation": 0.1619028780414156,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.3733427279985038,
            "riskCorrelation": 0.13210112826720447,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.22313736281658056,
            "riskCorrelation": -0.2840019810200229,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.16113479768396735,
            "riskCorrelation": -0.25297764552157287,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Macro / Demand",
        "mentions": 2,
        "sentimentScore": 0.25,
        "riskScore": 0.4,
        "uncertaintyScore": 0.4,
        "qualityScore": 53,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.4226931379613297,
        "riskCorrelation5d": 0.07148125360683129,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.01098916768686949,
            "riskCorrelation": 0.41485737077606094,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.5116272873008528,
            "riskCorrelation": 0.04976407931064012,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.4226931379613297,
            "riskCorrelation": 0.07148125360683129,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.5479266466576436,
            "riskCorrelation": -0.08869862796106782,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.34860427079508366,
            "riskCorrelation": -0.17022999274768158,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.40935142639339084,
            "riskCorrelation": -0.25900567472434544,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Capital Allocation",
        "mentions": 1,
        "sentimentScore": 0.0,
        "riskScore": 0.4,
        "uncertaintyScore": 0.5,
        "qualityScore": 44,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.13758277194920465,
        "riskCorrelation5d": 0.09953116141121703,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.5710759905233393,
            "riskCorrelation": 0.9021571957548664,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.30806785432595113,
            "riskCorrelation": 0.579676183703024,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.13758277194920465,
            "riskCorrelation": 0.09953116141121703,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.1287536813703979,
            "riskCorrelation": 0.02795557688080467,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.6041724882521102,
            "riskCorrelation": -0.6182095281510126,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.5760373392268651,
            "riskCorrelation": -0.6148259361843896,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Other",
        "mentions": 1,
        "sentimentScore": 0.0,
        "riskScore": 0.0,
        "uncertaintyScore": 0.0,
        "qualityScore": 54,
        "sentiment": "neut",
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 5
          }
        }
      },
      {
        "label": "iPad Innovations",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": -1.0,
        "riskCorrelation5d": 0.9999999999999999,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": -0.9999999999999998,
            "riskCorrelation": 0.9999999999999998,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 0.9999999999999999,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 0.9999999999999998,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": -0.9999999999999999,
            "riskCorrelation": 0.9999999999999999,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Regulation / Geopolitics",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.10736172841979404,
        "riskCorrelation5d": -0.36588968006501915,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.7176061922625393,
            "riskCorrelation": -0.07441041639954103,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.14605023092403815,
            "riskCorrelation": -0.5521292434260221,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.10736172841979404,
            "riskCorrelation": -0.36588968006501915,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.08479782173267443,
            "riskCorrelation": -0.4513648389834788,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.018814420700648506,
            "riskCorrelation": -0.322720081520946,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.08635091983527461,
            "riskCorrelation": -0.33562804255351525,
            "nEvents": 4
          }
        }
      }
    ],
    "sentimentHorizon": [
      {
        "horizonDays": 1,
        "rho": -0.087,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 3,
        "rho": -0.404,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 5,
        "rho": -0.774,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 7,
        "rho": -0.77,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 10,
        "rho": -0.895,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 21,
        "rho": -0.825,
        "nEvents": 5,
        "indicativeOnly": true
      }
    ],
    "transcript": {
      "sourceFile": "AAPL_2025_10_30_earnings_call_qa_features.json",
      "date": "2025-10-30",
      "exchanges": [
        {
          "exchangeIdx": 0,
          "speaker": "SPEAKER_00",
          "startSec": 76.289,
          "endSec": 121.48,
          "wordCount": 98,
          "text": "Eric, thanks for your comments. I think it's all about the product. The product lineup is incredibly strong. They're strongest ever. The 17 Pro is the most pro phone we've ever done. It's incredible in the design. The iPhone Air feels so thin and so light in your hand, it feels like it's going to fly away. And then the 17 phone is an incredible value and takes several of the features that were reserved for Pro before and brings them down to the consumer lineup. So overall, strongest iPhone lineup ever, and it's resonating around the world."
        },
        {
          "exchangeIdx": 1,
          "speaker": "SPEAKER_00",
          "startSec": 231.134,
          "endSec": 264.884,
          "wordCount": 73,
          "text": "Yeah, Ben, I was just there. It's incredibly vibrant and dynamic. The store traffic is up significantly year over year. The iPhone 17 has been, the family has been very well received there. We do believe that we'll return to growth in Q1 and that is largely based on the reception of the iPhone there. And so I couldn't be more pleased with with how things are going there in the early going."
        },
        {
          "exchangeIdx": 3,
          "speaker": "SPEAKER_00",
          "startSec": 464.92,
          "endSec": 524.287,
          "wordCount": 118,
          "text": "Okay. We did set a September quarter record for Upgraders, and so it was a great quarter from that point of view. It's really too early in the cycle on 17 to make any comments about Upgraders or Switchers. In terms of channel inventory, we ended the quarter toward the low end of the targeted range, obviously, because we had constraints on several models of the 16 and the 17. And for complete transparency and clarity, we're constrained today on several models of the iPhone 17. There's not a ramp issue. It's just we have very strong demand and we're working very hard to fulfill all the orders that we have. Great. Thank you, Tim. Thank you, Kevin."
        },
        {
          "exchangeIdx": 5,
          "speaker": "SPEAKER_00",
          "startSec": 629.857,
          "endSec": 674.035,
          "wordCount": 95,
          "text": "Yeah, the greater China revenue was down 4% in the year of year in the September quarter. It was driven by iPhone. And if you look at the iPhone, the majority of the sequential year over year change was due to supply constraints that I mentioned earlier. And so it was basically supply constraints that that drove the results. We're thrilled with what we're seeing right now with traffic being up significantly year over year and the reception of the 17 family. We expect to return to growth this quarter. Great. Thank you. Thank you."
        },
        {
          "exchangeIdx": 6,
          "speaker": "SPEAKER_00",
          "startSec": 737.367,
          "endSec": 751.171,
          "wordCount": 23,
          "text": "This is Tim. The advertising category, which is a combination of third party and first party, did set a record during the quarter."
        },
        {
          "exchangeIdx": 7,
          "speaker": "SPEAKER_00",
          "startSec": 764.637,
          "endSec": 779.673,
          "wordCount": 40,
          "text": "I'm sure I'm not saying that. I'm just saying that the combination of the two set a record. We don't, I'm dodging the question intentionally because we don't split it at that level. Okay. Okay. Understood. Thank you. Thank you."
        },
        {
          "exchangeIdx": 8,
          "speaker": "SPEAKER_00",
          "startSec": 819.228,
          "endSec": 866.14,
          "wordCount": 83,
          "text": "Yeah, the subsidies play a favorable role. The subsidies, as you know, are sort of across multiple categories from PCs to tablets to smartwatches and smartphones. And however, it's important to note, they only apply to certain price ranges. And so there's a maximum price and there's several of our products that sell above that price and therefore not eligible for a subsidy. But it does have a favorable effect and it's clearly, at least from our vantage point, driving some consumer demand."
        },
        {
          "exchangeIdx": 9,
          "speaker": "SPEAKER_00",
          "startSec": 989.176,
          "endSec": 1042.45,
          "wordCount": 109,
          "text": "David, I'll take this one. You're right. It goes from 1 1 to a projection of 1 4. And the 1 4 is based on sort of what we know right now and where the tariff rates and policies and and so forth are. So it assumes a. It's stable kind of environment for the quarter. It does comprehend the change that was just made, which we're very encouraged to see with the tariffs moving from 20% to 10% in China. And so that is factored in. And that is one of the reasons why it's not linear to volume, if you will. Does that make sense? Got it."
        },
        {
          "exchangeIdx": 12,
          "speaker": "SPEAKER_00",
          "startSec": 1075.137,
          "endSec": 1138.975,
          "wordCount": 133,
          "text": "We always like to remind people that buy an iPhone all the other things that we offer, and so you can bet that we're doing that. From a Mac point of view, the challenges that last year was sort of the mother of all Mac launches. All of these from Mac Mini to iMac to all the MacBook Pros all launched literally at the same time. And this year that compares to launching the 14 inch MacBook Pro. And so there's a very difficult compare. Of course, in the long run, I'm very bullish on the Mac. And you can see that the Mac, again, last quarter outgrew the market. And, and so we feel really well about how Mac is positioned, but this certain quarter is a extremely difficult compare. And Tim,"
        },
        {
          "exchangeIdx": 15,
          "speaker": "SPEAKER_00",
          "startSec": 1180.066,
          "endSec": 1226.995,
          "wordCount": 82,
          "text": "To be clear, the constraint was not related to manufacturing capacity per se. It was that we called the number of iPhone 16s that we were going to make and were a bit short of where the demand really was. So we could have sold more. We're not publicly at least estimating the extent of that. And then on iPhone 17 family, the demand is very strong. And so we obviously came out of the Q4 timeframe with lots of back orders."
        },
        {
          "exchangeIdx": 17,
          "speaker": "SPEAKER_00",
          "startSec": 1248.022,
          "endSec": 1288.201,
          "wordCount": 72,
          "text": "I think there are opportunities on the app store with artificial intelligence. And so I think, you know, as we have made our on-device models available for developers, and we've seen developers begin to adopt them. And so I think as that proliferates, there's an opportunity for developers and for Apple to benefit from that, from adding features to their apps and so forth. Thanks a lot, Tim. Appreciate it. Thank you."
        },
        {
          "exchangeIdx": 18,
          "speaker": "SPEAKER_00",
          "startSec": 1320.044,
          "endSec": 1345.576,
          "wordCount": 69,
          "text": "It's really too early to call the mix to be honest and we don't like to publicly disclose that because of the competitive for competitive reasons. But frankly, we don't really know what the mix will be yet because we have constraints on both sides of the the ledger at the top and at the entry. And so it will see what happens in as we get more supply."
        },
        {
          "exchangeIdx": 19,
          "speaker": "SPEAKER_00",
          "startSec": 1366.366,
          "endSec": 1405.972,
          "wordCount": 63,
          "text": "Yeah, we're obviously using PCC, our private cloud compute today for a number of queries for Siri. And we will continue to build it out. In fact, the manufacturing plant that makes the servers use for Apple intelligence. just started manufacturing in Houston a few weeks ago. And we've got a ramp plan there for use in our data centers. And it's robust."
        },
        {
          "exchangeIdx": 20,
          "speaker": "SPEAKER_00",
          "startSec": 1453.39,
          "endSec": 1477.977,
          "wordCount": 61,
          "text": "Uh, I'm not sure that there, uh, that one is a proxy for the other. Uh, the thing that I would say is that where we don't get into the model kind of demand. At the aggregate level, we are thrilled with how iPhone has been received. And that's the reason that we're expecting double digit growth in the current quarter."
        },
        {
          "exchangeIdx": 22,
          "speaker": "SPEAKER_00",
          "startSec": 1498.784,
          "endSec": 1540.027,
          "wordCount": 62,
          "text": "We're obviously creating Apple Foundation models within Apple. We ship them on device and use them in the private cloud compute as well. and we've got several in development. And so we also from a continually to surveil the market on M&A and are open to pursuing M&A if we think that it will advance our roadmap. Thank you. Yep. Thank you."
        },
        {
          "exchangeIdx": 23,
          "speaker": "SPEAKER_00",
          "startSec": 1580.746,
          "endSec": 1619.677,
          "wordCount": 75,
          "text": "I think that there are many factors that influence people's purchasing considerations and so, and we don't have a great in-depth survey yet on the current iPhone 17 because it's very new in the cycle and we give it some time to formulate. But I would say that Apple intelligence is a factor. And, you know, we're very bullish on it becoming a greater factor. And so that's the way that we look at it."
        }
      ]
    }
  },
  "CMG": {
    "symbol": "CMG",
    "name": "CHIPOTLE MEXICAN GRILL INC",
    "sector": "consumer-retail",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-29",
    "bias": "BULLISH",
    "finalScore": 11.01,
    "overallScore": 11.01,
    "confidence": 66,
    "probBull": 39,
    "probBear": 61,
    "mlScore": -29.8,
    "decisionInputs": {
      "epsSurprisePct": 0,
      "revenueSurprisePct": 0.59,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 3.51,
      "netMarginPct": 9.81,
      "fcfMarginPct": 15.25,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 0.24,
      "epsEstimate": 0.24,
      "revenueActual": "3.1B",
      "revenueEstimate": "3.1B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "471.0M",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "MCD",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Restaurants & Traffic",
        "sentiment": "neut",
        "description": "Restaurant traffic, pricing, volumes, and consumer staples demand",
        "sharedWith": [
          "MCD",
          "KO"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-03",
        "openReturnPct": -3.71,
        "oneDayReturnPct": 1.94,
        "oneWeekReturnPct": -4.98,
        "oneMonthReturnPct": -9.7
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-29",
        "openReturnPct": -21.35,
        "oneDayReturnPct": -18.18,
        "oneWeekReturnPct": -23.14,
        "oneMonthReturnPct": -13.88
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-23",
        "openReturnPct": -12.47,
        "oneDayReturnPct": -13.34,
        "oneWeekReturnPct": -18.76,
        "oneMonthReturnPct": -17.32
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-23",
        "openReturnPct": 90.24,
        "oneDayReturnPct": 1.6,
        "oneWeekReturnPct": 3.2,
        "oneMonthReturnPct": 3.84
      }
    ],
    "topicDetails": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "MCD",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 59,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 59,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Restaurants & Traffic",
        "sentiment": "neut",
        "description": "Restaurant traffic, pricing, volumes, and consumer staples demand",
        "sharedWith": [
          "MCD",
          "KO"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 59,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "QCOM": {
    "symbol": "QCOM",
    "name": "QUALCOMM INC/DE",
    "sector": "semis-hardware",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-29",
    "bias": "BULLISH",
    "finalScore": 10.82,
    "overallScore": 10.82,
    "confidence": 57,
    "probBull": 53,
    "probBear": 47,
    "mlScore": -2.49,
    "decisionInputs": {
      "epsSurprisePct": 2.63,
      "revenueSurprisePct": 0.08,
      "guidanceRevenueSurprisePct": -5.88,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -13.49,
      "netMarginPct": 69.53,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 4
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.95,
      "epsEstimate": 1.9,
      "revenueActual": "10.6B",
      "revenueEstimate": "10.6B",
      "guidanceRevenueMid": "9.6B",
      "guidanceRevenueConsensus": "10.2B",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Consumer Devices",
        "sentiment": "neut",
        "description": "Hardware cycles, devices, product refreshes, and consumer demand",
        "sharedWith": [
          "AAPL",
          "TSLA"
        ],
        "sharedCount": 3
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "neut",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "AMAT",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "warn",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "JPM",
          "V",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-04",
        "openReturnPct": -10.73,
        "oneDayReturnPct": -8.46,
        "oneWeekReturnPct": -7.0,
        "oneMonthReturnPct": -7.24
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-05",
        "openReturnPct": -1.07,
        "oneDayReturnPct": -3.63,
        "oneWeekReturnPct": -2.9,
        "oneMonthReturnPct": -2.45
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-30",
        "openReturnPct": -3.74,
        "oneDayReturnPct": -7.73,
        "oneWeekReturnPct": -8.27,
        "oneMonthReturnPct": 1.05
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-30",
        "openReturnPct": -6.52,
        "oneDayReturnPct": -8.92,
        "oneWeekReturnPct": -2.26,
        "oneMonthReturnPct": -1.23
      }
    ],
    "topicDetails": [
      {
        "label": "Consumer Devices",
        "sentiment": "neut",
        "description": "Hardware cycles, devices, product refreshes, and consumer demand",
        "sharedWith": [
          "AAPL",
          "TSLA"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 54,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "neut",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "AMAT",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 54,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 54,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "warn",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "TSLA",
          "NVDA",
          "JPM",
          "V",
          "UNH"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 54,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "COST": {
    "symbol": "COST",
    "name": "COSTCO WHOLESALE CORP /NEW",
    "sector": "consumer-retail",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-03-05",
    "bias": "NEUTRAL",
    "finalScore": 8.72,
    "overallScore": 8.72,
    "confidence": 65,
    "probBull": 49,
    "probBear": 51,
    "mlScore": -9.66,
    "decisionInputs": {
      "epsSurprisePct": 0.66,
      "revenueSurprisePct": 0.92,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 3.4,
      "netMarginPct": 2.92,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 4.58,
      "epsEstimate": 4.55,
      "revenueActual": "69.6B",
      "revenueEstimate": "69.0B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "neut",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8
      },
      {
        "label": "Retail Scale",
        "sentiment": "neut",
        "description": "Membership, inventory, traffic, and retail operating scale",
        "sharedWith": [
          "WMT",
          "AMZN"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-11",
        "openReturnPct": -13.34,
        "oneDayReturnPct": -0.11,
        "oneWeekReturnPct": -3.26,
        "oneMonthReturnPct": 7.52
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-25",
        "openReturnPct": -1.88,
        "oneDayReturnPct": -2.9,
        "oneWeekReturnPct": -2.96,
        "oneMonthReturnPct": -1.43
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-06-05",
        "openReturnPct": 97.15,
        "oneDayReturnPct": 40.86,
        "oneWeekReturnPct": -2.04,
        "oneMonthReturnPct": -2.84
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-29",
        "openReturnPct": -63.84,
        "oneDayReturnPct": 3.12,
        "oneWeekReturnPct": 61.46,
        "oneMonthReturnPct": -2.26
      }
    ],
    "topicDetails": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "PG"
        ],
        "sharedCount": 10,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 57,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 57,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "neut",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "MCD",
          "KO",
          "PG"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 57,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Retail Scale",
        "sentiment": "neut",
        "description": "Membership, inventory, traffic, and retail operating scale",
        "sharedWith": [
          "WMT",
          "AMZN"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 57,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "MRK": {
    "symbol": "MRK",
    "name": "Merck & Co., Inc.",
    "sector": "healthcare",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-30",
    "bias": "NEUTRAL",
    "finalScore": 8.05,
    "overallScore": 8.05,
    "confidence": 60,
    "probBull": 46,
    "probBear": 54,
    "mlScore": -15.4,
    "decisionInputs": {
      "epsSurprisePct": 15.23,
      "revenueSurprisePct": 2.75,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -0.7,
      "netMarginPct": -26.03,
      "fcfMarginPct": 17.97,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 3
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": -1.28,
      "epsEstimate": -1.51,
      "revenueActual": "16.3B",
      "revenueEstimate": "15.8B",
      "guidanceRevenueMid": "5.00",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "2.9B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "pos",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "LLY",
          "VRTX",
          "UNH"
        ],
        "sharedCount": 4
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-03",
        "openReturnPct": 91.51,
        "oneDayReturnPct": 2.15,
        "oneWeekReturnPct": 3.0,
        "oneMonthReturnPct": -4.32
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-30",
        "openReturnPct": -1.0,
        "oneDayReturnPct": -34.77,
        "oneWeekReturnPct": 0.0,
        "oneMonthReturnPct": 17.1
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-29",
        "openReturnPct": 33.89,
        "oneDayReturnPct": -1.06,
        "oneWeekReturnPct": -3.86,
        "oneMonthReturnPct": 70.19
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-24",
        "openReturnPct": 41.33,
        "oneDayReturnPct": 3.63,
        "oneWeekReturnPct": 4.18,
        "oneMonthReturnPct": -2.82
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 55,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Healthcare Pipeline",
        "sentiment": "pos",
        "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
        "sharedWith": [
          "LLY",
          "VRTX",
          "UNH"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 55,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "AMAT": {
    "symbol": "AMAT",
    "name": "APPLIED MATERIALS INC /DE",
    "sector": "semis-hardware",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-12",
    "bias": "NEUTRAL",
    "finalScore": 7.5,
    "overallScore": 7.5,
    "confidence": 51,
    "probBull": 39,
    "probBear": 61,
    "mlScore": -29.9,
    "decisionInputs": {
      "epsSurprisePct": 8.68,
      "revenueSurprisePct": 1.89,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 2.38,
      "epsEstimate": 2.19,
      "revenueActual": "7.0B",
      "revenueEstimate": "6.9B",
      "guidanceRevenueMid": "3.00",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Semiconductor Cycle",
        "sentiment": "pos",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "QCOM",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-11-13",
        "openReturnPct": -8.73,
        "oneDayReturnPct": 1.25,
        "oneWeekReturnPct": 34.72,
        "oneMonthReturnPct": 15.95
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-21",
        "openReturnPct": 52.55,
        "oneDayReturnPct": 1.66,
        "oneWeekReturnPct": 57.56,
        "oneMonthReturnPct": 25.67
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-08-14",
        "openReturnPct": -13.57,
        "oneDayReturnPct": -14.07,
        "oneWeekReturnPct": -13.68,
        "oneMonthReturnPct": -7.81
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-22",
        "openReturnPct": -2.42,
        "oneDayReturnPct": -1.88,
        "oneWeekReturnPct": -2.02,
        "oneMonthReturnPct": 14.05
      }
    ],
    "topicDetails": [
      {
        "label": "Semiconductor Cycle",
        "sentiment": "pos",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "QCOM",
          "MU",
          "SMCI"
        ],
        "sharedCount": 7,
        "mentions": null,
        "sentimentScore": 0.45,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 51,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "pos",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 51,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "AMZN": {
    "symbol": "AMZN",
    "name": "AMAZON COM INC",
    "sector": "call-analysis",
    "hasCallAnalysis": true,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-29",
    "bias": "NEUTRAL",
    "finalScore": 5.96,
    "overallScore": 5.96,
    "confidence": 50,
    "probBull": 54,
    "probBear": 46,
    "mlScore": -0.49,
    "decisionInputs": {
      "epsSurprisePct": -2.5,
      "revenueSurprisePct": 2.39,
      "guidanceRevenueSurprisePct": 4.13,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -14.93,
      "netMarginPct": 16.67,
      "fcfMarginPct": -10.01,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.56,
      "epsEstimate": 1.6,
      "revenueActual": "181.5B",
      "revenueEstimate": "177.3B",
      "guidanceRevenueMid": "196.5B",
      "guidanceRevenueConsensus": "188.7B",
      "freeCashFlow": "-18.2B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "callAnalysis": {
      "period": "Q4_2025",
      "callDate": "2026-02-05",
      "turnCount": 16,
      "overall": {
        "sentiment": 68,
        "confidence": 81,
        "risk": 22,
        "uncertainty": 32,
        "defensiveness": 18,
        "analystPressure": 22,
        "guidanceStrength": 56,
        "negativeMixed": 6
      },
      "prepared": {
        "positiveLang": 56,
        "negativeLang": 2,
        "riskLanguage": 14,
        "uncertainty": 24,
        "analystPressure": 12,
        "defensiveLang": 10,
        "guidanceStrength": 56
      },
      "qa": {
        "positiveLang": 19,
        "negativeLang": 2,
        "riskLanguage": 34,
        "uncertainty": 42,
        "analystPressure": 40,
        "defensiveLang": 30,
        "guidanceStrength": 46
      },
      "topics": [
        {
          "label": "Generative AI Initiatives",
          "sentiment": "pos",
          "sentimentScore": 0.4,
          "riskScore": 0.2,
          "negativeMixed": 0
        },
        {
          "label": "AWS Growth And Trends",
          "sentiment": "pos",
          "sentimentScore": 0.6,
          "riskScore": 0.23,
          "negativeMixed": 0
        },
        {
          "label": "Guidance Outlook",
          "sentiment": "neg",
          "sentimentScore": 0.15,
          "riskScore": 0.4,
          "negativeMixed": 0.5
        },
        {
          "label": "E Commerce Store Performance",
          "sentiment": "pos",
          "sentimentScore": 0.4,
          "riskScore": 0.2,
          "negativeMixed": 0
        },
        {
          "label": "Other",
          "sentiment": "neut",
          "sentimentScore": 0,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "Customer Experience Innovations",
          "sentiment": "pos",
          "sentimentScore": 0.7,
          "riskScore": 0.2,
          "negativeMixed": 0
        },
        {
          "label": "Cost Management And Efficiency",
          "sentiment": "warn",
          "sentimentScore": 0,
          "riskScore": 0.3,
          "negativeMixed": 0
        },
        {
          "label": "Capital Allocation",
          "sentiment": "neut",
          "sentimentScore": 0,
          "riskScore": 0.2,
          "negativeMixed": 0
        },
        {
          "label": "AWS",
          "sentiment": "neut",
          "sentimentScore": null,
          "riskScore": null,
          "negativeMixed": null
        },
        {
          "label": "Advertising",
          "sentiment": "neut",
          "sentimentScore": null,
          "riskScore": null,
          "negativeMixed": null
        },
        {
          "label": "AI Infrastructure",
          "sentiment": "neut",
          "description": "AI compute, data centers, networking, and infrastructure demand",
          "sharedWith": [
            "MSFT",
            "NVDA",
            "GOOGL",
            "META",
            "AMD",
            "AVGO",
            "ORCL",
            "SMCI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Generative AI Products",
          "sentiment": "neut",
          "description": "AI applications, copilots, automation, and productized model features",
          "sharedWith": [
            "MSFT",
            "GOOGL",
            "META",
            "PLTR",
            "CRM",
            "NOW",
            "APP",
            "AI"
          ],
          "sharedCount": 9
        },
        {
          "label": "Cloud Growth",
          "sentiment": "neut",
          "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
          "sharedWith": [
            "MSFT",
            "GOOGL",
            "ORCL",
            "NET",
            "DDOG",
            "DOCN",
            "SNOW"
          ],
          "sharedCount": 8
        },
        {
          "label": "Advertising Demand",
          "sentiment": "neut",
          "description": "Ad pricing, conversion, monetization, and marketer demand",
          "sharedWith": [
            "GOOGL",
            "META",
            "APP"
          ],
          "sharedCount": 4
        }
      ],
      "audio": {
        "available": true,
        "source": "Q&A audio features",
        "confidence": 68,
        "vocalStress": 45,
        "instability": 60,
        "paceControl": 61,
        "clarity": 59,
        "segmentCount": 10
      },
      "history": [
        {
          "quarter": "Q1_2024",
          "sentiment": 68,
          "risk": 21,
          "uncertainty": 31,
          "negativeMixed": 11,
          "excessReturn5d": 0.63
        },
        {
          "quarter": "Q2_2024",
          "sentiment": 62,
          "risk": 23,
          "uncertainty": 32,
          "negativeMixed": 16,
          "excessReturn5d": -0.94
        },
        {
          "quarter": "Q4_2024",
          "sentiment": 63,
          "risk": 23,
          "uncertainty": 34,
          "negativeMixed": 18,
          "excessReturn5d": -3.12
        },
        {
          "quarter": "Q3_2025",
          "sentiment": 69,
          "risk": 18,
          "uncertainty": 34,
          "negativeMixed": 12,
          "excessReturn5d": 3.15
        },
        {
          "quarter": "Q4_2025",
          "sentiment": 68,
          "risk": 22,
          "uncertainty": 32,
          "negativeMixed": 6,
          "excessReturn5d": -4.21
        }
      ]
    },
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "neut",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMD",
          "AVGO",
          "ORCL",
          "SMCI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Generative AI Products",
        "sentiment": "neut",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "PLTR",
          "CRM",
          "NOW",
          "APP",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Cloud Growth",
        "sentiment": "neut",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "ORCL",
          "NET",
          "DDOG",
          "DOCN",
          "SNOW"
        ],
        "sharedCount": 8
      },
      {
        "label": "Advertising Demand",
        "sentiment": "neut",
        "description": "Ad pricing, conversion, monetization, and marketer demand",
        "sharedWith": [
          "GOOGL",
          "META",
          "APP"
        ],
        "sharedCount": 4
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AVGO",
          "AMD",
          "SMCI"
        ],
        "sharedCount": 8
      },
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Retail Scale",
        "sentiment": "neut",
        "description": "Membership, inventory, traffic, and retail operating scale",
        "sharedWith": [
          "WMT",
          "COST"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-05",
        "openReturnPct": -8.98,
        "oneDayReturnPct": -5.55,
        "oneWeekReturnPct": -10.73,
        "oneMonthReturnPct": -3.75
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-30",
        "openReturnPct": 12.22,
        "oneDayReturnPct": 9.58,
        "oneWeekReturnPct": 9.67,
        "oneMonthReturnPct": 5.19
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-31",
        "openReturnPct": -7.22,
        "oneDayReturnPct": -8.27,
        "oneWeekReturnPct": -4.88,
        "oneMonthReturnPct": -3.75
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-01",
        "openReturnPct": 64.93,
        "oneDayReturnPct": -11.57,
        "oneWeekReturnPct": 1.5,
        "oneMonthReturnPct": 8.15
      }
    ],
    "topicDetails": [
      {
        "label": "Generative AI Initiatives",
        "mentions": 4,
        "sentimentScore": 0.4,
        "riskScore": 0.2,
        "uncertaintyScore": 0.3,
        "qualityScore": 66,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.08960096024368294,
        "riskCorrelation5d": -0.260310412492396,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.06605476579826797,
            "riskCorrelation": -0.480765288315586,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.059903017565259224,
            "riskCorrelation": -0.2672624589670537,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.08960096024368294,
            "riskCorrelation": -0.260310412492396,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.10434990042107731,
            "riskCorrelation": -0.43696385186655895,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.29649170358903293,
            "riskCorrelation": -0.09061343766977618,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.6281954767357681,
            "riskCorrelation": -0.07519330575705253,
            "nEvents": 5
          }
        }
      },
      {
        "label": "AWS Growth and Trends",
        "mentions": 4,
        "sentimentScore": 0.6,
        "riskScore": 0.225,
        "uncertaintyScore": 0.275,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": 0.023581145214603926,
        "riskCorrelation5d": -0.5561278440828304,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.10069407689140109,
            "riskCorrelation": -0.0640466133739622,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.19044033874759603,
            "riskCorrelation": -0.4402860458061598,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.023581145214603926,
            "riskCorrelation": -0.5561278440828304,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.399707037088223,
            "riskCorrelation": -0.6076551164376143,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.583321347713235,
            "riskCorrelation": -0.8196459828761553,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.16019418015816417,
            "riskCorrelation": -0.22796794235758877,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Guidance / Outlook",
        "mentions": 2,
        "sentimentScore": 0.15,
        "riskScore": 0.4,
        "uncertaintyScore": 0.55,
        "qualityScore": 51,
        "sentiment": "warn",
        "sentimentCorrelation5d": -0.715614957391847,
        "riskCorrelation5d": 0.5362558067570409,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.8019214441650225,
            "riskCorrelation": 0.8392314817781797,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.6904144164948279,
            "riskCorrelation": 0.5815275431798899,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.715614957391847,
            "riskCorrelation": 0.5362558067570409,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.8138253837773939,
            "riskCorrelation": 0.517399927681395,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.25845397656783403,
            "riskCorrelation": -0.21898520672652244,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.1824113417402716,
            "riskCorrelation": -0.45269449640559695,
            "nEvents": 5
          }
        }
      },
      {
        "label": "E-commerce Store Performance",
        "mentions": 2,
        "sentimentScore": 0.4,
        "riskScore": 0.2,
        "uncertaintyScore": 0.3,
        "qualityScore": 62,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.3736355010882629,
        "riskCorrelation5d": -0.09928363936797868,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.19969511112757146,
            "riskCorrelation": 0.1325090294320523,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.2027297808565372,
            "riskCorrelation": -0.24678509056301048,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.3736355010882629,
            "riskCorrelation": -0.09928363936797868,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": -0.7526419998957565,
            "riskCorrelation": 0.2840727940772529,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.8180524983229924,
            "riskCorrelation": 0.13962246133461145,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.36029033521219317,
            "riskCorrelation": -0.041674164323800784,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Capital Allocation",
        "mentions": 1,
        "sentimentScore": 0.0,
        "riskScore": 0.2,
        "uncertaintyScore": 0.5,
        "qualityScore": 50,
        "sentiment": "neut",
        "sentimentCorrelation5d": 1.0,
        "riskCorrelation5d": 0.9999999999999998,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": 0.9999999999999999,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": 0.9999999999999998,
            "riskCorrelation": 0.9999999999999998,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": 0.9999999999999998,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Cost Management and Efficiency",
        "mentions": 1,
        "sentimentScore": 0.0,
        "riskScore": 0.3,
        "uncertaintyScore": 0.5,
        "qualityScore": 47,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.7644394743353264,
        "riskCorrelation5d": -0.9944692357215893,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.4634644419801261,
            "riskCorrelation": -0.9091922473314377,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": 0.8200292533710308,
            "riskCorrelation": -0.9925711223548205,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": 0.7644394743353264,
            "riskCorrelation": -0.9944692357215893,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": 0.5347862585296811,
            "riskCorrelation": -0.8536680661662128,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": 0.33393925659074286,
            "riskCorrelation": 0.1900092690116873,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": -0.09037055974117403,
            "riskCorrelation": 0.6558348321087295,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Customer Experience Innovations",
        "mentions": 1,
        "sentimentScore": 0.7,
        "riskScore": 0.2,
        "uncertaintyScore": 0.2,
        "qualityScore": 69,
        "sentiment": "pos",
        "horizons": {}
      },
      {
        "label": "Other",
        "mentions": 1,
        "sentimentScore": 0.0,
        "riskScore": 0.0,
        "uncertaintyScore": 0.0,
        "qualityScore": 54,
        "sentiment": "neut",
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": null,
            "riskCorrelation": null,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Delivery Speed and Logistics",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.9999999999999999,
        "riskCorrelation5d": 1.0,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9999999999999998,
            "riskCorrelation": 0.9999999999999998,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": -0.9999999999999999,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": -0.9999999999999999,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": -0.9999999999999999,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Third-Party Seller Dynamics",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      },
      {
        "label": "Macro / Demand",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      },
      {
        "label": "Margins / Profitability",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      }
    ],
    "sentimentHorizon": [
      {
        "horizonDays": 1,
        "rho": 0.548,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 3,
        "rho": 0.271,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 5,
        "rho": 0.426,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 7,
        "rho": 0.707,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 10,
        "rho": 0.284,
        "nEvents": 5,
        "indicativeOnly": true
      },
      {
        "horizonDays": 21,
        "rho": -0.298,
        "nEvents": 5,
        "indicativeOnly": true
      }
    ],
    "transcript": {
      "sourceFile": "AMZN_2025_10_30_earnings_call_qa_features.json",
      "date": "2025-10-30",
      "exchanges": [
        {
          "exchangeIdx": 0,
          "speaker": "SPEAKER_04",
          "startSec": 27.183,
          "endSec": 77.015,
          "wordCount": 135,
          "text": "Yeah, on the capacity side, we brought in quite a bit of capacity, as I mentioned in my opening comments, 3.8 gigawatts capacity in the last year with another gigawatt plus coming in the fourth quarter. And we expect to double our overall capacity by the end of 2027. So we're bringing in quite a bit of capacity. Today, overall in the industry, maybe the bottleneck is power. I think at some point it may move to chips, but we're bringing in quite a bit of capacity. And as fast as we're bringing it in right now, we are monetizing it. And then on the training demand outside of our major customers. So first of all, as I mentioned on training two, it's really doing well. It's fully subscribed on training two. We have, you"
        },
        {
          "exchangeIdx": 1,
          "speaker": "SPEAKER_04",
          "startSec": 77.234,
          "endSec": 148.514,
          "wordCount": 185,
          "text": "a multi-billion dollar business at this point. It grew 150% quarter over quarter in revenue. And you see really big projects of scale now, like our project Rainier that we're doing with Anthropic, where they're running their next version of, they're training the next version of Claude on top of Trainium II on 500,000 Trainium II chips, going to a million Trainium II chips by the end of the year. As I mentioned, we have today with Trainium II, we have a small number of very large customers on it. But because Tranium is 30 to 40% more price performance than other options out there, and because as customers, as they start to contemplate broader scale of their production workloads moving to being AI focused and using inference, they badly care about price performance. And so we have a lot of demand for Tranium. Trainium 3 should preview at the end of this year with much fuller volumes coming in the beginning of 26. We have a lot of customers, both very large and I'll call it medium size who are quite interested in Trainium"
        },
        {
          "exchangeIdx": 2,
          "speaker": "SPEAKER_04",
          "startSec": 198.667,
          "endSec": 325.229,
          "wordCount": 369,
          "text": "Yeah. Well, first of all, we're always going to have multiple chip options for our customers. It's been true in every major technology building block or component that we've had in AWS. Really in the history of AWS, it's never just one player that over a long period of time has the entire market segment and can satisfy everybody's needs. on every dimension. And so we have a very deep relationship with NVIDIA. We have for a very long time, and we will for as long as I can foresee the future. We buy a lot of NVIDIA. We are not constrained in any way in buying NVIDIA. And I expect that we'll continue to buy more NVIDIA both next year and in the future. But we're different from most technology companies in that we have our own very strong chip team. And this is our ANAPERNA team. And you saw it first on the CPU side with what we built with Graviton, which is about 40% better price performance than the other X86 processors. And you're seeing it again on the custom silicon on the AI side with Tranium, which is about the same amount of price performance benefit for customers relative to other GPU options. And our customers to be able to use AI as expansively as they want. And remember, it's still relatively early days at this point. they're going to need better price performance and they care about it deeply. And so, you know, I mentioned earlier the momentum that Tranium 2 has. And I think that for us, as we think about Tranium 3, I expect Tranium 3 will be about 40% better than Tranium 2. And Tranium 2 is already very advantage on price performance. So we have to, of course, deliver the chip. We have to deliver it in volumes and deliver it quickly. And we have to continue to work on the software ecosystem, which gets better all the time. And as we have more proof points like we have with Project Rainier, with what Anthropics is doing on Tranium 2, it builds increasing credibility for Tranium. And I think customers are very bullish about it. I'm bullish about it as"
        },
        {
          "exchangeIdx": 3,
          "speaker": "SPEAKER_04",
          "startSec": 361.763,
          "endSec": 424.302,
          "wordCount": 171,
          "text": "Yeah, I think what is compelling for Anthropic around Project Rainier is, you know, really is the Tranium II chip, which, you know, we've built a very, first of all, we built a very large cluster that they can use in a very expansive way. And it's not simple to be able to build a cluster that has 500,000 plus chips going to a million. That's an infrastructure feat that's hard to do at scale. You know, some piece of it is the infrastructure capabilities that we've built over a long period of time in AWS that is, you know, unusual in the industry, but it's just also the performance of the chip and the price performance, both of which matter. And, you know, I think that, you know, Project Rainier is something that is specific for Anthropic, but we have a lot of other customers who are interested in employing large clusters of Tranium chips that we're going to hopefully give them a chance to do so with Tranium 3."
        },
        {
          "exchangeIdx": 4,
          "speaker": "SPEAKER_04",
          "startSec": 500.898,
          "endSec": 720.796,
          "wordCount": 618,
          "text": "Yeah, so I'll start with grocery mark. We have a very large grocery business. If you look at our entire grocery business, if I don't even count Whole Foods Market and Fresh, in the last 12 months, it's over $100 billion of gross merchandising sales, which would make us a top three grocery in the US. A good chunk of it is a lot of the items that you'd find in the middle aisles, so consumables and canned goods and pet food and health and beauty, very significant that continues to grow at a very good clip. But then, you know, we also have Whole Foods Market, which is the pioneer in organic foods, which is also growing at a faster clip than most grocery companies and with an attractive trajectory on profitability. And we'll expand our Whole Foods physical presence over the coming years here. And I'm also very excited about this new concept daily shop that we have, which is a smaller version of Whole Foods and Urban Settings, which we have three that we've launched that are off to very good starts that you should expect to see more of as well. And we have always been, you know, as you reference, we've talked a lot about having a larger mass physical presence. And we continue to experiment with various formats, but the one that we are most excited about is what you reference, which is the ability to provide perishable groceries with same-day deliveries. And if you think about how many of our customers are buying from us multiple times a week and who are buying things like shampoo or detergent or paper cups or water, where the ability to add milk and eggs and yogurt and other perishables to their order and have it live in the same shopping cart and then show up a few hours later is very compelling. And we started with a few markets about a year ago, and we were really taking it back at the adoption. Not just the number of people that started buying perishables from us very quickly, but how often they came back downstream to buy perishables and groceries from us in the future. And so we've now expanded that to 1,000 cities around the US. We'll be in 2,300 by the end of the year. And it's really changing the trajectory and the size of our grocery business. And I also believe that this many years tradition of the weekly stock up, grocery stock up is changing. And I think we're a big part of that. And I think there's a lot of potential there for the grocery side. It doesn't mean that we won't continue to experiment with other physical formats, but we're onto something very significant with what we're doing with perishables from our same day facilities. And then on your head cow question, you know, what I would tell you is, you know, the, The announcement that we made a few days ago was not really financially driven. And it's not even really AI driven, not right now at least. It really, it's culture. And if you grow as fast as we did for several years, you know, the size of businesses, the number of people, the number of locations, the types of businesses you're in, you end up with a lot more people than what you had before. and you end up with a lot more layers. And when that happens, sometimes without realizing it, you can weaken the ownership of the people that you have who are doing the actual work and who own most of the two-way door decisions, the ones that should be made quickly and"
        },
        {
          "exchangeIdx": 5,
          "speaker": "SPEAKER_04",
          "startSec": 721.117,
          "endSec": 753.871,
          "wordCount": 101,
          "text": "right at the front line. And it can lead to slowing you down. And as a leadership team, we are committed to operating like the world's largest startup. And that means removing layers. It means increasing the amount of ownership that people have. And it means inventing and moving quickly. And I don't know if there's ever been a time in the history of Amazon or maybe business in general with the technology transformation happening right now where it's important to be lean, it's important to be flat, and it's important to move fast. And that's what we're going to do"
        },
        {
          "exchangeIdx": 6,
          "speaker": "SPEAKER_04",
          "startSec": 790.287,
          "endSec": 854.817,
          "wordCount": 187,
          "text": "Robotics is a very substantial area of investment for us. We have over a million robots in our fulfillment network at this point. And I would say that while that's significant, we have a lot of invention in flight. So I expect that we'll have more over a period of time. You know, robotics are very important for us and for our customers and for our teammates because they improve safety, they boost productivity, they increase speed, and they let our human teammates focus on problem solving and what they do best. And we expect that our people will remain at the heart and the center of our fulfillment network as they have from when we first started working on robotics. And we expect that over time we will have a fulfillment network where robots and humans complement each other and work together. But I think you're going to continue to see us invest very significantly in robotics. It's going to help on the safety, the productivity, the speed, and ultimately some of the cost pieces, which will allow us to continue to improve the customer experience."
        },
        {
          "exchangeIdx": 7,
          "speaker": "SPEAKER_04",
          "startSec": 879.337,
          "endSec": 889.563,
          "wordCount": 31,
          "text": "I'm very excited about, and as a business, we're very excited about in the long term, the prospect of agentic commerce. And it has a chance to be good for customers."
        },
        {
          "exchangeIdx": 8,
          "speaker": "SPEAKER_04",
          "startSec": 890.018,
          "endSec": 1054.229,
          "wordCount": 476,
          "text": "It has a chance to be really good for e-commerce. And I think if you're If you know what you want to buy, there are a few experiences that are better than coming to Amazon. But if you don't know what you want, it's, you know, a physical store with a physical salesperson still has some advantages. Obviously, lots of people do it on Amazon all the time, but you very often want to ask questions and help, you know, get help narrowing what you're going to look for. And as you keep asking new questions, having a whole bunch of different options presented to you, And I think AI and agentic commerce are going to change the experience online where that experience where you're narrowing what you want when you don't know is going to get better online than it even is in physical environments. Now, we obviously have our own efforts here in agentic commerce. We have Rufus, which I talked about in my opening comments, which is continuing to get better and better and use more broadly. And we have features like Bye For Me where We will surface on Amazon even items that we don't stock that other merchants have. And then if customers want us to go and buy it for them on those merchants' websites, we will do that. And both of those have been successful for us. But we're also having conversations with and expect over time to partner with third-party agents. You know, I think that it reminds me in some ways of the beginning of search engines many years ago being sources of discovery for commerce. And, you know, you had to kind of figure out the right way to work together. And today, search engines are a very small part of our referral traffic. And third-party agents are a very small subset of that. But I do think that we will find ways to partner. We have to find a way, though, that makes the customer experience good. Right now, I would say the customer experience is not good. There's no personalization. There's no shopping history. The delivery estimates are frequently wrong. The prices are often wrong. So we've got to find a way to make the customer experience better and have the right exchange of value. But I do think that the exciting part of this and the promise is that AI and agentic commerce solutions are going to expand the amount of shopping that happens online. And I think that's really good for customers, and I think it's really good for Amazon because at the end of the day, you're going to buy from the outfit that allows you to have the broadest selection, great value, and continues to deliver for you very quickly and reliably. And I think that bodes well for us."
        },
        {
          "exchangeIdx": 9,
          "speaker": "SPEAKER_04",
          "startSec": 1103.082,
          "endSec": 1477.977,
          "wordCount": 978,
          "text": "I'll start on the AWS side. You know, we are seeing, you know, you know, we're really pleased with the results from this quarter, 20% year-over-year on a annualized run rate of $132 billion is unusual. And we have momentum. You can see it. And we see the growth in both our AI area, where we see it in inference. We see it in training. We see it in the use of our Tranium Custom Silicon. You know, bedrock continues to grow really quickly. SageMaker continues to grow quickly. And, you know, I think that the number of companies who are working on building agents is very significant. I do believe that a lot of the value that companies will realize over time in AI will come from agents. And I think that Building agents today is still harder than it should be. You need tools to make it easier, which is why we built strands, which is an open source capability that lets people build agents from any model that they can imagine. But even more so when you talk to enterprises or companies that care a lot about security and scale, They're starting to build agents and they don't really feel like they've got, they've had building blocks that allow them to have the type of secure scalable agents that they need to bet their businesses and their customer experience and their data on. And that's why, that was really the inspiration behind Agent Core was to build another set of primitive building blocks like we built in the early days of AWS where it was compute and storage and database We defined a set of building blocks that you needed to be able to deploy agents securely and scalably that we provide an agent core. And then when we talk to our customers, it really resonates. There is not anything else like it. It's changing their timeframe and their receptivity to building agents. And it's very compelling for them. So I do think the combination of what we're doing to enable agents to be built and run securely and scalably as well as some of the agents that we're building ourselves that our customers are excited about are compelling for them. And I think the other place we see a lot of growth in AWS also is just the number of enterprises who have gotten back to moving from on-premises infrastructure to the cloud. And we continue to earn the lion's share of those transformations. You know, I look at the momentum we have right now, and I believe that we can continue to grow at a clip like this for a while. You know, I think on the advertising side, you know, that is also an area where I think collectively we feel very pleased about the progress. Every single one of our advertising offerings this quarter grew in a meaningful way. I think there's a few things going on for us. What I think of as a pretty unusual full funnel offering. And if you look at the top of the funnel, which typically tends to be awareness building and broad scale, to be able to use our own prime video and our live sports capabilities, as well as going all the way down to the bottom of the funnel at point of sale, being able to use sponsored products, That's, you know, most people don't have a full funnel offering as robust as that. And then when you layer on top of it, the combination of the audience curation and development we can do, along with the advantage measurement, it just all leads to a return on advertising spend that's very unusual. And I think there are multiple places where we can expect to continue to grow. You know, one is, In our stores business, I still think if you look at the worldwide market segment share of retail, still 80 to 85% of it lives in physical stores. And that equation is going to flip over time. And I think AI is going to only accelerate that. So I think we have a significant opportunity still in our existing stores. And then I think video, we've only been at this for a little bit of time, but it's already a very large amount of advertising revenue. and we're still relatively early stage. I think that will continue to be a big area of growth. And then as you reference the amount, the demand side platform or Amazon DSP, that is growing really quickly as well. Some of it had to do with the fact that we had some features. We always had a number of the core components people wanted around some of our properties, the measurement capabilities, Amazon Marketing Cloud, We lacked some features for a while as we were building out our DSP that customers told us mattered. And the team over the last 20 months have closed those gaps in a very significant way so that now people feel like our DSP is fully featured. And then you look at some of the partnerships that we've done. The Roku partnership gives us the largest connected TV footprint in the US. And you layer on top of that, what we've recently done in providing our DSP customers the opportunity to integrate with the ad inventory in Netflix, and Spotify, and SiriusXM, it's powerful. And so we are growing very quickly on the demand side platform. So very optimistic about what we're doing there. We've continued work to do, obviously, but I don't think we're close to being able to grow that Thanks for joining us on the call today and for your questions. A replay will be available on our Investor Relations website for at least three months. We appreciate your interest in Amazon and look forward to speaking"
        }
      ]
    }
  },
  "PG": {
    "symbol": "PG",
    "name": "PROCTER & GAMBLE Co",
    "sector": "consumer-retail",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-24",
    "bias": "NEUTRAL",
    "finalScore": 4.07,
    "overallScore": 4.07,
    "confidence": 54,
    "probBull": 48,
    "probBear": 52,
    "mlScore": -10.83,
    "decisionInputs": {
      "epsSurprisePct": 1.92,
      "revenueSurprisePct": -1.32,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -4.38,
      "netMarginPct": 18.52,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 3
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 1.59,
      "epsEstimate": 1.56,
      "revenueActual": "21.2B",
      "revenueEstimate": "21.5B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "COST"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "neut",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "KO"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-22",
        "openReturnPct": 60.69,
        "oneDayReturnPct": 14.67,
        "oneWeekReturnPct": 1.23,
        "oneMonthReturnPct": 10.24
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-24",
        "openReturnPct": -57.71,
        "oneDayReturnPct": -49.18,
        "oneWeekReturnPct": -2.93,
        "oneMonthReturnPct": -2.62
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-29",
        "openReturnPct": 24.9,
        "oneDayReturnPct": -2.38,
        "oneWeekReturnPct": -2.39,
        "oneMonthReturnPct": -61.3
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-24",
        "openReturnPct": 21.94,
        "oneDayReturnPct": 93.4,
        "oneWeekReturnPct": 62.06,
        "oneMonthReturnPct": 5.16
      }
    ],
    "topicDetails": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "MCD",
          "WMT",
          "COST"
        ],
        "sharedCount": 10,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 51,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 51,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "neut",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "MCD",
          "KO"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.14,
        "uncertaintyScore": 0.18,
        "qualityScore": 51,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "MCD": {
    "symbol": "MCD",
    "name": "MCDONALDS CORP",
    "sector": "consumer-retail",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-07",
    "bias": "NEUTRAL",
    "finalScore": 1.75,
    "overallScore": 1.75,
    "confidence": 58,
    "probBull": 42,
    "probBear": 58,
    "mlScore": -23.1,
    "decisionInputs": {
      "epsSurprisePct": 3.28,
      "revenueSurprisePct": 0.73,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -7.01,
      "netMarginPct": 30.43,
      "fcfMarginPct": 26.55,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "up"
      },
      {
        "period": "1D",
        "direction": "up"
      },
      {
        "period": "1W",
        "direction": "up"
      },
      {
        "period": "1M",
        "direction": "up"
      }
    ],
    "fundamentals": {
      "epsActual": 2.83,
      "epsEstimate": 2.74,
      "revenueActual": "6.5B",
      "revenueEstimate": "6.5B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "1.7B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "warn",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "warn",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "KO",
          "PG"
        ],
        "sharedCount": 8
      },
      {
        "label": "Restaurants & Traffic",
        "sentiment": "warn",
        "description": "Restaurant traffic, pricing, volumes, and consumer staples demand",
        "sharedWith": [
          "CMG",
          "KO"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-11",
        "openReturnPct": 2.05,
        "oneDayReturnPct": 2.74,
        "oneWeekReturnPct": 1.86,
        "oneMonthReturnPct": 1.06
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-05",
        "openReturnPct": -46.13,
        "oneDayReturnPct": -2.38,
        "oneWeekReturnPct": 62.49,
        "oneMonthReturnPct": 1.35
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-06",
        "openReturnPct": 48.76,
        "oneDayReturnPct": 8.13,
        "oneWeekReturnPct": 41.93,
        "oneMonthReturnPct": 1.67
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-01",
        "openReturnPct": -2.87,
        "oneDayReturnPct": -53.56,
        "oneWeekReturnPct": 1.28,
        "oneMonthReturnPct": -39.85
      }
    ],
    "topicDetails": [
      {
        "label": "Margins & Cost Control",
        "sentiment": "warn",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "TSLA",
          "CMG",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Payments & Consumer Spend",
        "sentiment": "warn",
        "description": "Consumer spending, transaction volume, credit, and retail activity",
        "sharedWith": [
          "V",
          "JPM",
          "XYZ",
          "WMT",
          "COST",
          "KO",
          "PG"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Restaurants & Traffic",
        "sentiment": "warn",
        "description": "Restaurant traffic, pricing, volumes, and consumer staples demand",
        "sharedWith": [
          "CMG",
          "KO"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "NOW": {
    "symbol": "NOW",
    "name": "ServiceNow, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-04-22",
    "bias": "NEUTRAL",
    "finalScore": -1.31,
    "overallScore": -1.31,
    "confidence": 48,
    "probBull": 45,
    "probBear": 55,
    "mlScore": -17.73,
    "decisionInputs": {
      "epsSurprisePct": -10.91,
      "revenueSurprisePct": 0.53,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": 5.66,
      "netMarginPct": 12.44,
      "fcfMarginPct": 40.56,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "down"
      },
      {
        "period": "1D",
        "direction": "down"
      },
      {
        "period": "1W",
        "direction": "down"
      },
      {
        "period": "1M",
        "direction": "down"
      }
    ],
    "fundamentals": {
      "epsActual": 0.49,
      "epsEstimate": 0.55,
      "revenueActual": "3.8B",
      "revenueEstimate": "3.8B",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "1.5B",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Generative AI Products",
        "sentiment": "neg",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "APP",
          "AI"
        ],
        "sharedCount": 9
      },
      {
        "label": "Developer Platforms",
        "sentiment": "neg",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "SNOW",
          "PLTR",
          "CRM"
        ],
        "sharedCount": 6
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "neg",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-01-28",
        "openReturnPct": -8.56,
        "oneDayReturnPct": -9.94,
        "oneWeekReturnPct": -20.82,
        "oneMonthReturnPct": -15.58
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-10-29",
        "openReturnPct": 1.65,
        "oneDayReturnPct": 2.52,
        "oneWeekReturnPct": -5.81,
        "oneMonthReturnPct": -9.83
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-07-23",
        "openReturnPct": 8.18,
        "oneDayReturnPct": 4.16,
        "oneWeekReturnPct": -1.39,
        "oneMonthReturnPct": -7.29
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-04-23",
        "openReturnPct": 10.78,
        "oneDayReturnPct": 15.49,
        "oneWeekReturnPct": 17.87,
        "oneMonthReturnPct": 23.58
      }
    ],
    "topicDetails": [
      {
        "label": "Generative AI Products",
        "sentiment": "neg",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "APP",
          "AI"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 47,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Developer Platforms",
        "sentiment": "neg",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "SNOW",
          "PLTR",
          "CRM"
        ],
        "sharedCount": 6,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 47,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 47,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "neg",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 47,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "SMCI": {
    "symbol": "SMCI",
    "name": "Super Micro Computer, Inc.",
    "sector": "semis-hardware",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-05",
    "bias": "NEUTRAL",
    "finalScore": -1.38,
    "overallScore": -1.38,
    "confidence": 58,
    "probBull": 47,
    "probBear": 53,
    "mlScore": -12.91,
    "decisionInputs": {
      "epsSurprisePct": 23.64,
      "revenueSurprisePct": -17.33,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -19.23,
      "netMarginPct": 4.72,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "down"
      },
      {
        "period": "1D",
        "direction": "down"
      },
      {
        "period": "1W",
        "direction": "down"
      },
      {
        "period": "1M",
        "direction": "down"
      }
    ],
    "fundamentals": {
      "epsActual": 0.68,
      "epsEstimate": 0.55,
      "revenueActual": "10.2B",
      "revenueEstimate": "12.4B",
      "guidanceRevenueMid": "11.8B",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "AVGO",
          "ORCL"
        ],
        "sharedCount": 9
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "warn",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "QCOM",
          "AMAT",
          "MU"
        ],
        "sharedCount": 7
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "AMD"
        ],
        "sharedCount": 8
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-03",
        "openReturnPct": 11.05,
        "oneDayReturnPct": 13.78,
        "oneWeekReturnPct": 7.99,
        "oneMonthReturnPct": 5.53
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-04",
        "openReturnPct": -5.12,
        "oneDayReturnPct": -11.33,
        "oneWeekReturnPct": -20.02,
        "oneMonthReturnPct": -26.81
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-05",
        "openReturnPct": -17.34,
        "oneDayReturnPct": -18.29,
        "oneWeekReturnPct": -19.54,
        "oneMonthReturnPct": -29.43
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-06",
        "openReturnPct": -5.59,
        "oneDayReturnPct": -1.4,
        "oneWeekReturnPct": 36.61,
        "oneMonthReturnPct": 26.14
      }
    ],
    "topicDetails": [
      {
        "label": "AI Infrastructure",
        "sentiment": "warn",
        "description": "AI compute, data centers, networking, and infrastructure demand",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AMD",
          "AVGO",
          "ORCL"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Semiconductor Cycle",
        "sentiment": "warn",
        "description": "Chip demand, supply, memory, equipment, and hardware cycle",
        "sharedWith": [
          "NVDA",
          "AMD",
          "AVGO",
          "QCOM",
          "AMAT",
          "MU"
        ],
        "sharedCount": 7,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Data Center Capex",
        "sentiment": "warn",
        "description": "Capex intensity, server buildout, and capacity constraints",
        "sharedWith": [
          "MSFT",
          "NVDA",
          "GOOGL",
          "META",
          "AMZN",
          "AVGO",
          "AMD"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SNOW",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.28,
        "qualityScore": 52,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "SNOW": {
    "symbol": "SNOW",
    "name": "Snowflake Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-25",
    "bias": "NEUTRAL",
    "finalScore": -4.93,
    "overallScore": -4.93,
    "confidence": 50,
    "probBull": 50,
    "probBear": 50,
    "mlScore": -6.92,
    "decisionInputs": {
      "epsSurprisePct": -23.08,
      "revenueSurprisePct": 2.4,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "down"
      },
      {
        "period": "1D",
        "direction": "down"
      },
      {
        "period": "1W",
        "direction": "down"
      },
      {
        "period": "1M",
        "direction": "down"
      }
    ],
    "fundamentals": {
      "epsActual": -0.8,
      "epsEstimate": -0.65,
      "revenueActual": "1.3B",
      "revenueEstimate": "1.2B",
      "guidanceRevenueMid": "1.00",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Cloud Growth",
        "sentiment": "neg",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "NET",
          "DDOG",
          "DOCN"
        ],
        "sharedCount": 8
      },
      {
        "label": "Developer Platforms",
        "sentiment": "neg",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "PLTR",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "neg",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-03",
        "openReturnPct": -7.92,
        "oneDayReturnPct": -11.41,
        "oneWeekReturnPct": -16.79,
        "oneMonthReturnPct": -11.5
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-08-27",
        "openReturnPct": 10.78,
        "oneDayReturnPct": 20.27,
        "oneWeekReturnPct": 12.57,
        "oneMonthReturnPct": 12.41
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-30",
        "openReturnPct": 0.0,
        "oneDayReturnPct": 2.19,
        "oneWeekReturnPct": 2.52,
        "oneMonthReturnPct": 5.67
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-21",
        "openReturnPct": 7.19,
        "oneDayReturnPct": 13.43,
        "oneWeekReturnPct": 14.82,
        "oneMonthReturnPct": 24.58
      }
    ],
    "topicDetails": [
      {
        "label": "Cloud Growth",
        "sentiment": "neg",
        "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
        "sharedWith": [
          "MSFT",
          "AMZN",
          "GOOGL",
          "ORCL",
          "NET",
          "DDOG",
          "DOCN"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 49,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Developer Platforms",
        "sentiment": "neg",
        "description": "Enterprise software usage, data workflows, and developer productivity",
        "sharedWith": [
          "GTLB",
          "DDOG",
          "PLTR",
          "CRM",
          "NOW"
        ],
        "sharedCount": 6,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 49,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "ABNB",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 49,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "neg",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD",
          "ZS"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 49,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "ABNB": {
    "symbol": "ABNB",
    "name": "Airbnb, Inc.",
    "sector": "consumer-retail",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-05-07",
    "bias": "BEARISH",
    "finalScore": -15.35,
    "overallScore": -15.35,
    "confidence": 68,
    "probBull": 52,
    "probBear": 48,
    "mlScore": -3.55,
    "decisionInputs": {
      "epsSurprisePct": -16.13,
      "revenueSurprisePct": 2.21,
      "guidanceRevenueSurprisePct": 1.43,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": -3.6,
      "netMarginPct": 5.97,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 1
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "down"
      },
      {
        "period": "1D",
        "direction": "down"
      },
      {
        "period": "1W",
        "direction": "down"
      },
      {
        "period": "1M",
        "direction": "down"
      }
    ],
    "fundamentals": {
      "epsActual": 0.26,
      "epsEstimate": 0.31,
      "revenueActual": "2.7B",
      "revenueEstimate": "2.6B",
      "guidanceRevenueMid": "3.5B",
      "guidanceRevenueConsensus": "3.5B",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ZS",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Travel & Marketplace",
        "sentiment": "warn",
        "description": "Marketplace liquidity, take rate, bookings, and platform engagement",
        "sharedWith": [
          "XYZ",
          "APP"
        ],
        "sharedCount": 3
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2026-02-12",
        "openReturnPct": 9.06,
        "oneDayReturnPct": 4.65,
        "oneWeekReturnPct": 6.04,
        "oneMonthReturnPct": 13.79
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-11-06",
        "openReturnPct": 4.12,
        "oneDayReturnPct": 29.04,
        "oneWeekReturnPct": 1.24,
        "oneMonthReturnPct": 4.03
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-08-06",
        "openReturnPct": -7.75,
        "oneDayReturnPct": -8.02,
        "oneWeekReturnPct": -4.49,
        "oneMonthReturnPct": -4.51
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-01",
        "openReturnPct": -2.35,
        "oneDayReturnPct": 1.01,
        "oneWeekReturnPct": 2.44,
        "oneMonthReturnPct": 7.17
      }
    ],
    "topicDetails": [
      {
        "label": "Guidance Quality",
        "sentiment": "warn",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ZS",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 61,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Travel & Marketplace",
        "sentiment": "warn",
        "description": "Marketplace liquidity, take rate, bookings, and platform engagement",
        "sharedWith": [
          "XYZ",
          "APP"
        ],
        "sharedCount": 3,
        "mentions": null,
        "sentimentScore": 0.1,
        "riskScore": 0.22,
        "uncertaintyScore": 0.28,
        "qualityScore": 61,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "ZS": {
    "symbol": "ZS",
    "name": "Zscaler, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-26",
    "bias": "BEARISH",
    "finalScore": -22.27,
    "overallScore": -22.27,
    "confidence": 66,
    "probBull": 45,
    "probBear": 55,
    "mlScore": -17.8,
    "decisionInputs": {
      "epsSurprisePct": -60,
      "revenueSurprisePct": 2.11,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 2
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "down"
      },
      {
        "period": "1D",
        "direction": "down"
      },
      {
        "period": "1W",
        "direction": "down"
      },
      {
        "period": "1M",
        "direction": "down"
      }
    ],
    "fundamentals": {
      "epsActual": -0.08,
      "epsEstimate": -0.05,
      "revenueActual": "815.8M",
      "revenueEstimate": "798.9M",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Cybersecurity Demand",
        "sentiment": "neg",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "PANW",
          "CRWD",
          "NET"
        ],
        "sharedCount": 4
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "AI"
        ],
        "sharedCount": 40
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "neg",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD"
        ],
        "sharedCount": 8
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-11-25",
        "openReturnPct": -6.78,
        "oneDayReturnPct": -13.03,
        "oneWeekReturnPct": -16.53,
        "oneMonthReturnPct": -21.3
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-02",
        "openReturnPct": -75.39,
        "oneDayReturnPct": -1.45,
        "oneWeekReturnPct": 1.54,
        "oneMonthReturnPct": 12.02
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-05-29",
        "openReturnPct": 4.67,
        "oneDayReturnPct": 9.79,
        "oneWeekReturnPct": 20.68,
        "oneMonthReturnPct": 22.38
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-29",
        "openReturnPct": 4.67,
        "oneDayReturnPct": 9.79,
        "oneWeekReturnPct": 20.68,
        "oneMonthReturnPct": 22.38
      }
    ],
    "topicDetails": [
      {
        "label": "Cybersecurity Demand",
        "sentiment": "neg",
        "description": "Security spending, platform consolidation, and threat environment",
        "sharedWith": [
          "PANW",
          "CRWD",
          "NET"
        ],
        "sharedCount": 4,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 62,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "AI"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": 0,
        "riskScore": 0.32,
        "uncertaintyScore": 0.18,
        "qualityScore": 62,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Enterprise Sales Efficiency",
        "sentiment": "neg",
        "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
        "sharedWith": [
          "CRM",
          "NOW",
          "SNOW",
          "DDOG",
          "GTLB",
          "PANW",
          "CRWD"
        ],
        "sharedCount": 8,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 62,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "AI": {
    "symbol": "AI",
    "name": "C3.ai, Inc.",
    "sector": "software-cloud",
    "hasCallAnalysis": false,
    "latestQuarter": "CY2026Q1",
    "reportDate": "2026-02-25",
    "bias": "STRONG BEARISH",
    "finalScore": -72.18,
    "overallScore": -72.18,
    "confidence": 92,
    "probBull": 39,
    "probBear": 61,
    "mlScore": -29.25,
    "decisionInputs": {
      "epsSurprisePct": -28.77,
      "revenueSurprisePct": -29.84,
      "guidanceRevenueSurprisePct": -35.57,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 5,
      "quartersWithGuidance": 4
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "down"
      },
      {
        "period": "1D",
        "direction": "down"
      },
      {
        "period": "1W",
        "direction": "down"
      },
      {
        "period": "1M",
        "direction": "down"
      }
    ],
    "fundamentals": {
      "epsActual": -0.94,
      "epsEstimate": -0.73,
      "revenueActual": "53.3M",
      "revenueEstimate": "75.9M",
      "guidanceRevenueMid": "50.0M",
      "guidanceRevenueConsensus": "77.6M",
      "freeCashFlow": "n/a",
      "dataSource": "SEC companyfacts"
    },
    "notes": "",
    "topics": [
      {
        "label": "Generative AI Products",
        "sentiment": "neg",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "NOW",
          "APP"
        ],
        "sharedCount": 9
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS"
        ],
        "sharedCount": 40
      }
    ],
    "pastReactions": [
      {
        "quarter": "CY2025Q4",
        "reportDate": "2025-12-03",
        "openReturnPct": -39.97,
        "oneDayReturnPct": 2.07,
        "oneWeekReturnPct": 6.06,
        "oneMonthReturnPct": -5.73
      },
      {
        "quarter": "CY2025Q3",
        "reportDate": "2025-09-03",
        "openReturnPct": -10.37,
        "oneDayReturnPct": -7.31,
        "oneWeekReturnPct": -1.68,
        "oneMonthReturnPct": 14.87
      },
      {
        "quarter": "CY2025Q2",
        "reportDate": "2025-05-28",
        "openReturnPct": 15.51,
        "oneDayReturnPct": 20.76,
        "oneWeekReturnPct": 10.08,
        "oneMonthReturnPct": 6.73
      },
      {
        "quarter": "CY2025Q1",
        "reportDate": "2025-02-26",
        "openReturnPct": null,
        "oneDayReturnPct": null,
        "oneWeekReturnPct": null,
        "oneMonthReturnPct": null
      }
    ],
    "topicDetails": [
      {
        "label": "Generative AI Products",
        "sentiment": "neg",
        "description": "AI applications, copilots, automation, and productized model features",
        "sharedWith": [
          "MSFT",
          "GOOGL",
          "META",
          "AMZN",
          "PLTR",
          "CRM",
          "NOW",
          "APP"
        ],
        "sharedCount": 9,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 92,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      },
      {
        "label": "Guidance Quality",
        "sentiment": "neg",
        "description": "Management guidance availability and surprise versus consensus",
        "sharedWith": [
          "MU",
          "PANW",
          "LLY",
          "PLTR",
          "GTLB",
          "DDOG",
          "GOOGL",
          "DOCN",
          "V",
          "JPM",
          "CRWD",
          "NET",
          "APP",
          "ORCL",
          "AMD",
          "XYZ",
          "META",
          "KO",
          "AVGO",
          "VRTX",
          "CRM",
          "MSFT",
          "NVDA",
          "WMT",
          "UNH",
          "AAPL",
          "CMG",
          "QCOM",
          "COST",
          "MRK",
          "AMAT",
          "AMZN",
          "PG",
          "MCD",
          "NOW",
          "SMCI",
          "SNOW",
          "ABNB",
          "ZS"
        ],
        "sharedCount": 40,
        "mentions": null,
        "sentimentScore": -0.2,
        "riskScore": 0.38,
        "uncertaintyScore": 0.18,
        "qualityScore": 92,
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": null,
        "nEvents": null,
        "horizons": {}
      }
    ],
    "sentimentHorizon": []
  },
  "TSLA": {
    "symbol": "TSLA",
    "name": "Tesla Inc.",
    "sector": "call-analysis",
    "hasCallAnalysis": true,
    "latestQuarter": "Q4_2025",
    "reportDate": "2026-01-28",
    "bias": "NEUTRAL",
    "finalScore": null,
    "overallScore": null,
    "confidence": 50,
    "probBull": 50,
    "probBear": 50,
    "mlScore": null,
    "decisionInputs": {
      "epsSurprisePct": null,
      "revenueSurprisePct": null,
      "guidanceRevenueSurprisePct": null,
      "guidanceEpsSurprisePct": null,
      "revenueGrowthPct": null,
      "netMarginPct": null,
      "fcfMarginPct": null,
      "quartersWithConsensus": 0,
      "quartersWithGuidance": 0
    },
    "horizons": [
      {
        "period": "Open",
        "direction": "flat"
      },
      {
        "period": "1D",
        "direction": "flat"
      },
      {
        "period": "1W",
        "direction": "flat"
      },
      {
        "period": "1M",
        "direction": "flat"
      }
    ],
    "fundamentals": {
      "epsActual": null,
      "epsEstimate": null,
      "revenueActual": "n/a",
      "revenueEstimate": "n/a",
      "guidanceRevenueMid": "n/a",
      "guidanceRevenueConsensus": "n/a",
      "freeCashFlow": "n/a",
      "dataSource": "call panel only"
    },
    "notes": "Call-analysis panel available; current bias/fundamental signal row was not present in the broader earnings-stat export.",
    "callAnalysis": {
      "period": "Q4_2025",
      "callDate": "2026-01-28",
      "turnCount": 57,
      "overall": {
        "sentiment": 63,
        "confidence": 76,
        "risk": 15,
        "uncertainty": 29,
        "defensiveness": 13,
        "analystPressure": 12,
        "guidanceStrength": 27,
        "negativeMixed": 7
      },
      "prepared": {
        "positiveLang": 22,
        "negativeLang": 7,
        "riskLanguage": 7,
        "uncertainty": 21,
        "analystPressure": 2,
        "defensiveLang": 5,
        "guidanceStrength": 27
      },
      "qa": {
        "positiveLang": 12,
        "negativeLang": 2,
        "riskLanguage": 27,
        "uncertainty": 39,
        "analystPressure": 30,
        "defensiveLang": 25,
        "guidanceStrength": 17
      },
      "topics": [
        {
          "label": "Robotaxi Service",
          "sentiment": "pos",
          "sentimentScore": 0.38,
          "riskScore": 0.09,
          "negativeMixed": 0
        },
        {
          "label": "Other",
          "sentiment": "neut",
          "sentimentScore": 0.09,
          "riskScore": 0,
          "negativeMixed": 0
        },
        {
          "label": "AI And Compute Infrastructure",
          "sentiment": "neut",
          "sentimentScore": 0.15,
          "riskScore": 0.19,
          "negativeMixed": 0.12
        },
        {
          "label": "Optimus Robot Development",
          "sentiment": "neut",
          "sentimentScore": 0.26,
          "riskScore": 0.19,
          "negativeMixed": 0
        },
        {
          "label": "Autonomy And Full Self Driving FSD",
          "sentiment": "warn",
          "sentimentScore": 0.51,
          "riskScore": 0.27,
          "negativeMixed": 0
        },
        {
          "label": "Model Lineup And Production Updates",
          "sentiment": "neut",
          "sentimentScore": 0.2,
          "riskScore": 0.1,
          "negativeMixed": 0
        },
        {
          "label": "Cost Management And Efficiency Initiatives",
          "sentiment": "warn",
          "sentimentScore": 0.27,
          "riskScore": 0.2,
          "negativeMixed": 0.33
        },
        {
          "label": "Global Expansion And Market Penetration",
          "sentiment": "pos",
          "sentimentScore": 0.6,
          "riskScore": 0.1,
          "negativeMixed": 0
        },
        {
          "label": "Capital Allocation",
          "sentiment": "neut",
          "sentimentScore": 0.15,
          "riskScore": 0.2,
          "negativeMixed": 0
        },
        {
          "label": "Regulation Geopolitics",
          "sentiment": "neg",
          "sentimentScore": -0.3,
          "riskScore": 0.7,
          "negativeMixed": 1
        },
        {
          "label": "Consumer Devices",
          "sentiment": "neut",
          "description": "Hardware cycles, devices, product refreshes, and consumer demand",
          "sharedWith": [
            "AAPL",
            "QCOM"
          ],
          "sharedCount": 3
        },
        {
          "label": "Margins & Cost Control",
          "sentiment": "neut",
          "description": "Operating leverage, efficiency, pricing, and cost discipline",
          "sharedWith": [
            "AMZN",
            "META",
            "MSFT",
            "AAPL",
            "CMG",
            "MCD",
            "WMT",
            "COST",
            "PG"
          ],
          "sharedCount": 10
        },
        {
          "label": "Regulation & Geopolitics",
          "sentiment": "warn",
          "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
          "sharedWith": [
            "GOOGL",
            "META",
            "AAPL",
            "NVDA",
            "QCOM",
            "JPM",
            "V",
            "UNH"
          ],
          "sharedCount": 9
        }
      ],
      "audio": {
        "available": false,
        "source": "No extracted audio feature file in this snapshot",
        "confidence": null,
        "vocalStress": null,
        "instability": null,
        "paceControl": null,
        "clarity": null,
        "segmentCount": 0
      },
      "history": [
        {
          "quarter": "Q1_2024",
          "sentiment": 66,
          "risk": 14,
          "uncertainty": 29,
          "negativeMixed": 14,
          "excessReturn5d": 12.19
        },
        {
          "quarter": "Q3_2024",
          "sentiment": 68,
          "risk": 10,
          "uncertainty": 24,
          "negativeMixed": 7,
          "excessReturn5d": -2.36
        },
        {
          "quarter": "Q4_2024",
          "sentiment": 59,
          "risk": 24,
          "uncertainty": 35,
          "negativeMixed": 21,
          "excessReturn5d": -7.74
        },
        {
          "quarter": "Q2_2025",
          "sentiment": 60,
          "risk": 16,
          "uncertainty": 31,
          "negativeMixed": 13,
          "excessReturn5d": 0.97
        },
        {
          "quarter": "Q3_2025",
          "sentiment": 66,
          "risk": 15,
          "uncertainty": 27,
          "negativeMixed": 9,
          "excessReturn5d": -4.51
        },
        {
          "quarter": "Q4_2025",
          "sentiment": 63,
          "risk": 15,
          "uncertainty": 29,
          "negativeMixed": 7,
          "excessReturn5d": 0.5
        }
      ]
    },
    "topics": [
      {
        "label": "Consumer Devices",
        "sentiment": "neut",
        "description": "Hardware cycles, devices, product refreshes, and consumer demand",
        "sharedWith": [
          "AAPL",
          "QCOM"
        ],
        "sharedCount": 3
      },
      {
        "label": "Margins & Cost Control",
        "sentiment": "neut",
        "description": "Operating leverage, efficiency, pricing, and cost discipline",
        "sharedWith": [
          "AMZN",
          "META",
          "MSFT",
          "AAPL",
          "CMG",
          "MCD",
          "WMT",
          "COST",
          "PG"
        ],
        "sharedCount": 10
      },
      {
        "label": "Regulation & Geopolitics",
        "sentiment": "warn",
        "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
        "sharedWith": [
          "GOOGL",
          "META",
          "AAPL",
          "NVDA",
          "QCOM",
          "JPM",
          "V",
          "UNH"
        ],
        "sharedCount": 9
      }
    ],
    "topicDetails": [
      {
        "label": "Robotaxi Service",
        "mentions": 11,
        "sentimentScore": 0.382,
        "riskScore": 0.091,
        "uncertaintyScore": 0.282,
        "qualityScore": 81,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.2929111928680096,
        "riskCorrelation5d": -0.5081582605722547,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.1568080217872669,
            "riskCorrelation": -0.7612047810954368,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": -0.26100869379966185,
            "riskCorrelation": -0.32496933064893485,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": -0.2929111928680096,
            "riskCorrelation": -0.5081582605722547,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": -0.5957528012382836,
            "riskCorrelation": -0.4279755488128292,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": -0.3901719966084305,
            "riskCorrelation": -0.816569476793953,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": -0.08439475306870903,
            "riskCorrelation": -0.7419911199713404,
            "nEvents": 6
          }
        }
      },
      {
        "label": "Other",
        "mentions": 9,
        "sentimentScore": 0.089,
        "riskScore": 0.0,
        "uncertaintyScore": 0.0,
        "qualityScore": 73,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.677698538996753,
        "riskCorrelation5d": null,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.8358931918855942,
            "riskCorrelation": null,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": 0.6634458950339148,
            "riskCorrelation": null,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": 0.677698538996753,
            "riskCorrelation": null,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": 0.25342991781348334,
            "riskCorrelation": null,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": 0.22139117565900665,
            "riskCorrelation": null,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": 0.21551716720698358,
            "riskCorrelation": null,
            "nEvents": 6
          }
        }
      },
      {
        "label": "AI and Compute Infrastructure",
        "mentions": 8,
        "sentimentScore": 0.15,
        "riskScore": 0.188,
        "uncertaintyScore": 0.412,
        "qualityScore": 68,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.3601419725627751,
        "riskCorrelation5d": -0.7329358566726036,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.011418172129112435,
            "riskCorrelation": -0.362885339800237,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": 0.4803067021704162,
            "riskCorrelation": -0.6021689937143809,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": 0.3601419725627751,
            "riskCorrelation": -0.7329358566726036,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": 0.05607941699754552,
            "riskCorrelation": -0.649770856510508,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": -0.3252046433136466,
            "riskCorrelation": -0.7986079983113883,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": -0.28760139579314825,
            "riskCorrelation": -0.732337896787431,
            "nEvents": 6
          }
        }
      },
      {
        "label": "Optimus Robot Development",
        "mentions": 7,
        "sentimentScore": 0.257,
        "riskScore": 0.186,
        "uncertaintyScore": 0.286,
        "qualityScore": 69,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.44916289126004755,
        "riskCorrelation5d": 0.5552847351003413,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.41441114741680013,
            "riskCorrelation": -0.24578468440373913,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.5651920491905809,
            "riskCorrelation": 0.5267042865906071,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.44916289126004755,
            "riskCorrelation": 0.5552847351003413,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.029956087102237437,
            "riskCorrelation": 0.7562450318055391,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": -0.28209301024344274,
            "riskCorrelation": 0.5201242520783463,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": -0.3595231392023448,
            "riskCorrelation": 0.32737104859452987,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Autonomy and Full Self-Driving (FSD)",
        "mentions": 7,
        "sentimentScore": 0.514,
        "riskScore": 0.271,
        "uncertaintyScore": 0.314,
        "qualityScore": 74,
        "sentiment": "warn",
        "sentimentCorrelation5d": 0.3090465165916811,
        "riskCorrelation5d": -0.3184639948034451,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.026897303045433293,
            "riskCorrelation": 0.34150726466870285,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": 0.11762160765395313,
            "riskCorrelation": -0.2376019193792396,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": 0.3090465165916811,
            "riskCorrelation": -0.3184639948034451,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": 0.5517296805180358,
            "riskCorrelation": -0.43966245396965653,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": 0.7678974382637781,
            "riskCorrelation": -0.4721178773287731,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": 0.5664876583989831,
            "riskCorrelation": -0.4244298834990194,
            "nEvents": 6
          }
        }
      },
      {
        "label": "Cost Management and Efficiency Initiatives",
        "mentions": 3,
        "sentimentScore": 0.267,
        "riskScore": 0.2,
        "uncertaintyScore": 0.3,
        "qualityScore": 61,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.8546025234435167,
        "riskCorrelation5d": 0.5105466457782789,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9996522341575734,
            "riskCorrelation": 0.8701260942069672,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": -0.888651904030273,
            "riskCorrelation": 0.5691177831011881,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": -0.8546025234435167,
            "riskCorrelation": 0.5105466457782789,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": -0.9854141398228943,
            "riskCorrelation": 0.7900105625239642,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": 0.9565776280135123,
            "riskCorrelation": -0.981396152240933,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": 0.9460162863758025,
            "riskCorrelation": -0.9874044512440411,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Model Lineup and Production Updates",
        "mentions": 3,
        "sentimentScore": 0.2,
        "riskScore": 0.1,
        "uncertaintyScore": 0.4,
        "qualityScore": 62,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.17837835480689387,
        "riskCorrelation5d": -0.2272891882923381,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.5294918661236258,
            "riskCorrelation": -0.4817741419256927,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": 0.08583234365750587,
            "riskCorrelation": -0.0020466137516527093,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": 0.17837835480689387,
            "riskCorrelation": -0.2272891882923381,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": 0.5356538797183769,
            "riskCorrelation": -0.17937290631840874,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": 0.44225213152627635,
            "riskCorrelation": -0.8025272507501773,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": 0.21582306464924403,
            "riskCorrelation": -0.8518198380763708,
            "nEvents": 6
          }
        }
      },
      {
        "label": "Capital Allocation",
        "mentions": 2,
        "sentimentScore": 0.15,
        "riskScore": 0.2,
        "uncertaintyScore": 0.5,
        "qualityScore": 56,
        "sentiment": "neut",
        "sentimentCorrelation5d": -0.8217580262944645,
        "riskCorrelation5d": -0.24121466352522603,
        "nEvents": 4,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.5004907789904741,
            "riskCorrelation": 0.44126321683239156,
            "nEvents": 4
          },
          "3": {
            "sentimentCorrelation": -0.8144059262472704,
            "riskCorrelation": -0.3126381462234857,
            "nEvents": 4
          },
          "5": {
            "sentimentCorrelation": -0.8217580262944645,
            "riskCorrelation": -0.24121466352522603,
            "nEvents": 4
          },
          "7": {
            "sentimentCorrelation": -0.9233823481188423,
            "riskCorrelation": -0.3615627273784253,
            "nEvents": 4
          },
          "10": {
            "sentimentCorrelation": -0.2911501250170907,
            "riskCorrelation": 0.5354716311247144,
            "nEvents": 4
          },
          "21": {
            "sentimentCorrelation": 0.18380584597088617,
            "riskCorrelation": 0.40527544146608857,
            "nEvents": 4
          }
        }
      },
      {
        "label": "Global Expansion and Market Penetration",
        "mentions": 2,
        "sentimentScore": 0.6,
        "riskScore": 0.1,
        "uncertaintyScore": 0.2,
        "qualityScore": 70,
        "sentiment": "pos",
        "sentimentCorrelation5d": null,
        "riskCorrelation5d": 0.9999999999999998,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": null,
            "riskCorrelation": 0.9999999999999999,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": null,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": null,
            "riskCorrelation": 0.9999999999999998,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": null,
            "riskCorrelation": 0.9999999999999999,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": null,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": null,
            "riskCorrelation": 1.0,
            "nEvents": 2
          }
        }
      },
      {
        "label": "4680 Battery Cell Production",
        "mentions": 1,
        "sentimentScore": -0.3,
        "riskScore": 0.7,
        "uncertaintyScore": 0.6,
        "qualityScore": 29,
        "sentiment": "neg",
        "sentimentCorrelation5d": 0.9996774097963113,
        "riskCorrelation5d": -0.8222062729750133,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.9050520067111432,
            "riskCorrelation": -0.5238658956497736,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.991607937032947,
            "riskCorrelation": -0.9002406085728976,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.9996774097963113,
            "riskCorrelation": -0.8222062729750133,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": 0.9404419897797935,
            "riskCorrelation": -0.6002465092692908,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": 0.7938990810747635,
            "riskCorrelation": -0.3307290735400457,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": 0.809338164220444,
            "riskCorrelation": -0.3549920545953137,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Regulation / Geopolitics",
        "mentions": 1,
        "sentimentScore": -0.3,
        "riskScore": 0.7,
        "uncertaintyScore": 0.6,
        "qualityScore": 29,
        "sentiment": "neg",
        "sentimentCorrelation5d": 0.8884298115942572,
        "riskCorrelation5d": 0.0466984251287867,
        "nEvents": 3,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.9860124451760752,
            "riskCorrelation": 0.6373481550554356,
            "nEvents": 3
          },
          "3": {
            "sentimentCorrelation": 0.8625068574748376,
            "riskCorrelation": -0.006994720838435293,
            "nEvents": 3
          },
          "5": {
            "sentimentCorrelation": 0.8884298115942572,
            "riskCorrelation": 0.0466984251287867,
            "nEvents": 3
          },
          "7": {
            "sentimentCorrelation": 0.9509962174637873,
            "riskCorrelation": 0.20772087710609433,
            "nEvents": 3
          },
          "10": {
            "sentimentCorrelation": -0.9953548356540507,
            "riskCorrelation": -0.5810534536830984,
            "nEvents": 3
          },
          "21": {
            "sentimentCorrelation": -0.7910431984091891,
            "riskCorrelation": -0.925321555497458,
            "nEvents": 3
          }
        }
      },
      {
        "label": "Energy Storage Solutions",
        "mentions": 1,
        "sentimentScore": 0.6,
        "riskScore": 0.0,
        "uncertaintyScore": 0.3,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.08468917795694293,
        "riskCorrelation5d": -0.01736490854649483,
        "nEvents": 6,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.1729564775884691,
            "riskCorrelation": -0.05968013252616237,
            "nEvents": 6
          },
          "3": {
            "sentimentCorrelation": -0.095292934427199,
            "riskCorrelation": -0.03487404038855682,
            "nEvents": 6
          },
          "5": {
            "sentimentCorrelation": -0.08468917795694293,
            "riskCorrelation": -0.01736490854649483,
            "nEvents": 6
          },
          "7": {
            "sentimentCorrelation": -0.3462153072702991,
            "riskCorrelation": 0.13045414277885814,
            "nEvents": 6
          },
          "10": {
            "sentimentCorrelation": -0.1954887075902697,
            "riskCorrelation": 0.23067238611255603,
            "nEvents": 6
          },
          "21": {
            "sentimentCorrelation": -0.06498199576724434,
            "riskCorrelation": 0.25461546986635325,
            "nEvents": 6
          }
        }
      },
      {
        "label": "Market Demand and Sales Strategies",
        "mentions": 1,
        "sentimentScore": 0.6,
        "riskScore": 0.0,
        "uncertaintyScore": 0.4,
        "qualityScore": 71,
        "sentiment": "pos",
        "sentimentCorrelation5d": -0.06635000138602574,
        "riskCorrelation5d": 0.17908720104957798,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": 0.23388421675567986,
            "riskCorrelation": -0.0652190909695918,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": -0.28223823068825954,
            "riskCorrelation": 0.2763711295070504,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": -0.06635000138602574,
            "riskCorrelation": 0.17908720104957798,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.09515409802321684,
            "riskCorrelation": 0.1584379906243478,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.6230729264927344,
            "riskCorrelation": -0.09849598265123988,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.6169831408084658,
            "riskCorrelation": -0.10179589149764849,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Margins / Profitability",
        "mentions": 1,
        "sentimentScore": 0.0,
        "riskScore": 0.0,
        "uncertaintyScore": 0.5,
        "qualityScore": 54,
        "sentiment": "neut",
        "horizons": {}
      },
      {
        "label": "Macro / Demand",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": -1.0,
        "riskCorrelation5d": -0.9999999999999998,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.9999999999999998,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": -1.0,
            "riskCorrelation": -0.9999999999999999,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": 1.0,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": 0.9999999999999999,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Guidance / Outlook",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 1.0,
        "riskCorrelation5d": -0.9999999999999998,
        "nEvents": 2,
        "horizons": {
          "1": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "3": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "5": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "7": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -0.9999999999999998,
            "nEvents": 2
          },
          "10": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          },
          "21": {
            "sentimentCorrelation": 1.0,
            "riskCorrelation": -1.0,
            "nEvents": 2
          }
        }
      },
      {
        "label": "Regulatory Challenges",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "sentimentCorrelation5d": 0.6501579105402963,
        "riskCorrelation5d": -0.7855949141074102,
        "nEvents": 5,
        "horizons": {
          "1": {
            "sentimentCorrelation": -0.06587494329971179,
            "riskCorrelation": -0.3279974460582593,
            "nEvents": 5
          },
          "3": {
            "sentimentCorrelation": 0.5787228251170338,
            "riskCorrelation": -0.7363402430024316,
            "nEvents": 5
          },
          "5": {
            "sentimentCorrelation": 0.6501579105402963,
            "riskCorrelation": -0.7855949141074102,
            "nEvents": 5
          },
          "7": {
            "sentimentCorrelation": 0.8271833492206538,
            "riskCorrelation": -0.69215526415071,
            "nEvents": 5
          },
          "10": {
            "sentimentCorrelation": 0.5317023969142128,
            "riskCorrelation": -0.4329385690457335,
            "nEvents": 5
          },
          "21": {
            "sentimentCorrelation": 0.2022839934448849,
            "riskCorrelation": -0.1973413012475483,
            "nEvents": 5
          }
        }
      },
      {
        "label": "Manufacturing",
        "mentions": 0,
        "sentimentScore": null,
        "riskScore": null,
        "uncertaintyScore": null,
        "qualityScore": null,
        "sentiment": "neut",
        "horizons": {}
      }
    ],
    "sentimentHorizon": [
      {
        "horizonDays": 1,
        "rho": -0.092,
        "nEvents": 6,
        "indicativeOnly": false
      },
      {
        "horizonDays": 3,
        "rho": 0.143,
        "nEvents": 6,
        "indicativeOnly": false
      },
      {
        "horizonDays": 5,
        "rho": 0.305,
        "nEvents": 6,
        "indicativeOnly": false
      },
      {
        "horizonDays": 7,
        "rho": 0.366,
        "nEvents": 6,
        "indicativeOnly": false
      },
      {
        "horizonDays": 10,
        "rho": 0.612,
        "nEvents": 6,
        "indicativeOnly": false
      },
      {
        "horizonDays": 21,
        "rho": 0.555,
        "nEvents": 6,
        "indicativeOnly": false
      }
    ]
  }
};

window.SHARED_TOPICS = [
  {
    "label": "AI Infrastructure",
    "description": "AI compute, data centers, networking, and infrastructure demand",
    "tickers": [
      "MSFT",
      "NVDA",
      "GOOGL",
      "META",
      "AMZN",
      "AMD",
      "AVGO",
      "ORCL",
      "SMCI"
    ]
  },
  {
    "label": "Generative AI Products",
    "description": "AI applications, copilots, automation, and productized model features",
    "tickers": [
      "MSFT",
      "GOOGL",
      "META",
      "AMZN",
      "PLTR",
      "CRM",
      "NOW",
      "APP",
      "AI"
    ]
  },
  {
    "label": "Cloud Growth",
    "description": "Cloud revenue, usage growth, infrastructure services, and enterprise adoption",
    "tickers": [
      "MSFT",
      "AMZN",
      "GOOGL",
      "ORCL",
      "NET",
      "DDOG",
      "DOCN",
      "SNOW"
    ]
  },
  {
    "label": "Cybersecurity Demand",
    "description": "Security spending, platform consolidation, and threat environment",
    "tickers": [
      "PANW",
      "CRWD",
      "ZS",
      "NET"
    ]
  },
  {
    "label": "Developer Platforms",
    "description": "Enterprise software usage, data workflows, and developer productivity",
    "tickers": [
      "GTLB",
      "DDOG",
      "SNOW",
      "PLTR",
      "CRM",
      "NOW"
    ]
  },
  {
    "label": "Advertising Demand",
    "description": "Ad pricing, conversion, monetization, and marketer demand",
    "tickers": [
      "GOOGL",
      "META",
      "AMZN",
      "APP"
    ]
  },
  {
    "label": "Consumer Devices",
    "description": "Hardware cycles, devices, product refreshes, and consumer demand",
    "tickers": [
      "AAPL",
      "TSLA",
      "QCOM"
    ]
  },
  {
    "label": "Semiconductor Cycle",
    "description": "Chip demand, supply, memory, equipment, and hardware cycle",
    "tickers": [
      "NVDA",
      "AMD",
      "AVGO",
      "QCOM",
      "AMAT",
      "MU",
      "SMCI"
    ]
  },
  {
    "label": "Data Center Capex",
    "description": "Capex intensity, server buildout, and capacity constraints",
    "tickers": [
      "MSFT",
      "NVDA",
      "GOOGL",
      "META",
      "AMZN",
      "AVGO",
      "AMD",
      "SMCI"
    ]
  },
  {
    "label": "Margins & Cost Control",
    "description": "Operating leverage, efficiency, pricing, and cost discipline",
    "tickers": [
      "AMZN",
      "META",
      "MSFT",
      "AAPL",
      "TSLA",
      "CMG",
      "MCD",
      "WMT",
      "COST",
      "PG"
    ]
  },
  {
    "label": "Guidance Quality",
    "description": "Management guidance availability and surprise versus consensus",
    "tickers": [
      "MU",
      "PANW",
      "LLY",
      "PLTR",
      "GTLB",
      "DDOG",
      "GOOGL",
      "DOCN",
      "V",
      "JPM",
      "CRWD",
      "NET",
      "APP",
      "ORCL",
      "AMD",
      "XYZ",
      "META",
      "KO",
      "AVGO",
      "VRTX",
      "CRM",
      "MSFT",
      "NVDA",
      "WMT",
      "UNH",
      "AAPL",
      "CMG",
      "QCOM",
      "COST",
      "MRK",
      "AMAT",
      "AMZN",
      "PG",
      "MCD",
      "NOW",
      "SMCI",
      "SNOW",
      "ABNB",
      "ZS",
      "AI"
    ]
  },
  {
    "label": "Payments & Consumer Spend",
    "description": "Consumer spending, transaction volume, credit, and retail activity",
    "tickers": [
      "V",
      "JPM",
      "XYZ",
      "WMT",
      "COST",
      "MCD",
      "KO",
      "PG"
    ]
  },
  {
    "label": "Healthcare Pipeline",
    "description": "Drug pipeline, utilization, approvals, reimbursement, and healthcare demand",
    "tickers": [
      "LLY",
      "VRTX",
      "MRK",
      "UNH"
    ]
  },
  {
    "label": "Restaurants & Traffic",
    "description": "Restaurant traffic, pricing, volumes, and consumer staples demand",
    "tickers": [
      "CMG",
      "MCD",
      "KO"
    ]
  },
  {
    "label": "Retail Scale",
    "description": "Membership, inventory, traffic, and retail operating scale",
    "tickers": [
      "WMT",
      "COST",
      "AMZN"
    ]
  },
  {
    "label": "Travel & Marketplace",
    "description": "Marketplace liquidity, take rate, bookings, and platform engagement",
    "tickers": [
      "ABNB",
      "XYZ",
      "APP"
    ]
  },
  {
    "label": "Enterprise Sales Efficiency",
    "description": "Enterprise seat expansion, sales cycles, retention, and large deal activity",
    "tickers": [
      "CRM",
      "NOW",
      "SNOW",
      "DDOG",
      "GTLB",
      "PANW",
      "CRWD",
      "ZS"
    ]
  },
  {
    "label": "Regulation & Geopolitics",
    "description": "Regulatory pressure, antitrust, export controls, and policy exposure",
    "tickers": [
      "GOOGL",
      "META",
      "AAPL",
      "TSLA",
      "NVDA",
      "QCOM",
      "JPM",
      "V",
      "UNH"
    ]
  }
];

window.BACKTEST = {
  "total_tests": 195,
  "directional_tests": 194,
  "directional_hits": 101,
  "directional_accuracy": 52.06,
  "avg_one_week_return_pct": 0.16,
  "by_prediction": [
    {
      "direction": "bullish",
      "count": 167,
      "directional_accuracy": 51.81,
      "avg_return_pct": 0.14,
      "avg_one_week_return_pct": 0.14,
      "avg_close_return_pct": -0.13
    },
    {
      "direction": "neutral",
      "count": 0,
      "directional_accuracy": null,
      "avg_return_pct": "",
      "avg_one_week_return_pct": "",
      "avg_close_return_pct": ""
    },
    {
      "direction": "bearish",
      "count": 28,
      "directional_accuracy": 53.57,
      "avg_return_pct": 0.24,
      "avg_one_week_return_pct": 0.24,
      "avg_close_return_pct": -0.65
    }
  ],
  "prediction_vs_actual": [
    {
      "predicted": "bullish",
      "total": 167,
      "actual_bullish": 86,
      "actual_neutral": 1,
      "actual_bearish": 80,
      "bullish_pct": 51.5,
      "neutral_pct": 0.6,
      "bearish_pct": 47.9
    },
    {
      "predicted": "neutral",
      "total": 0,
      "actual_bullish": 0,
      "actual_neutral": 0,
      "actual_bearish": 0,
      "bullish_pct": null,
      "neutral_pct": null,
      "bearish_pct": null
    },
    {
      "predicted": "bearish",
      "total": 28,
      "actual_bullish": 13,
      "actual_neutral": 0,
      "actual_bearish": 15,
      "bullish_pct": 46.43,
      "neutral_pct": 0,
      "bearish_pct": 53.57
    }
  ]
};

window.MODEL_AUDIT = [
  {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-30",
    "combined_score": 12.15,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.684,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 8.1
  },
  {
    "ticker": "ABNB",
    "company_name": "Airbnb, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-07",
    "combined_score": -15.35,
    "combined_prediction": "bearish",
    "combined_hit": true,
    "ml_probability_bullish": 0.6024,
    "ml_prediction": "bullish",
    "ml_hit": false,
    "mechanical_prediction": "bearish",
    "mechanical_hit": true,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -5.42
  },
  {
    "ticker": "AI",
    "company_name": "C3.ai, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-25",
    "combined_score": -72.18,
    "combined_prediction": "bearish",
    "combined_hit": true,
    "ml_probability_bullish": 0.3741,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bearish",
    "mechanical_hit": true,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -8.92
  },
  {
    "ticker": "AMAT",
    "company_name": "APPLIED MATERIALS INC /DE",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-12",
    "combined_score": 7.5,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.389,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 13.75
  },
  {
    "ticker": "AMD",
    "company_name": "ADVANCED MICRO DEVICES INC",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-05",
    "combined_score": 32.01,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5756,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 25.4
  },
  {
    "ticker": "AMZN",
    "company_name": "AMAZON COM INC",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-29",
    "combined_score": 5.96,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.6364,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 3.09
  },
  {
    "ticker": "APP",
    "company_name": "AppLovin Corp",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-06",
    "combined_score": 39.59,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5577,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 3.48
  },
  {
    "ticker": "AVGO",
    "company_name": "Broadcom Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-03-04",
    "combined_score": 26.61,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5906,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 5.81
  },
  {
    "ticker": "CMG",
    "company_name": "CHIPOTLE MEXICAN GRILL INC",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-29",
    "combined_score": 11.01,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.4287,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 0.85
  },
  {
    "ticker": "COST",
    "company_name": "COSTCO WHOLESALE CORP /NEW",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-03-05",
    "combined_score": 8.72,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.4676,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 2.63
  },
  {
    "ticker": "CRM",
    "company_name": "Salesforce, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-25",
    "combined_score": 25.1,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.3731,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 5.03
  },
  {
    "ticker": "CRWD",
    "company_name": "CrowdStrike Holdings, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-03-03",
    "combined_score": 40.99,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5328,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 12.93
  },
  {
    "ticker": "DDOG",
    "company_name": "Datadog, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-07",
    "combined_score": 56.73,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.6212,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 10.2
  },
  {
    "ticker": "DOCN",
    "company_name": "DigitalOcean Holdings, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-05",
    "combined_score": 47.44,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.4469,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 4.2
  },
  {
    "ticker": "GOOGL",
    "company_name": "Alphabet Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-29",
    "combined_score": 55.79,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5503,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 13.73
  },
  {
    "ticker": "GTLB",
    "company_name": "Gitlab Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-03-03",
    "combined_score": 59.78,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.2549,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -13.26
  },
  {
    "ticker": "JPM",
    "company_name": "JPMORGAN CHASE & CO",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-14",
    "combined_score": 41.75,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.4551,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 0.61
  },
  {
    "ticker": "KO",
    "company_name": "COCA COLA CO",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-28",
    "combined_score": 28.36,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5535,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 1.12
  },
  {
    "ticker": "LLY",
    "company_name": "ELI LILLY & Co",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-30",
    "combined_score": 72.19,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.8139,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 1.48
  },
  {
    "ticker": "MCD",
    "company_name": "MCDONALDS CORP",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-07",
    "combined_score": 1.75,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.4227,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -2.58
  },
  {
    "ticker": "META",
    "company_name": "Meta Platforms, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-29",
    "combined_score": 29.38,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.5795,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -7.82
  },
  {
    "ticker": "MRK",
    "company_name": "Merck & Co., Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-30",
    "combined_score": 8.05,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.4237,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 2.02
  },
  {
    "ticker": "MSFT",
    "company_name": "MICROSOFT CORP",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-29",
    "combined_score": 24.49,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.5117,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -0.87
  },
  {
    "ticker": "MU",
    "company_name": "MICRON TECHNOLOGY INC",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-03-18",
    "combined_score": 93.46,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.2808,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -23.02
  },
  {
    "ticker": "NET",
    "company_name": "Cloudflare, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-07",
    "combined_score": 40.41,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.6199,
    "ml_prediction": "bullish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -23.07
  },
  {
    "ticker": "NOW",
    "company_name": "ServiceNow, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-22",
    "combined_score": -1.31,
    "combined_prediction": "bearish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5455,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bearish",
    "mechanical_hit": true,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -14.32
  },
  {
    "ticker": "NVDA",
    "company_name": "NVIDIA CORP",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-25",
    "combined_score": 22.77,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.4002,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -6.25
  },
  {
    "ticker": "ORCL",
    "company_name": "ORACLE CORP",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-03-10",
    "combined_score": 32.03,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5706,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 2.34
  },
  {
    "ticker": "PANW",
    "company_name": "Palo Alto Networks Inc",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-17",
    "combined_score": 81.27,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.612,
    "ml_prediction": "bullish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -11.41
  },
  {
    "ticker": "PG",
    "company_name": "PROCTER & GAMBLE Co",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-24",
    "combined_score": 4.07,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.4607,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -3.21
  },
  {
    "ticker": "PLTR",
    "company_name": "Palantir Technologies Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-04",
    "combined_score": 70.94,
    "combined_prediction": "bullish",
    "combined_hit": false,
    "ml_probability_bullish": 0.5633,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": false,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -6.87
  },
  {
    "ticker": "QCOM",
    "company_name": "QUALCOMM INC/DE",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-29",
    "combined_score": 10.82,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.6214,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 29.84
  },
  {
    "ticker": "SMCI",
    "company_name": "Super Micro Computer, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-05",
    "combined_score": -1.38,
    "combined_prediction": "bearish",
    "combined_hit": false,
    "ml_probability_bullish": 0.5913,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bearish",
    "mechanical_hit": false,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 14.98
  },
  {
    "ticker": "SNOW",
    "company_name": "Snowflake Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-25",
    "combined_score": -4.93,
    "combined_prediction": "bearish",
    "combined_hit": false,
    "ml_probability_bullish": 0.5871,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bearish",
    "mechanical_hit": false,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 4.87
  },
  {
    "ticker": "UNH",
    "company_name": "UNITEDHEALTH GROUP INC",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-21",
    "combined_score": 17.4,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.6004,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 7.15
  },
  {
    "ticker": "V",
    "company_name": "VISA INC.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-04-28",
    "combined_score": 44.68,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.4896,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 3.07
  },
  {
    "ticker": "VRTX",
    "company_name": "VERTEX PHARMACEUTICALS INC / MA",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-04",
    "combined_score": 26.22,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.8172,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 4.29
  },
  {
    "ticker": "WMT",
    "company_name": "Walmart Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-19",
    "combined_score": 21.32,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.6047,
    "ml_prediction": "bullish",
    "ml_hit": true,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 2.47
  },
  {
    "ticker": "XYZ",
    "company_name": "Block, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-05-07",
    "combined_score": 31.34,
    "combined_prediction": "bullish",
    "combined_hit": true,
    "ml_probability_bullish": 0.4075,
    "ml_prediction": "bearish",
    "ml_hit": false,
    "mechanical_prediction": "bullish",
    "mechanical_hit": true,
    "actual_direction": "bullish",
    "actual_one_week_return_pct": 0.31
  },
  {
    "ticker": "ZS",
    "company_name": "Zscaler, Inc.",
    "signal_quarter": "CY2026Q1",
    "target_report_date": "2026-02-26",
    "combined_score": -22.27,
    "combined_prediction": "bearish",
    "combined_hit": true,
    "ml_probability_bullish": 0.5055,
    "ml_prediction": "bearish",
    "ml_hit": true,
    "mechanical_prediction": "bearish",
    "mechanical_hit": true,
    "actual_direction": "bearish",
    "actual_one_week_return_pct": -1.97
  }
];
