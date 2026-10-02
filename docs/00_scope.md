# Scope (frozen at Phase 3A)

- Market: Crypto spot, no leverage
- Provider: Kraken (behind a replaceable DataProvider interface)
- Assets: BTC/USD (XBTUSD), ETH/USD (ETHUSD)
- Timeframes: 15m and 60m, compared
- Direction: Long only (Short after validation)
- Period: mid-2020 to present; regimes tested separately, "normal period" is NOT assumed
- Initial data: OHLCV only; Funding/OI/Bid-Ask/Order Book deferred until a research reason exists
- Capital: 1000 USD
- AI goals: (1) probability of upward move, (2) probability TP before SL
- Success: edge after costs, out of sample. Accepted alternative: "no sufficient edge found"
