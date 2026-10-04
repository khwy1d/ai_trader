# 0002 Timestamp convention (verified 2026-10-04)

Kraken OHLCVT timestamp = bar OPEN time (UTC). Bar T covers [T, T+timeframe).

Evidence: 15m bars rebuilt from raw exchange trades (REST Trades endpoint), window 2026-06-29 12:00-14:00 UTC,
BTC/USD and ETH/USD: open/high/low/close equal to the CSV on all 16 bars. The label=CLOSE hypothesis matched 0 of 16.
Volume differences were caused by duplicated trade ids at pagination boundaries in the fetch code (8 in BTC, 2 in ETH),
equal to the trade-count differences bar by bar. Not a data fault.

Limits: one two-hour window, one day. Re-check on an older window is optional.

Rules:
1. A feature computed from bar T is available only at T + timeframe.
2. Earliest possible execution is the OPEN of bar T+1. Never fill at the close that produced the signal.
3. Any code that fetches trades must drop duplicate ids before aggregating.
