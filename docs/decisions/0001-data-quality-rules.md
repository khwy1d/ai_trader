# 0001 Data quality rules (frozen 2026-10-04)

Source: Kraken OHLCVT Full 2026Q2, archive SHA-256 recorded in MANIFEST.json (Drive).
Processed: BTC/USD, ETH/USD, 15m and 60m, UTC, from 2020-07-01 to 2026-06-30. Raw is never modified.

Observed: coverage 99.88-99.95% per year; 15m->60m aggregation matches the 60m files exactly;
no inconsistent OHLC bars; BTC and ETH gaps coincide on the same dates (probable venue outages, cause unverified).

Rules:
1. Gaps are never filled. They are documented in processed/kraken/GAPS_*.parquet.
2. No trades are simulated inside a gap. Stops and targets inside a gap execute at the first bar after it.
3. Features or labels crossing a gap are flagged; the include/exclude rule is fixed before any performance result is viewed.
4. Large moves (>10%) are real events and are kept. Slippage assumptions widen on such bars.
5. Open question: timestamp = bar open or bar close. Must be tested before any feature is built (Phase 3D).
6. Data ends 2026-06-30; July-October 2026 is not covered.
