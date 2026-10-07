# 0004 Backtest conventions, cost scenarios, baselines (frozen 2026-10-07)

Venue is undecided, so every result is reported under three cost scenarios:
- perp_A: perpetuals, long+short, max leverage 3x, fee 5 bps/side, slippage 3 bps/side.
- spot_B: spot, long only, 1x, fee 22 bps/side (maker), slippage 3 bps/side.
- spot_C: spot, long only, 1x, fee 38 bps/side (taker), slippage 3 bps/side.
Fees from the Kraken schedule as of July 2026. Slippage is an ASSUMPTION.

Engine: one position at a time, entry at next open, TP/SL/120h exits from decision 0003, risk 1% of equity per trade
divided by stop distance, capped by max leverage. Mark-to-market at bar closes.

Timeframes: 15m, 60m, 240m (240m = 4 consecutive complete 60m bars, UTC-aligned, incomplete buckets dropped).
Barriers tried (only two, to limit trials): (k_tp, k_sl) = (2,1) and (3,1).

Baselines (untuned, fixed before any result): MA cross 20/50, momentum sign over 48 bars, mean reversion z=2 on 50 bars,
buy and hold, and a random-entry null (100 seeds, entry probability 5% per bar, same exits and costs).
A strategy only counts as "different from noise" if net expectancy > 0 AND it is at or above the 95th percentile of the null.
Even then it must survive the stability and walk-forward checks before any claim.
