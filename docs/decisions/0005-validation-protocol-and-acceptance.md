# 0005 Validation protocol, venue focus, acceptance criteria (frozen 2026-10-07)

Evidence (phase 3F, dev period 2020-07..2024-12): all baselines lose after costs at 15m and 60m; spot long-only loses at every
scale tested; at 240m on perpetuals the two cases above the random null (MA cross) have bootstrap CIs that include the
null median and zero, flip sign across years and barriers, and match the number of false flags expected by chance (1.8 of 36).

Decisions:
1. Primary venue scenario: perpetuals (perp_A). Spot scenarios stay frozen as a secondary path.
2. Primary timeframe 240m. 60m only with wide barriers (4:2, 6:2) so that cost <= ~0.1R. 15m dropped.
3. Walk-forward: expanding train, first test start 2022-07-01, 6-month test windows to the end of the dev period.
   A sample is trainable only if its label resolved (exit bar) before the test window starts. The final test period stays closed.
4. Every comparison is appended to docs/trials.csv; results are read against expected_max_normal(total comparisons).
5. Funding rates and perpetual-contract candles are to be added before any live-like claim (phase 3I).

Pre-registered acceptance for a model in phase 3H (written before any model result):
- pooled out-of-sample net expectancy (R, after costs) > 0;
- bootstrap 95% CI of mean net R excludes the random-entry null median;
- positive net expectancy in >= 3 of 5 test windows and with the same sign on BTC and ETH;
- the z-score of mean net R exceeds expected_max_normal(total comparisons so far).
Models: regularised logistic regression and a small gradient-boosted tree model. Entry rule: expected value
p*(k_tp/k_sl) - (1-p) - cost_R > 0, with cost_R = round-trip cost / stop distance; no threshold tuning on test windows.
