# 0003 Splits and labeling rules (frozen 2026-10-07)

Split: development = [2020-07-01, 2025-01-01); final test = [2025-01-01, 2026-07-01), opened ONCE at the very end.
Walk-forward and all tuning happen inside development only. Dev code uses ai_trader.splits.dev_only and assert_no_final_test.

Labels: take-profit / stop-loss decide the outcome; time is only a cap (120 h, measured from entry).
- Signal at bar close, entry at next bar OPEN. Barriers = entry +/- k * ATR(14) with ATR computed up to the signal bar.
- Both barriers in one bar: stop assumed first, flagged ambiguous. A bar opening beyond a barrier exits at the open.
- Invalid (excluded): entry delayed by a data gap, path crossing a gap, or path truncated by end of data.
- Labels resolving after a split boundary are purged (finalize_dev).
- Long and short are labelled separately. Short on spot prices is a proxy: funding, borrow fees and liquidation are NOT modelled
  here and enter in the cost phase.
- The (k_tp, k_sl) grid is a hyperparameter chosen by walk-forward validation inside development; every combination tried counts as a trial.
