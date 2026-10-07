# 0006 Phase 3H model evaluation rules (frozen 2026-10-07, before any model result)

Amends 0005:
- Models: regularised logistic regression (StandardScaler + L2, C=0.1) and sklearn HistGradientBoostingClassifier
  (depth 3, 150 iterations, learning rate 0.05, min leaf 100, L2 1.0). Fixed; no tuning on any test window.
- Feature `body_pct` removed (correlation 0.997 with `ret_1`).
- Target: take-profit before stop; timeouts count as 0.
- Entry: EV = p*(k_tp/k_sl) - (1-p) - cost_R > 0 per side, larger EV wins; cost_R = 2*cost / stop distance (perp_A).
- Execution: same engine, exits and costs as the baselines; signals only inside test windows (first test 2022-07-01, 6-month windows).
- Nulls: random entries 50/50 sides, and random entries with the model's own long/short mix (100 seeds each).
  The dev sample is a strong bull market (long-only gross R > 0 and short-only < 0 in all 16 datasets), so a long-biased
  model must beat the side-matched null, not only the symmetric one.
- Acceptance (per model/timeframe/barrier, on BOTH assets): net R > 0; bootstrap 95% CI lower bound above both null medians;
  >= 3 of 5 windows positive; z above expected_max_normal(total comparisons logged so far + this round).
- Reported next to net R: gross R and AUC per side, because wider-stop trade selection lowers cost in R without any
  directional skill.
- This is screening-level evidence on the development period. Passing it only permits proceeding to funding costs,
  perpetual-contract data and a longer walk-forward; it never permits a performance claim.
