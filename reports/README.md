# Results

Start with the [experiment index](paper-small-v2/README.md).

The live MATH smoke run scored **3/3**, using **18 requests**, **26,740 tokens**,
all five workers and a decision agent, all assigned to `strong`.
Configured prices are zero; quota spacing dominates latency.

- [Evaluation: answers and usage](paper-small-v2/evaluation/math-eval-slow-01/README.md)
- [Evaluation: observed policy](paper-small-v2/evaluation/math-eval-slow-01/policy.json)
- [Training: progress/settings](paper-small-v2/training/math-s42-p1/README.md)
- [Training: rollout history](paper-small-v2/training/math-s42-p1/policy-history.json)

Training and evaluation are separate. Incomplete attempts are labeled. Four
successful MATH training rollouts span a resumed run; old logs have no controlled
snapshots after each update. New training exports fixed-probe history.

These are integration results, not reproduced published scores or demonstrated
savings. No matched completed MATH baseline is available yet. Offline demos live
locally in `results/archive/loose-runs/` and are excluded from this collection.

Regenerate: `python -m agentbalance report --source results/paper-small-v2 --out reports/paper-small-v2`.
