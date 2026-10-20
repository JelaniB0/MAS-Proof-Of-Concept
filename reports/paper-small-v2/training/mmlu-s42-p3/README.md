# mmlu-s42-p3 training

Status: **completed**. Unique training questions: **4**. Updates: **2**.
Successful rollouts: **4**. Requests across the available resume chain: **22**.

- [Training progress](progress.csv): scores, rewards, losses, agent counts and model choices.
- [Policy history](policy-history.json): probabilities and graphs over time, including fixed probes when recorded.
- [Probe history](probe-history.csv): decisions on the same training probes at each update (empty for older runs).
- [Current policy](current-policy.json): latest recorded probe decisions or last sampled training decision.
- [Summary](summary.json): settings, attempts and limits of this evidence.

Training scores are not held-out evaluation scores.
Policy-gradient loss is not prediction error; a smaller loss alone does not establish better accuracy or efficiency.
Rollout decisions combine learning, different questions and random sampling.
Only fixed-probe snapshots compare the same questions after parameter updates; older runs have none.
Old update boundaries are left unknown rather than reconstructed from batch size.
