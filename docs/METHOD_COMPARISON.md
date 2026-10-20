# AgentBalance: paper and implementation

Specification: [AgentBalance v1](https://arxiv.org/html/2512.11426v1), sections 2–4
and appendices B, D and F. The pinned author implementation is
[e5aeac4](https://github.com/usail-hkust/AgentBalance_/tree/e5aeac4366fe6296d4a1381ef537508e3af784e9).
This comparison concerns `controller_method: "paper"`; the legacy controller differs.

## What the paper does

| Topic | Paper method |
| --- | --- |
| Cost evaluation | Token counts multiplied by per-token prices give monetary cost. Correctness competes with cost and latency in the training reward; performance–cost curves measure efficiency. |
| Budget evaluation | Optimize expected cost and latency under budgets. Difficulty offsets, pool caps and penalty sweeps produce operating points; performance at budgets and frontier AUC compare methods. |
| Performance evaluation | Grade MMLU/MATH accuracy and HumanEval Pass@1 on held-out tasks. Compare baselines and independent runs at matched resource budgets. |
| Dynamic topology | Profile/group models, choose a pool using difficulty, assign models to roles, represent agents, gate workers, sample forward edges and prune long paths with a predicted hop bound. Joint policy-gradient training updates these decisions. |

## What this repository does after cleanup

### 1. Cost evaluation

[`Model.cost`](../agentbalance/config.py) applies configured input/output rates
in USD per million tokens. [`execute`](../agentbalance/runner.py) records per-agent
usage and sums graph cost. Missing provider usage uses conservative reservations;
a failed request can have unknown usage, which stays explicitly marked.

[`train_policy`](../agentbalance/adaptive/training.py) supports:

```text
raw reward = correctness - cost_weight × USD - latency_weight × graph_seconds
paper loss = -reward × joint_log_probability + length_weight × hop_penalty
```

Normalized rewards divide USD and seconds by per-query budgets. The paper pipeline
explicitly selects raw units. This objective does not prove the controller finds
a globally optimal configuration.

**Match:** cost/performance/latency reward and joint decision likelihood follow the
core method. **Differences:** USD replaces the reference price currency; Groq
substitutes for its backbone roster; local profiling measures calibration cost
instead of recreating reasoning multipliers and activated-parameter estimates.
Changing rates or currency requires reviewing penalty magnitudes. All current
rates are zero, so the monetary reward term vanishes even with a positive
`cost_weight`. Tokens/requests still consume quota. Direct token/request penalties
are not implemented.

### 2. Budget evaluation

The controller checks estimated graph USD and latency before admission; pool caps
and difficulty offsets constrain the resource tier. [`RunAllowance`](../agentbalance/adaptive/training.py)
adds overall request/USD caps. [`execute`](../agentbalance/runner.py) reserves the
next ready set/wave, tracks wall time and stops further calls on overrun or failure.
Admitted requests can still finish; these guards are not provider billing controls.

[`pipeline.py`](../agentbalance/pipeline.py) creates four operating points.
[`metrics.py`](../agentbalance/adaptive/metrics.py) computes per-seed upper concave
frontiers, interpolated performance-at-budget and AUC within a shared fixed-strong
endpoint. Cost is total evaluation USD; latency is mean graph seconds. Partial
and simulated evaluations are excluded by default. Zero-cost AUC is undefined.

**Match:** pool caps, penalty sweeps and budget/frontier metrics are present.
**Differences:** hard graph/request guards extend expected-budget optimization;
the common endpoint is our strong star baseline, not an author baseline run.
Three candidates can produce overlapping/identical pools, so four operating points
may not represent four distinct tiers. Estimated latency uses calibrated model
times/dependencies and omits explicit quota spacing; measured/runtime latency
includes it. This explains the large estimate/measurement gap in the slow demo.

### 3. Performance evaluation

[`Scorer`](../agentbalance/adaptive/scoring.py) and
[`paper_scoring.py`](../agentbalance/paper_scoring.py) grade benchmark answers using
parsers and Docker tests. MATH normalization is not general symbolic equivalence.
HumanEval hidden tests stay in the grader. Evaluation rejects exact normalized
prompt overlap with recorded training keys; paraphrases and all pretraining
contamination cannot be excluded by this check.

Comparisons use cheap, medium, strong and balanced fixed policies. AgentPrune,
AgentDropout, G-Designer and MasRouter are not implemented. Policy training freezes
MPNet embeddings and updates the controller/ease head with separate rates.
Difficulty pretraining separately fine-tunes the last four MPNet layers plus an
ease head using equal BCE/MSE losses.

**Match:** benchmark metrics, held-out evaluation and core learning objective exist.
**Differences:** public MATH substitutes for the unavailable author sample; models,
weights, graders and API timing differ. Four training and three evaluation
questions are integration evidence. Published scores and comparative efficiency
are not reproduced. The paper also uses learning rate `0.1`; lowering it is a local
stability experiment, not a fidelity correction.

### 4. Dynamic topology

[`PaperController.select`](../agentbalance/adaptive/paper.py) builds one graph per
question before hosted execution:

```mermaid
flowchart LR
    Q[Question] --> D[Estimate difficulty]
    D --> P[Select model pool]
    P --> M[Match models to worker roles]
    M --> R[Represent assigned workers]
    R --> G[Gate workers; keep at least two]
    G --> E[Sample forward edges]
    E --> H[Prune paths using hop bound]
    H --> S[Attach decision agent]
    S --> X[Execute and grade]
    X --> U[Update controller during training]
```

Role templates/order are predefined. Edges go from earlier to later workers,
guaranteeing an acyclic graph. Gates remove workers; edges determine which upstream
answers reach each worker. The decision agent receives retained workers' outputs.
Your config fixes its model to `strong`; it is outside worker matching/gating, as
in the pinned reference runner. `star`/`linear` hold topology and roster fixed;
`adaptive` enables learned pruning/edges.

The implementation uses shared projected dot products for edge scores, Binary
Concrete sampling with logistic noise/straight-through values, an expected-hop
simplex, and repeated removal of the weakest edge on the current longest worker
path. The training loss includes the pre-pruning hop penalty.

**Match:** the backbone-then-topology mechanisms are implemented; keeping at least
two workers also matches the paper. **Differences:** independent parameterization
makes author checkpoints incompatible. Sampling/log-likelihood handling follows
the stated distributions without copying reference relaxation quirks. Distinct hop
logits/repeated critical-path pruning follow the equations rather than the pinned
source's repeated-scalar/one-removal behavior. Custom Python scheduling replaces
the author execution stack. The graph is query-specific; it is not repeatedly
rewritten during one answer.

## Implementation versus evidence

All four mechanisms exist in code. The completed MATH run demonstrates 3/3
correctness using six `strong` roles and 15 connections per question. It does not
demonstrate resource savings.
[Observed decisions](../reports/paper-small-v2/evaluation/math-eval-slow-01/policy.json)
are exported separately.

Training sampled teams of 5, 3, 6 and 6 agents across four successful rollouts.
Different questions and random decisions confound any apparent learning trend.
New fixed-probe snapshots compare the same training questions after updates;
matched held-out baselines are still required to demonstrate useful optimization.
