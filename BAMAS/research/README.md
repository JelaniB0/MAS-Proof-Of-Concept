# Difficulty-Aware LLM Selection Using LinUCB

## 1. Project Overview

This project investigates how contextual multi-armed bandits, specifically Linear Upper Confidence Bound (LinUCB), can improve the efficiency of Large Language Model (LLM) selection.

Given a task, the system uses a pretrained difficulty predictor to estimate its difficulty across three historical model groups: weak, intermediate, and strong. These estimates are provided to LinUCB as contextual information.

LinUCB uses this context to select an LLM to answer the task. The objective is to achieve high task correctness while minimizing API cost, optimizing cost-effectiveness.

## 2. Current Methodology

The system consists of four main components:

1. **Difficulty Prediction:** A frozen, pretrained difficulty predictor estimates task difficulty across the historical weak, intermediate, and strong model groups.

2. **Context Construction:** The three predicted difficulty scores are combined with an intercept to form the context vector:

   `x = [1, d_weak, d_intermediate, d_strong]`

   This context vector is the only task-specific information currently provided to LinUCB.

3. **LLM Selection Using LinUCB:** Each available LLM is treated as an arm. LinUCB maintains a separate reward model for each arm and uses predicted reward estimates and an exploration bonus to select an LLM.

4. **Reward and Learning:** After the selected LLM generates an answer, the system evaluates its correctness and calculates its actual API cost. These observations are combined into a reward, which updates the selected arm's parameters. Over repeated tasks, LinUCB learns which models are more cost-effective under different difficulty contexts.

The difficulty predictor remains frozen throughout the experiment. It supplies contextual information but does not learn from LinUCB's decisions.

## 3. System Architecture

The system follows a sequential contextual bandit workflow:

1. A task is received as input.
2. The frozen difficulty predictor processes the task and generates three difficulty estimates corresponding to the historical weak, intermediate, and strong model groups.
3. The estimates are assembled into a context vector.
4. LinUCB calculates an upper confidence bound (UCB) for each available LLM arm.
5. The arm with the highest UCB score is selected.
6. The selected LLM receives the task and generates an answer.
7. The evaluator compares the generated answer against the reference answer to determine correctness.
8. API token usage is used to calculate the actual cost of the call.
9. Correctness and cost are combined into a reward.
10. LinUCB updates the selected arm's parameters using the observed context and reward.

The predictor remains frozen. Only the LinUCB parameters are updated as tasks are processed.

### Context Vector

`x = [1, d_weak, d_intermediate, d_strong]`

### Reward Function

`R = Q - λ_c(C / B)`

Where:

- `Q` = answer correctness (0 or 1)
- `C` = actual API cost
- `B` = cost normalization budget
- `λ_c` = cost penalty weight

The reward parameters are provisional and will be examined during experimental evaluation.

## 4. Experimental Setup

### Models

The current implementation uses three Azure-hosted LLM deployments as LinUCB arms. These are distinct from the historical models used to train the difficulty predictor.

| Arm | Deployment | Relative capability |
|---|---|---|
| 0 | `gpt-5-nano` | Lower |
| 1 | `gpt-4.1-mini` | Intermediate |
| 2 | `gpt-5.1` | Higher |

### Dataset

The current implementation includes a 20-question GSM8K pilot to verify that the end-to-end pipeline runs successfully.

GSM8K was included in the difficulty predictor's training data, so this pilot is intended as a proof of concept rather than an independent evaluation.

The final evaluation dataset, split, sample count, and evaluation protocol are still to be finalized.

### Evaluation Metrics

- **Accuracy:** Proportion of questions answered correctly.
- **Average API Cost:** Mean API cost per task.
- **Cumulative Reward:** Sum of observed rewards across tasks.
- **Model Selection Frequency:** Number and proportion of times each model is selected.
- **Cost-effectiveness:** Correctness relative to API expenditure.

### LinUCB Configuration

Current configuration:

- Context dimension: 4
- Number of arms: 3
- Exploration parameter (`alpha`): 0.5
- Reward parameters: Provisional; subject to further evaluation.

## 5. Planned Baseline Comparisons

The LinUCB approach will be compared against alternative model-selection strategies, including:

- **Random Selection:** Randomly selects an available model.
- **Always Cheapest:** Always selects the lowest-cost model.
- **Always Weak:** Always selects the lower-capability model.
- **Always Intermediate:** Always selects the intermediate-capability model.
- **Always Strong:** Always selects the higher-capability model.
- **Hard Threshold Selection:** Uses predefined difficulty thresholds to select a model based on predicted difficulty, without contextual bandit learning.

These baselines will provide comparisons against fixed model-selection strategies, random selection, and a rule-based difficulty-aware approach.

These comparisons are planned and are not yet part of the completed implementation.

## 6. Installation and Environment Setup

The experiment requires Python 3.11, the project repository, the pretrained difficulty predictor, and Azure OpenAI API access.

The difficulty predictor is maintained in a separate repository and must be accessible locally.

### Activate the Environment

```bash
cd ~/Documents/research/predictor-repo/difficulty-predictor
source .venv/bin/activate
```

### Configure Environment Variables

```bash
export DIFFICULTY_PREDICTOR_DIR="$HOME/Documents/research/predictor-repo/difficulty-predictor"
export AZURE_OPENAI_API_KEY="YOUR_API_KEY"
export AZURE_OPENAI_DEPLOYMENT="YOUR_DEPLOYMENT_NAME"
```

Replace the placeholders with your own Azure configuration. Do not commit API keys or other credentials to the repository.

### Run the Experiment

Navigate to the research directory:

```bash
cd ~/Documents/research/MAS-Proof-Of-Concept/BAMAS/research
```

Run:

```bash
python run_experiment.py
```

The experiment loads the pretrained difficulty predictor, generates task difficulty features, selects an answering model using LinUCB, evaluates the response, calculates cost and reward, and updates the bandit.

Results are saved incrementally as a CSV file in the `results/` directory.

**Note:** The current runner is configured for a small GSM8K pilot. Before running the final evaluation, confirm dataset independence, verify Azure pricing, and configure the experiment runner for the approved evaluation dataset.

## 7. Running Tests

From the `research/` directory:

```bash
python -m pytest tests/
```

## 8. Limitations and Assumptions

### Difficulty Predictor Limitations

The bandit relies on a pretrained difficulty predictor that produces three difficulty scores for the weak, intermediate, and strong model groups. These scores represent predicted historical quality deficits, not calibrated probabilities of failure.

The current context uses only the three difficulty scores and an intercept term. Other potentially useful features, such as prompt length or score differences, are not currently included.

### Model Selection Limitations

The current experiment uses three Azure-hosted models: GPT-5 Nano, GPT-4.1 Mini, and GPT-5.1. These models represent different capability and cost levels, but they are not identical to the historical models used to train the difficulty predictor.

Results may therefore depend on how well historical difficulty estimates transfer to the current answering models.

### Cost and Reward Assumptions

API cost is estimated using input and output token counts and model-specific pricing. The reward function combines correctness and normalized API cost. Its cost weight and normalization budget influence the tradeoff and will require further justification.

### Dataset Limitations

The GSM8K pilot is not an independent evaluation because GSM8K was used in training the difficulty predictor. A suitable evaluation dataset and protocol must be established before drawing conclusions about generalization or the effectiveness of the proposed approach.