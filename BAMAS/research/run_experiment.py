import os
import sys
from pathlib import Path
from llm_client import generate_answer
from evaluator import evaluate_answer, calculate_cost, calculate_reward
import json
import csv
from contextual_bandit import LinUCB
import numpy as np

PREDICTOR_DIR = Path(os.environ["DIFFICULTY_PREDICTOR_DIR"])
sys.path.insert(0, str(PREDICTOR_DIR))

from difficulty.model import DifficultyPredictor
from difficulty.inputs import controller_features

ARMS = {
    0: {
        "name": "nano",
        "deployment": "gpt-5-nano"
    },
    1: {
        "name": "mini",
        "deployment": "gpt-4.1-mini"
    },
    2: {
        "name": "strong",
        "deployment": "gpt-5.1"
    }
}

predictor_path = (
    PREDICTOR_DIR
    / "difficulty"
    / "runs"
    / "all_routerbench_v1"
    / "main"
    / "selected_model"
)

predictor = DifficultyPredictor.load(
    predictor_path,
    device="auto"
)

dataset_path = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "gsm8k_train_w_curriculum.jsonl"
)

samples = []

with open(dataset_path, "r") as file:
    for line in file:
        samples.append(json.loads(line))


samples = samples[:20]

print("Number of questions:", len(samples))


bandit = LinUCB(n_arms=3, d=4, alpha=0.5)

# GPT 5.1 cost assumptions

budget = 0.001
cost_weight = 0.2

total_correct = 0
total_cost = 0
total_reward = 0

results = []
MODEL_PRICES = {
    0: {"input": 0.05, "output": 0.40},  # GPT-5 nano
    1: {"input": 0.40, "output": 1.60},  # GPT-4.1 mini
    2: {"input": 1.375, "output": 11.00}  # GPT-5.1, provisional
}
def save_results(results, output_path):
    if not results:
        return

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

output_path = Path("results")
output_path.mkdir(exist_ok=True)

results_file = output_path / "gsm8k_demo_cleanrun.csv"

for i, sample in enumerate(samples, start=1):

    question = sample["question"]
    reference_answer = sample["answer"]

    # getting difficulty context from frozen predictor
    features = controller_features(predictor, [question])[0]

    context = np.array(
        [1.0] + features["difficulty_vector"],
        dtype=float
    )

    # Calculate UCB scores for every model
    ucb_scores = {}

    for arm in range(bandit.n_arms):
        ucb_scores[arm] = bandit.calculate_ucb(arm, context)

    # Give every arm one initial observation
    if i <= bandit.n_arms:
        selected_arm = i - 1
    else:
        selected_arm = max(ucb_scores, key=ucb_scores.get)

    print(f"\n{'='*55}")
    print(f"QUESTION {i}/{len(samples)}")
    print(f"{'='*55}")

    print("\nDifficulty context:")
    print(f"  Weak:         {context[1]:.4f}")
    print(f"  Intermediate: {context[2]:.4f}")
    print(f"  Strong:       {context[3]:.4f}")

    print("\nLinUCB scores:")

    for arm, score in ucb_scores.items():
        print(f"  {ARMS[arm]['name']}: {score:.4f}")

    print(f"\nSelected model: {ARMS[selected_arm]['name']}")

    if i <= bandit.n_arms:
        print("Selection reason: Initial exploration")
    else:
        print("Selection reason: Highest UCB score")
    # generate answer
    selected_deployment = ARMS[selected_arm]["deployment"]

    answer, input_tokens, output_tokens = generate_answer(
    question,
    deployment=selected_deployment
    )

    # evaluating correctness
    correctness = evaluate_answer(answer, reference_answer)


     # calculating API cost
    cost = calculate_cost(
        input_tokens,
        output_tokens,
        MODEL_PRICES[selected_arm]["input"],
        MODEL_PRICES[selected_arm]["output"]
    )

    # calcukating reward
    reward = calculate_reward(
        correctness,
        cost,
        budget,
        cost_weight
    )
    results.append({
    "question_id": i,
    "selected_arm": selected_arm,
    "difficulty_weak": context[1],
    "difficulty_intermediate": context[2],
    "difficulty_strong": context[3],
    "input_tokens": input_tokens,
    "output_tokens": output_tokens,
    "correctness": correctness,
    "cost": cost,
    "reward": reward
    })

    # update LinUCB
    bandit.update(selected_arm, context, reward)

    print("\nLinUCB learning update:")
    print(f"  Updated model: {ARMS[selected_arm]['name']}")
    print(f"  Observed reward: {reward:.4f}")
    print("  Model parameters updated successfully.")

    save_results(results, results_file)
    # track experiment totals
    total_correct += correctness
    total_cost += cost
    total_reward += reward

    print(f"\nQuestion {i}/{len(samples)}")
    print("Selected arm:", selected_arm)
    print("Answer:", answer)
    print("Correctness:", correctness)
    print("Cost ($):", cost)
    print("Reward:", reward)


# final results
print("\n========== EXPERIMENT SUMMARY ==========")
print("Questions:", len(samples))
print("Accuracy:", total_correct / len(samples))
print("Total API cost ($):", total_cost)
print("Average API cost ($):", total_cost / len(samples))
print("Average reward:", total_reward / len(samples))

# save experiment results to file
print("Results saved to:", results_file)