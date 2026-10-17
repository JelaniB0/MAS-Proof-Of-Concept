from eapae_agent_sys.utils.config_loader import ConfigLoader
from eapae_agent_sys.planning.low_level_instantiator import LowLevelInstantiator
from eapae_agent_sys.execution.execution_engine import ExecutionEngine

config_loader = ConfigLoader()
instantiator = LowLevelInstantiator(config_loader)
engine = ExecutionEngine(config_loader, instantiator)

import json
import random

budget = 2000

with open("data/gsm8k/test.jsonl", "r") as f:
    questions = [json.loads(line) for line in f]

question = random.choice(questions)
task = question["question"]

print("Selected question:")
print(task)

for topology_name, pattern_name in [
    ("LINEAR", "linear_topology"),
    ("STAR", "star_topology"),
]:
    print(f"\n\n########## RUNNING {topology_name} ##########")
    agent_pool, feasible = instantiator.solve(pattern_name, budget)
    if not feasible:
        print(f"Could not build a feasible agent pool for {topology_name}")
        continue
    if topology_name == "LINEAR":
        result, cost = engine._execute_linear_template(agent_pool, budget, task, "gsm8k")
    else:
        result, cost = engine._execute_star_template(agent_pool, budget, task, "gsm8k")
    print(f"\nTOTAL COST for {topology_name}: {cost}")
    for step in result["history"]:
        print(step)
        