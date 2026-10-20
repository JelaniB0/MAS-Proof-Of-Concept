"""Budget-aware fixed-topology baselines over the configured candidate roles."""

from dataclasses import asdict, dataclass, field
from itertools import combinations

from .config import Config, number
from .graph import ROLES, EDGES, INSTRUCTIONS, dag_waves, config_messages, token_reserve


class InfeasiblePlan(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    topology: str
    policy: str
    assignments: dict[str, str]
    reserved_cost_usd: float
    estimated_latency_seconds: float
    utility_proxy: float
    difficulty: float
    budget_usd: float
    latency_budget_seconds: float
    graph_edges: tuple[tuple[str, str], ...] | None = None
    instructions: dict[str, str] = field(default_factory=dict)
    selection: dict = field(default_factory=dict)

    @property
    def edges(self):
        return self.graph_edges if self.graph_edges is not None else EDGES[self.topology]

    def to_dict(self):
        return {**asdict(self), "edges": self.edges}


def make_plan(config: Config, task: str, topology="star", policy="balanced", budget_usd=0.02,
              latency_budget_seconds=120, difficulty=0.5, max_strong_agents=1):
    if not task.strip():
        raise ValueError("Task cannot be empty")
    if topology not in EDGES or policy not in {"cheap", "medium", "strong", "balanced"}:
        raise ValueError("Unknown topology or policy")
    if policy == "medium" and config.medium_model is None:
        raise ValueError("The medium policy requires medium_model in your configuration")
    number(budget_usd, "budget_usd")
    number(latency_budget_seconds, "latency_budget_seconds", minimum=0.001)
    number(difficulty, "difficulty", maximum=1)
    number(max_strong_agents, "max_strong_agents", maximum=3)
    if not isinstance(max_strong_agents, int):
        raise ValueError("max_strong_agents must be an integer")
    pool = (config.cheap_model, config.strong_model)
    instructions = config.agent_templates or INSTRUCTIONS
    roles = tuple(instructions)
    edges = tuple((r, "synthesizer") for r in roles[:-1]) if topology == "star" else tuple(zip(roles, roles[1:]))
    schedule = dag_waves(roles, edges)
    def assignments():
        if policy != "balanced":
            chosen = config.medium_model if policy == "medium" else pool[policy == "strong"]
            yield (chosen,) * len(roles)
            return
        for count in range(min(max_strong_agents, len(roles)) + 1):
            for indices in combinations(range(len(roles)), count):
                yield tuple(pool[int(i in indices)] for i in range(len(roles)))
    choices = assignments()
    candidates = []
    for ids in choices:
        if policy == "balanced" and ids.count(config.strong_model) > max_strong_agents:
            continue
        models = dict(zip(roles, (config.get(i) for i in ids)))
        prompts = {role: config_messages(config, task, role, {a: "\U0001f7e6" * config.context_chars for a, b in edges if b == role},
                                        instructions) for role in roles}
        cost = sum(model.cost(token_reserve(prompts[role]),
                              config.max_output_tokens) for role, model in models.items())
        latency = sum(max(models[r].latency_seconds for r in wave) for wave in schedule)
        # Manual quality ratings and task difficulty, not predicted benchmark accuracy.
        weights = {"analyst": 1, "critic": 1.2, "synthesizer": 2}
        demand = 0.35 + 0.65 * difficulty
        utility = sum(weights.get(r, 1) * min(m.quality, demand) for r, m in models.items())
        if cost <= budget_usd + 1e-12 and latency <= latency_budget_seconds:
            candidates.append(Plan(topology, policy, dict(zip(roles, ids)), cost, latency, utility,
                                   difficulty, budget_usd, latency_budget_seconds,
                                   edges if config.agent_templates else None,
                                   dict(instructions) if config.agent_templates else {}))
    if not candidates:
        raise InfeasiblePlan(f"No {policy}/{topology} plan fits the cost and latency estimates. "
                             "Increase a budget, reduce output/context limits, or adjust model profiles.")
    return min(candidates, key=lambda p: (-p.utility_proxy, p.reserved_cost_usd, p.estimated_latency_seconds))
