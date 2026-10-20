"""Trainable backbone-then-topology controller, independently implemented from the paper.

Discrete choices use exact categorical/Bernoulli sampling and REINFORCE. We do
not claim to reproduce the paper's Gumbel relaxations or published checkpoints.
"""

from dataclasses import asdict
import copy
import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical

from ..graph import INSTRUCTIONS, dag_waves, messages, token_reserve
from ..planner import InfeasiblePlan, Plan
from ..files import replace_file
from .encoder import DifficultyEstimator, TextEncoder
from .pools import construct_pools


def identity(config):
    """Tie weights to profiles/templates, never environment secrets."""
    data = {"models": [asdict(m) for m in config.models], "agents": config.agent_templates or INSTRUCTIONS,
            "context_chars": config.context_chars, "max_output_tokens": config.max_output_tokens}
    if config.min_workers != 1:
        data["min_workers"] = config.min_workers
    if config.pool_count is not None or config.pool_min_size != 1:
        data["pool_settings"] = [config.pool_count, config.pool_min_size]
    if config.pool_max_size is not None:
        data["pool_max_size"] = config.pool_max_size
    if config.controller_method != "legacy" or config.explicit_pools:
        data["method"] = [config.controller_method, config.difficulty_offset,
                          config.pool_upper_bound, config.explicit_pools, config.decision_model, config.ablation]
        data["agent_behavior"] = [config.agent_tools, config.prompt_prefixes]
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def model_names(config):
    # Bind new checkpoints to resolved model names, without storing credentials.
    from ..providers import expand
    return {m.id: expand(m.model) for m in config.models}


class Controller(nn.Module):
    def __init__(self, config, encoder, hidden=64, pool_count=None):
        super().__init__()
        self.config, self.encoder, self.hidden = config, encoder, hidden
        self.instructions = config.agent_templates or INSTRUCTIONS
        self.roles = tuple(self.instructions)
        self.pools, self.filtered = construct_pools(config.models,
                                                   config.pool_count if pool_count is None else pool_count,
                                                   config.pool_min_size, config.pool_max_size)
        self.ids = tuple(m.id for m in config.models)
        dimension = encoder.dimension
        self.difficulty = DifficultyEstimator(dimension, hidden)
        self.profile_projection = nn.Linear(dimension, hidden)
        self.model_fusion = nn.Linear(3 * hidden, hidden)
        self.role_projection = nn.Linear(dimension, hidden)
        self.query_projection = nn.Linear(dimension, hidden)
        self.role_match = nn.Linear(3 * hidden, hidden)
        self.pool_score = nn.Linear(hidden, 1)
        self.cost_aversion = nn.Parameter(torch.tensor(0.0))
        self.context_query = nn.Linear(2 * hidden, hidden)
        self.context_key = nn.Linear(hidden, hidden)
        self.context_value = nn.Linear(hidden, hidden)
        self.context_output = nn.Linear(hidden, hidden)
        self.context_scale = nn.Parameter(torch.tensor(0.1))
        self.gate = nn.Linear(2 * hidden, 1)
        self.edge_source = nn.Linear(hidden, hidden)
        self.edge_target = nn.Linear(hidden, hidden)
        self.hops = nn.Linear(hidden, len(self.roles) - 1)
        self.training_info = {"difficulty_steps": 0, "policy_steps": 0, "fixture": encoder.name == "hash-test"}

    def describe_profiles(self):
        texts = []
        for m in self.config.models:
            texts.extend([m.performance_profile or f"Model {m.model}, configured quality {m.quality}",
                          f"Input price {m.input_usd_per_million}; output price {m.output_usd_per_million} USD per million tokens",
                          f"Model type {m.model_type}"])
        return self.encoder.encode(texts).reshape(len(self.ids), 3, -1)

    def select(self, task, budget_usd=0.02, latency_budget_seconds=120, sample=False,
               topology="adaptive", max_pool=None):
        if not task.strip():
            raise ValueError("Task cannot be empty")
        if not math.isfinite(budget_usd) or budget_usd < 0 or not math.isfinite(latency_budget_seconds) or latency_budget_seconds <= 0:
            raise ValueError("Budgets must be finite; cost nonnegative and latency positive")
        if topology not in {"adaptive", "star", "linear"}:
            raise ValueError("Unknown learned topology mode")
        log_prob, entropy = [], []
        decisions = {"filtered_models": self.filtered, "pools": self.pools,
                     "training": dict(self.training_info), "encoder": self.encoder.name, "actions": []}

        def choose(probabilities, binary=False, label="choice"):
            distribution = Bernoulli(probs=probabilities) if binary else Categorical(probs=probabilities)
            action = distribution.sample() if sample else (
                (probabilities >= .5).float() if binary else probabilities.argmax())
            term = distribution.log_prob(action).sum()
            log_prob.append(term)
            entropy.append(distribution.entropy().sum())
            decisions["actions"].append({"decision": label, "action": int(action.item()),
                                         "log_probability": float(term.detach())})
            return int(action.item())

        query = self.encoder.encode([task])[0]
        demand = self.difficulty(query)
        q = self.query_projection(query).tanh()
        profiles = self.profile_projection(self.describe_profiles()).tanh()
        model_vectors = self.model_fusion(profiles.flatten(1)).tanh()
        pool_vectors = torch.stack([model_vectors[[self.ids.index(i) for i in pool]].mean(0) for pool in self.pools])
        limit = len(self.pools) - 1 if max_pool is None else max_pool
        if not 0 <= limit < len(self.pools):
            raise ValueError("max_pool is outside available pool indices")
        tiers = torch.linspace(0, 1, len(self.pools))
        logits = self.pool_score(pool_vectors).squeeze(-1) - nn.functional.softplus(self.cost_aversion) * tiers
        # Monotone difficulty bucket boundaries, with a smooth width for gradients.
        weights = logits[:limit + 1].softmax(0)
        boundaries = weights.cumsum(0)[:-1]
        cdf = torch.sigmoid((boundaries - demand) / .1)
        probabilities = torch.cat([cdf, cdf.new_ones(1)]) - torch.cat([cdf.new_zeros(1), cdf])
        probabilities = probabilities.clamp_min(1e-7)
        probabilities = probabilities / probabilities.sum()
        pool_index = choose(probabilities, label="pool")
        pool = self.pools[pool_index]
        decisions.update(difficulty=float(demand.detach()), pool_probabilities=probabilities.detach().tolist(),
                         selected_pool=pool_index, model_probabilities={}, keep_probabilities={}, edge_probabilities={})
        candidates = [self.ids.index(i) for i in pool]
        context = model_vectors[candidates].mean(0)
        role_vectors = self.role_projection(self.encoder.encode(list(self.instructions.values()))).tanh()
        assigned, representations = {}, []
        for index, role in enumerate(self.roles):
            role_match = self.role_match(torch.cat([role_vectors[index], q, context])).tanh()
            probabilities = (model_vectors[candidates] @ role_match / math.sqrt(self.hidden)).softmax(0)
            selected = candidates[choose(probabilities, label=f"model:{role}")]
            assigned[role] = self.ids[selected]
            decisions["model_probabilities"][role] = dict(zip(pool, probabilities.detach().tolist()))
            query_context = self.context_query(torch.cat([role_vectors[index], q]))
            weights = (self.context_key(profiles[selected]) @ query_context / math.sqrt(self.hidden)).softmax(0)
            attended = (weights[:, None] * self.context_value(profiles[selected])).sum(0)
            representations.append(role_vectors[index] + self.context_scale * self.context_output(attended))
        representations = torch.stack(representations)
        global_context = representations.mean(0)
        keep = self.gate(torch.cat([representations, global_context.expand_as(representations)], dim=-1)).squeeze(-1).sigmoid()
        active = []
        if topology == "adaptive":
            for index, role in enumerate(self.roles[:-1]):
                decisions["keep_probabilities"][role] = float(keep[index].detach())
                if choose(keep[index], binary=True, label=f"keep:{role}"):
                    active.append(role)
            if len(active) < self.config.min_workers:
                best = torch.topk(keep[:-1], self.config.min_workers).indices.tolist()
                forced = [self.roles[i] for i in best if self.roles[i] not in active]
                active = [r for r in self.roles[:-1] if r in active or r in forced]
                decisions["forced_workers"] = forced
                if self.config.min_workers == 1:
                    decisions["forced_worker"] = active[0]
        else:
            active = list(self.roles[:-1])
        active.append("synthesizer")
        decisions["pruned_agents"] = [role for role in self.roles if role not in active]
        assigned = {role: assigned[role] for role in active}
        edges = []
        length_penalty = q.sum() * 0
        if topology == "star":
            edges = [(role, "synthesizer") for role in active[:-1]]
        elif topology == "linear":
            edges = list(zip(active, active[1:]))
        else:
            # Fixed template order restricts admissible edges to an acyclic graph.
            # Worker->final edges guarantee all retained work reaches the answer.
            edges = [(role, "synthesizer") for role in active[:-1]]
            edge_scores = {}
            for index, source in enumerate(active[:-1]):
                for target in active[index + 1:-1]:
                    a, b = representations[self.roles.index(source)], representations[self.roles.index(target)]
                    probability = (self.edge_source(a) @ self.edge_target(b) / math.sqrt(self.hidden)).sigmoid()
                    edge_scores[(source, target)] = float(probability.detach())
                    decisions["edge_probabilities"][f"{source}->{target}"] = float(probability.detach())
                    if choose(probability, binary=True, label=f"edge:{source}->{target}"):
                        edges.append((source, target))
            hop_probs = self.hops(representations[[self.roles.index(r) for r in active]].mean(0))[:len(active)-1].softmax(0)
            hop_limit = choose(hop_probs, label="hops") + 1
            expected_hops = (hop_probs * torch.arange(1, len(active), dtype=torch.float)).sum()
            decisions.update(hop_probabilities=hop_probs.detach().tolist(), hop_limit=hop_limit)

            def critical_path():
                paths = {}
                for role in active:
                    prior = [paths[a] + [(a, b)] for a, b in edges if b == role]
                    paths[role] = max(prior, key=len, default=[])
                return paths["synthesizer"]

            length_penalty = (expected_hops.new_tensor(len(critical_path())) - expected_hops).relu()
            removed = []
            while len(critical_path()) > hop_limit:
                removable = [edge for edge in critical_path() if edge[1] != "synthesizer"]
                edge = min(removable, key=lambda e: edge_scores[e])
                edges.remove(edge)
                removed.append(edge)
            decisions["hop_pruned_edges"] = removed
        schedule = dag_waves(active, edges)
        cost = 0
        for role, model_id in assigned.items():
            incoming = {a: "\U0001f7e6" * self.config.context_chars for a, b in edges if b == role}
            prompt = messages(task, role, incoming, self.config.context_chars, self.instructions)
            cost += self.config.get(model_id).cost(token_reserve(prompt), self.config.max_output_tokens)
        latency = sum(max(self.config.get(assigned[r]).latency_seconds for r in wave) for wave in schedule)
        decisions["feasible"] = cost <= budget_usd + 1e-12 and latency <= latency_budget_seconds
        decisions["sampled"] = sample
        decisions["joint_log_probability"] = float(torch.stack(log_prob).sum().detach())
        plan = Plan(topology, "learned", assigned, cost, latency, 0, float(demand.detach()),
                    budget_usd, latency_budget_seconds, tuple(edges), dict(self.instructions), decisions)
        return plan, torch.stack(log_prob).sum(), torch.stack(entropy).sum(), length_penalty

    def plan(self, task, **options):
        with torch.no_grad():
            plan, _, _, _ = self.select(task, **options)
        if not plan.selection["feasible"]:
            raise InfeasiblePlan("Learned configuration exceeds cost/latency estimates; no model calls were made")
        return plan

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {"format": 2, "config_identity": identity(self.config), "encoder": self.encoder.name,
                      "dimension": self.encoder.dimension, "hidden": self.hidden, "pool_count": len(self.pools),
                      "resolved_models": model_names(self.config),
                      "training": self.training_info, "state_dict": self.state_dict()}
        if self.config.controller_method == "paper":
            checkpoint.update(format=3, method="paper", normalize=self.encoder.normalize,
                              difficulty_encoder=getattr(self, "difficulty_encoder", self.encoder).name)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(checkpoint, temporary)
        replace_file(temporary, path)


def load_controller(path, config, allow_untrained=False):
    # Never load arbitrary pickled objects from a checkpoint.
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("format") not in {1, 2, 3} or data.get("config_identity") != identity(config):
        raise ValueError("Checkpoint does not match these model profiles, templates, or execution settings; retrain it")
    if data.get("format") in {2, 3} and data["resolved_models"] != model_names(config):
        raise ValueError("Resolved model names changed; use research transfer with the original source configuration/environment")
    if not allow_untrained and (not data["training"].get("policy_steps") or data["training"].get("fixture")):
        raise ValueError("Checkpoint is untrained or fixture-only; use a trained checkpoint or explicit --allow-untrained for inspection")
    encoder = TextEncoder(data["encoder"], data["dimension"], normalize=data.get("normalize", True))
    controller = build_controller(config, encoder, data["hidden"], data["pool_count"])
    if data.get("difficulty_encoder") and data["difficulty_encoder"] != encoder.name:
        controller.difficulty_encoder = TextEncoder(data["difficulty_encoder"], normalize=False,
            max_length=data["training"].get("difficulty_max_length", 384))
    controller.load_state_dict(data["state_dict"])
    controller.training_info = data["training"]
    controller.eval()
    return controller


def transfer_controller(path, source_config, target_config, pool_count=None):
    """Explicit zero-shot transfer; preserve learned shared weights and leakage history."""
    source = load_controller(path, source_config, allow_untrained=True)
    if list(source.instructions.items()) != list((target_config.agent_templates or INSTRUCTIONS).items()):
        raise ValueError("Transfer requires identical ordered role templates; train a new controller for a different roster")
    if source_config.min_workers != target_config.min_workers:
        raise ValueError("Transfer requires the same min_workers")
    if source_config.controller_method != target_config.controller_method:
        raise ValueError("Transfer requires the same controller_method")
    target = build_controller(target_config, source.encoder, source.hidden, pool_count)
    if hasattr(source, "difficulty_encoder"):
        target.difficulty_encoder = source.difficulty_encoder
    target.load_state_dict(source.state_dict(), strict=True)
    target.training_info = copy.deepcopy(source.training_info)
    history = target.training_info.setdefault("transfers", [])
    history.append({"source_checkpoint": str(path), "source_config_identity": identity(source_config),
                    "target_config_identity": identity(target_config),
                    "source_models": model_names(source_config), "target_models": model_names(target_config),
                    "source_pools": source.pools, "target_pools": target.pools,
                    "source_policy_steps": source.training_info.get("policy_steps", 0)})
    target.training_info["target_policy_steps"] = 0
    target.eval()
    return target


def build_controller(config, encoder, hidden=None, pool_count=None):
    if config.controller_method == "paper":
        from .paper import PaperController
        return PaperController(config, encoder, hidden or 256, pool_count)
    return Controller(config, encoder, hidden or 64, pool_count)
