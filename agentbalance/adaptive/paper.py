"""Independent implementation of AgentBalance v1 equations 3-11.

The paper is the specification; implementation quirks in e5aeac4 are not copied.
Workers are gated, then a separate decision node collects all retained workers,
as in the reference ABGraph. Backbones remain external inference services.
"""

import math

import torch
from torch import nn
from torch.distributions import Bernoulli, Categorical
from torch.nn import functional as F

from .controller import Controller
from .pools import construct_pools
from ..graph import INSTRUCTIONS, dag_waves, config_messages, token_reserve
from ..planner import Plan


class EaseHead(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(dimension, 512), nn.GELU(),
                                    nn.Dropout(.1), nn.Linear(512, 1))

    def forward(self, x):
        return self.layers(x).squeeze(-1).sigmoid()


class PaperController(Controller):
    def __init__(self, config, encoder, hidden=256, pool_count=None):
        nn.Module.__init__(self)
        self.config, self.encoder, self.hidden = config, encoder, hidden
        self.difficulty_encoder = encoder
        self.instructions = config.agent_templates or INSTRUCTIONS
        self.roles = tuple(self.instructions)
        if config.explicit_pools:
            self.pools = [tuple(p) for p in config.explicit_pools]
            self.filtered = [m.id for m in config.models if not any(m.id in p for p in self.pools)]
        else:
            self.pools, self.filtered = construct_pools(config.models,
                config.pool_count if pool_count is None else pool_count, config.pool_min_size, config.pool_max_size)
        self.ids = tuple(m.id for m in config.models)
        dim = encoder.dimension
        self.difficulty = EaseHead(dim)
        self.profile_projection = nn.Linear(dim, hidden)
        self.cost_type = nn.Linear(2 * dim, hidden)
        self.model_fusion = nn.Linear(dim + hidden, hidden)
        self.pool_context = nn.Linear(3 * dim, hidden)
        self.role_projection = nn.Linear(dim, hidden)
        self.query_projection = nn.Linear(dim, hidden)
        self.role_match = nn.Linear(3 * hidden, hidden)
        self.pool_score = nn.Linear(hidden, 1)
        self.cost_aversion = nn.Parameter(torch.tensor(0.))
        self.context_query = nn.Linear(2 * hidden, hidden)
        self.context_key = nn.Linear(hidden, hidden, bias=False)
        self.context_value = nn.Linear(hidden, hidden, bias=False)
        self.context_output = nn.Linear(hidden, hidden)
        self.context_scale = nn.Parameter(torch.tensor(.1))
        self.gate = nn.Linear(2 * hidden, 1)
        self.edge_projection = nn.Linear(hidden, hidden)
        self.hops = nn.Linear(hidden, max(1, len(self.roles) - 2))
        self.training_info = {"difficulty_steps": 0, "policy_steps": 0,
                              "fixture": encoder.name == "hash-test", "method": "paper-equations-v1"}

    def select(self, task, budget_usd=.02, latency_budget_seconds=120, sample=False,
               topology="adaptive", max_pool=None):
        if not task.strip() or not math.isfinite(budget_usd) or budget_usd < 0:
            raise ValueError("Task must be nonempty and cost budget finite/nonnegative")
        if not math.isfinite(latency_budget_seconds) or latency_budget_seconds <= 0:
            raise ValueError("Latency budget must be finite and positive")
        if topology not in {"adaptive", "star", "linear"}:
            raise ValueError("Unknown topology")
        terms, entropies = [], []
        decisions = {"method": "paper-equations-v1", "training": dict(self.training_info),
                     "encoder": self.encoder.name, "pools": self.pools, "filtered_models": self.filtered,
                     "actions": [], "model_probabilities": {}, "keep_probabilities": {}, "edge_probabilities": {}}

        def choose(prob, label, binary=False):
            prob = prob.clamp(1e-6, 1-1e-6) if binary else prob
            dist = Bernoulli(probs=prob) if binary else Categorical(probs=prob)
            if binary and sample:
                # Binary Concrete: logistic noise is the difference of two Gumbels.
                u = torch.rand_like(prob).clamp(1e-6, 1-1e-6)
                soft = torch.sigmoid((torch.logit(prob) + torch.logit(u)) / .5)
                hard = (soft >= .5).to(soft.dtype)
                value = hard.detach() + soft - soft.detach()
            else:
                value = dist.sample() if sample else ((prob >= .5).float() if binary else prob.argmax())
            action = (value.detach() >= .5).float() if binary else value.detach()
            term = dist.log_prob(action).sum()
            terms.append(term)
            entropies.append(dist.entropy().sum())
            decisions["actions"].append({"decision": label, "action": int(action.item()),
                                         "log_probability": float(term.detach())})
            return int(action.item()), value

        raw_query = self.encoder.encode([task])[0]
        ease = self.difficulty(self.difficulty_encoder.encode([task])[0])
        demand = 1 - ease
        effective = (demand + self.config.difficulty_offset).clamp(0, 1)
        profiles_raw = self.encoder.encode([
            text for m in self.config.models for text in (
                m.performance_profile or f"Model {m.model}, accuracy {m.quality}",
                f"Input {m.input_usd_per_million}, output {m.output_usd_per_million} USD per million tokens",
                f"Model type: {m.model_type}")]).reshape(len(self.ids), 3, -1)
        profiles = self.profile_projection(profiles_raw)
        models = self.model_fusion(torch.cat([profiles_raw[:, 0],
            self.cost_type(profiles_raw[:, 1:].flatten(1))], dim=-1))
        vectors = torch.stack([models[[self.ids.index(i) for i in pool]].mean(0) for pool in self.pools])
        limit = self.config.pool_upper_bound if max_pool is None else max_pool
        limit = len(self.pools)-1 if limit is None else limit
        if not 0 <= limit < len(self.pools):
            raise ValueError("Pool upper bound exceeds the available pools")
        curve = torch.linspace(0, 1, len(self.pools))
        logits = self.pool_score(vectors).squeeze(-1) - F.softplus(self.cost_aversion) * curve
        boundaries = logits[:limit+1].softmax(0).cumsum(0)[:-1]
        cdf = torch.sigmoid((boundaries-effective)/.1)
        prob = (torch.cat([cdf, cdf.new_ones(1)])-torch.cat([cdf.new_zeros(1), cdf])).clamp_min(1e-7)
        prob = prob/prob.sum()
        if self.config.ablation == "random-pool":
            prob = torch.ones_like(prob)/len(prob)
        pool_index, _ = choose(prob, "pool")
        pool = self.pools[pool_index]
        candidates = [self.ids.index(i) for i in pool]
        context = self.pool_context(profiles_raw[candidates].flatten(1).mean(0))
        q = self.query_projection(raw_query)
        # The decision node is outside worker generation/gating in the reference.
        workers = self.roles[:-1]
        role_vectors = self.role_projection(self.encoder.encode([self.instructions[r] for r in workers]))
        assigned, representations = {}, []
        for index, role in enumerate(workers):
            v = self.role_match(torch.cat([role_vectors[index], q, context]))
            p = (models[candidates] @ v).softmax(0)
            if self.config.ablation == "random-matching":
                p = torch.ones_like(p)/len(p)
            selected, _ = choose(p, f"model:{role}")
            selected = candidates[selected]
            assigned[role] = self.ids[selected]
            decisions["model_probabilities"][role] = dict(zip(pool, p.detach().tolist()))
            query = self.context_query(torch.cat([role_vectors[index], q]))
            attention = (self.context_key(profiles[selected]) @ query / math.sqrt(self.hidden)).softmax(0)
            attended = (attention[:, None]*self.context_value(profiles[selected])).sum(0)
            representations.append(role_vectors[index] if self.config.ablation == "role-only" else
                                   role_vectors[index]+self.context_scale*self.context_output(attended))
        representations = torch.stack(representations)
        global_context = representations.mean(0).expand_as(representations)
        keep = self.gate(torch.cat([representations, global_context], dim=-1)).squeeze(-1).sigmoid()
        active, relaxed = [], {}
        if topology == "adaptive" and self.config.ablation != "no-gating":
            for i, role in enumerate(workers):
                selected, value = choose(keep[i], f"keep:{role}", True)
                decisions["keep_probabilities"][role] = float(keep[i].detach())
                relaxed[role] = value
                if selected:
                    active.append(role)
            for i in torch.argsort(keep, descending=True).tolist():
                if len(active) >= self.config.min_workers:
                    break
                if workers[i] not in active:
                    active.append(workers[i])
                    relaxed[workers[i]] = keep[i]*0+1
            active = [r for r in workers if r in active]
        else:
            active = list(workers)
            relaxed = {r: q.new_tensor(1.) for r in active}
        h = torch.stack([representations[workers.index(r)]*relaxed[r] for r in active])
        worker_edges, scores = [], {}
        penalty = q.sum()*0
        if topology == "linear":
            worker_edges = list(zip(active, active[1:]))
        elif topology == "adaptive" and self.config.ablation == "no-topology":
            worker_edges = [(a,b) for i,a in enumerate(active) for b in active[i+1:]]
        elif topology == "adaptive":
            projected = self.edge_projection(h)
            for i, source in enumerate(active):
                for j in range(i+1, len(active)):
                    target = active[j]
                    p = (projected[i] @ projected[j]).sigmoid()
                    scores[(source, target)] = float(p.detach())
                    decisions["edge_probabilities"][f"{source}->{target}"] = float(p.detach())
                    selected, _ = choose(p, f"edge:{source}->{target}", True)
                    if selected:
                        worker_edges.append((source, target))
            n = len(active)
            if n > 1:
                hop_logits = self.hops(h.mean(0))[:n-1]
                simplex = F.gumbel_softmax(hop_logits, tau=1., hard=False) if sample else hop_logits.softmax(0)
                hop_limit = 1+(simplex*torch.arange(1, n, dtype=h.dtype)).sum()
            else:
                simplex, hop_limit = h.new_ones(1), h.new_tensor(1.)

            def critical_path():
                paths = {}
                for r in active:
                    paths[r] = max((paths[a]+[(a,b)] for a,b in worker_edges if b == r), key=len, default=[])
                return max(paths.values(), key=len, default=[])

            path = critical_path()
            penalty = F.relu(hop_limit.new_tensor(len(path))-hop_limit)
            removed = []
            while len(path) > float(hop_limit.detach()):
                edge = min(path, key=lambda e: scores[e])
                worker_edges.remove(edge)
                removed.append(edge)
                path = critical_path()
            decisions.update(hop_limit=float(hop_limit.detach()), hop_probabilities=simplex.detach().tolist(),
                             hop_pruned_edges=removed)
        # Reference allocator supplies a decision backbone separately. Here use
        # the highest configured capability within the selected pool, recorded.
        assigned = {r: assigned[r] for r in active}
        assigned["synthesizer"] = self.config.decision_model or max(pool, key=lambda i: self.config.get(i).quality)
        edges = worker_edges+[(r, "synthesizer") for r in active]
        if topology == "linear":
            edges = worker_edges+[(active[-1], "synthesizer")]
        nodes = [*active, "synthesizer"]
        schedule = dag_waves(nodes, edges)
        cost = sum(self.config.get(assigned[r]).cost(token_reserve(config_messages(self.config, task, r,
            {a: "\U0001f7e6"*self.config.context_chars for a,b in edges if b == r},
            self.instructions)), self.config.max_output_tokens) for r in nodes)
        # Event-driven critical-path estimate, including the decision call.
        durations = {}
        for wave in schedule:
            for r in wave:
                durations[r] = self.config.get(assigned[r]).latency_seconds + max(
                    (durations[a] for a,b in edges if b == r), default=0.)
        latency = durations["synthesizer"]
        decisions.update(ablation=self.config.ablation, difficulty=float(demand.detach()), ease=float(ease.detach()),
                         effective_difficulty=float(effective.detach()), selected_pool=pool_index,
                         pool_probabilities=prob.detach().tolist(), sampled=sample,
                         pruned_agents=[r for r in workers if r not in active],
                         decision_model_rule="configured" if self.config.decision_model else "highest-profile-quality-in-selected-pool",
                         feasible=cost <= budget_usd+1e-12 and latency <= latency_budget_seconds,
                         joint_log_probability=float(torch.stack(terms).sum().detach()))
        return (Plan(topology, "learned", assigned, cost, latency, 0, float(demand.detach()),
                     budget_usd, latency_budget_seconds, tuple(edges), dict(self.instructions), decisions),
                torch.stack(terms).sum(), torch.stack(entropies).sum(), penalty)
