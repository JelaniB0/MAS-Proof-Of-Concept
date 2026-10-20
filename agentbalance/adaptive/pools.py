"""Deterministic Pareto filtering and k-medoids over configured model profiles."""

import math


def profile(model):
    # A documented common workload: 1,000 input and 500 output tokens.
    return (model.quality, model.calibration_cost_usd if model.calibration_cost_usd is not None
            else model.cost(1000, 500), model.latency_seconds)


def pareto_models(models):
    def dominates(a, b):
        aq, ac, al = profile(a)
        bq, bc, bl = profile(b)
        return aq >= bq and ac <= bc and al <= bl and (aq > bq or ac < bc or al < bl)
    return tuple(m for m in models if not any(dominates(other, m) for other in models))


def construct_pools(models, count=None, min_size=1, max_size=None):
    retained = pareto_models(models)
    if not retained:
        raise ValueError("No models available for pool construction")
    count = min(4, max(1, len(retained) // 2)) if count is None else count
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= len(retained):
        raise ValueError("Pool count must be between 1 and the number of Pareto models")
    if isinstance(min_size, bool) or not isinstance(min_size, int) or not 1 <= min_size <= len(retained):
        raise ValueError("Minimum pool size exceeds the available Pareto models; recalibrate profiles or reduce pool_min_size")
    if max_size is not None and (isinstance(max_size, bool) or not isinstance(max_size, int) or max_size < min_size):
        raise ValueError("Maximum pool size must be an integer at least the minimum size")
    raw = [(q, math.log(max(c, 1e-9)), math.log(max(t, 1e-6)))
           for q, c, t in map(profile, retained)]
    low = [min(row[j] for row in raw) for j in range(3)]
    span = [max(row[j] for row in raw) - low[j] for j in range(3)]
    points = [tuple((row[j] - low[j]) / (span[j] or 1) for j in range(3)) for row in raw]
    distance = lambda i, j: sum((a - b) ** 2 for a, b in zip(points[i], points[j]))
    medoids = [min(range(len(raw)), key=lambda i: (raw[i][0], retained[i].id))]
    while len(medoids) < count:
        medoids.append(max((i for i in range(len(raw)) if i not in medoids),
                           key=lambda i: (min(distance(i, j) for j in medoids), -i)))
    for _ in range(100):
        # Medoids own themselves, including identical profile points.
        clusters = [[i for i in range(len(raw)) if
                     (i if i in medoids else min(medoids, key=lambda j: (distance(i, j), j))) == m]
                    for m in medoids]
        updated = [min(cluster, key=lambda i: (sum(distance(i, j) for j in cluster), i))
                   for cluster in clusters]
        if updated == medoids:
            break
        medoids = updated
    # Supplement undersized pools with nearest retained neighbors. Membership may
    # overlap, so small candidate sets can exercise pool selection AND matching.
    for cluster, medoid in zip(clusters, medoids):
        if max_size is not None:
            cluster[:] = sorted(cluster, key=lambda i: (distance(i, medoid), retained[i].id))[:max_size]
        neighbors = sorted((i for i in range(len(retained)) if i not in cluster),
                           key=lambda i: (distance(i, medoid), retained[i].id))
        cluster.extend(neighbors[:max(0, min_size - len(cluster))])
        cluster.sort()
    pools = [tuple(retained[i].id for i in cluster) for cluster in clusters]
    lookup = {m.id: m for m in retained}
    pools.sort(key=lambda pool: (sum(lookup[i].quality for i in pool) / len(pool),
                                sum(profile(lookup[i])[1] for i in pool) / len(pool)))
    return pools, [m.id for m in models if m not in retained]
