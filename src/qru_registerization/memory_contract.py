"""Clean-work contracts and certified contiguous workspace renewal.

Norms refer to fixed ideal unitary representatives. Certificates are analytic
inequalities evaluated in floating point, without interval arithmetic.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np


@dataclass(frozen=True)
class WorkContract:
    clean_error: float
    leakage: float

    @property
    def channel_bound(self):
        return min(2., 2*self.clean_error+self.leakage)


def clean_contract(k0, ideal):
    """Contract from a unitary dilation's zero-work block."""
    k0, ideal = np.asarray(k0), np.asarray(ideal)
    if k0.shape != ideal.shape or k0.ndim != 2 or k0.shape[0] != k0.shape[1]:
        raise ValueError('square clean and ideal blocks of equal size required')
    if not np.allclose(ideal.conj().T@ideal, np.eye(len(ideal)), atol=1e-10):
        raise ValueError('ideal block must be unitary')
    loss = np.eye(len(k0))-k0.conj().T@k0
    eig = np.linalg.eigvalsh((loss+loss.conj().T)/2)
    if eig[0] < -1e-9:
        raise ValueError('clean block must be a contraction')
    return WorkContract(float(np.linalg.norm(k0-ideal, 2)), float(max(0., eig[-1])))


def shared_bound(contracts):
    """Arbitrary retained-unitary interventions, one initially clean workspace.

    Each call is unitary on the same finite workspace and retained register.
    Interventions act only on the retained register. No input-preservation
    assumption is needed. Distinct estimators and tables are allowed.
    """
    cs = list(contracts)
    if any(c.clean_error < 0 or not 0 <= c.leakage <= 1+1e-9 for c in cs):
        raise ValueError('nonnegative errors and probabilities required')
    roots = np.sqrt([c.leakage for c in cs])
    cross = sum(a*b for a,b in combinations(roots, 2))
    return min(2., 2*sum(c.clean_error for c in cs)+sum(roots**2)+4*cross)


def partitions(length):
    """All contiguous partitions, represented by exclusive segment ends."""
    if length < 1:
        raise ValueError('positive sequence length required')
    for mask in range(1 << (length-1)):
        yield tuple(j for j in range(1,length) if mask & (1 << (j-1)))+(length,)


def partition_bound(ends, contracts):
    start = 0
    total = 0.
    for end in ends:
        total += contracts[start,end].channel_bound
        start = end
    return min(2.,total)


def select_partition_exhaustive(length, contracts, costs, budget):
    """Minimum certified error among contiguous partitions within budget.

    Exhaustive search is intentional for the small exact-channel benchmark.
    The certificate, rather than measured channel error, determines selection.
    """
    candidates=[]
    for ends in partitions(length):
        starts=(0,)+ends[:-1]
        cost=sum(costs[a,b] for a,b in zip(starts,ends))
        if cost <= budget:
            candidates.append((partition_bound(ends,contracts),cost,ends))
    if not candidates:
        raise ValueError('budget admits no partition')
    bound,cost,ends=min(candidates)
    return ends,bound,cost


def select_partition(length, contracts, costs, budget):
    """Exact DAG dynamic program over nonnegative integer costs.

    O(k² Q) transitions before gcd cost compression; pseudo-polynomial in a
    general binary-encoded budget. Equal estimator-pair costs give O(k³)
    transitions. Explicit path tuples add copying/comparison overhead.
    Ties agree with the archived rule: bound, cost, segment ends.
    """
    from functools import reduce
    from math import gcd
    if length < 1 or budget < 0:
        raise ValueError("positive length and nonnegative budget required")
    vals = [costs[a,b] for a in range(length) for b in range(a+1,length+1)]
    if any(not np.isfinite(v) or v < 0 or int(v) != v for v in vals):
        raise ValueError("DP requires nonnegative integer costs")
    unit = reduce(gcd, (int(v) for v in vals)) or 1
    limit = int(budget // unit)
    states = [dict() for _ in range(length+1)]
    states[0][0] = (0., ())
    for end in range(1,length+1):
        for start in range(end):
            cost = int(costs[start,end]) // unit
            error = contracts[start,end].channel_bound
            for used,(bound,ends) in states[start].items():
                total = used+cost
                if total <= limit:
                    candidate=(min(2.,bound+error),ends+(end,))
                    if total not in states[end] or candidate < states[end][total]:
                        states[end][total]=candidate
    if not states[length]:
        raise ValueError("budget admits no partition")
    bound,used,ends=min((bound,used,ends) for used,(bound,ends) in states[length].items())
    if bound == 2.:
        # Once every feasible final certificate saturates, all errors tie.
        # A locally smaller unsaturated error can have discarded a globally
        # preferred lexicographic path. Solve the cost-only DAG in this case.
        cheapest=[None]*(length+1)
        cheapest[0]=(0,())
        for end in range(1,length+1):
            cheapest[end]=min((cheapest[start][0]+int(costs[start,end]),
                               cheapest[start][1]+(end,)) for start in range(end))
        cost,ends=cheapest[length]
        return ends,2.,cost
    return ends,bound,used*unit


def perturbation_budget(contracts, dilation_errors=(), reset_errors=()):
    """Add full-dilation operator errors and decorrelating-reset diamond errors.

    Reset guarantees must hold jointly with every reference; local workspace
    fidelity alone cannot justify decorrelation.
    """
    eta=np.asarray(list(dilation_errors),float)
    reset=np.asarray(list(reset_errors),float)
    if np.any(eta<0) or np.any(reset<0) or not np.all(np.isfinite(np.r_[eta,reset])):
        raise ValueError("finite nonnegative error bounds required")
    return min(2.,sum(c.channel_bound for c in contracts)+2*sum(eta)+sum(reset))
