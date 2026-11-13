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


def select_partition(length, contracts, costs, budget):
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
