"""Input-preserving certificates without reconstructing a channel or dilation.

The exact branch route enumerates the declared finite input set. The uniform
moment route avoids this enumeration. Floating-point evaluation is explicit;
these routines do not claim interval-arithmetic certification of roundoff.
"""
from dataclasses import dataclass
import numpy as np
from .interface import phase_taper_amplitudes
from .memory_contract import WorkContract


def score_word_laws(scores, phase_bits, calibrated=True):
    scores=np.asarray(scores,float)
    if scores.ndim!=1 or not len(scores) or np.any(abs(scores)>1+1e-10):
        raise ValueError("a nonempty vector of scores in [-1,1] is required")
    n=2**phase_bits
    if phase_bits<2:raise ValueError("sine calibration requires at least two bits")
    amps=phase_taper_amplitudes(phase_bits,"sine")
    omega=np.arccos(np.clip(-scores,-1,1))
    positive=abs(np.fft.fft(amps[None,:]*np.exp(1j*omega[:,None]*np.arange(n)),axis=1))**2/n
    negative=abs(np.fft.fft(amps[None,:]*np.exp(-1j*omega[:,None]*np.arange(n)),axis=1))**2/n
    laws=(positive+negative)/2
    values=-np.cos(2*np.pi*np.arange(n)/n)
    if calibrated:values/=np.cos(np.pi/n)
    return values,laws


def generator_rotations(h,angle,values):
    h=np.asarray(h,complex)
    if h.ndim!=2 or h.shape[0]!=h.shape[1] or not np.allclose(h,h.conj().T,atol=1e-12):
        raise ValueError("Hermitian target generator required")
    eigen,vectors=np.linalg.eigh(h)
    if max(abs(eigen))>1+1e-10:raise ValueError("target generator norm must be at most one")
    phase=np.exp(-1j*angle*np.asarray(values)[:,None]*eigen[None,:])
    return np.einsum('ai,ni,bi->nab',vectors,phase,vectors.conj())


@dataclass
class BranchCertificate:
    contract: WorkContract
    literal_isometry_error: float
    zero_blocks: np.ndarray
    ideal_blocks: np.ndarray


def branch_certificate(blocks,ideal):
    blocks=np.asarray(blocks,complex);ideal=np.asarray(ideal,complex)
    if blocks.shape!=ideal.shape or blocks.ndim!=3:raise ValueError("branch matrices must have matching shapes")
    dim=blocks.shape[-1];eye=np.eye(dim)
    gram=blocks.conj().transpose(0,2,1)@blocks
    error=float(np.max(np.linalg.svd(blocks-ideal,compute_uv=False)))
    leakage=max(0.,float(np.max(np.linalg.eigvalsh(eye-gram))))
    overlap=blocks.conj().transpose(0,2,1)@ideal
    literal=max(0.,float(np.max(np.linalg.eigvalsh(2*eye-overlap-overlap.conj().transpose(0,2,1)))))**.5
    return BranchCertificate(WorkContract(error,leakage),literal,blocks,ideal)


def preserved_segment_certificates(scores,phase_bits,pulses,interventions=None):
    """All O(k²) segment contracts via batched target matrices.

    pulses: (Hermitian target matrix with norm <=1, angle).
    Interventions are common target-only unitaries. Complexity is polynomial
    in branch count M, N=2**m and target dimension, not in their logarithms.
    No process_channel, estimator_matrix, Choi or SDP is invoked.
    """
    k=len(pulses)
    if not k:raise ValueError("at least one pulse required")
    dim=np.asarray(pulses[0][0]).shape[0];eye=np.eye(dim)
    gates=[eye]*(k-1) if interventions is None else list(interventions)
    if len(gates)!=k-1:raise ValueError("one target intervention between pulses required")
    for gate in gates:
        if np.shape(gate)!=(dim,dim) or not np.allclose(np.asarray(gate).conj().T@gate,eye,atol=1e-10):
            raise ValueError("target interventions must be unitaries")
    values,laws=score_word_laws(scores,phase_bits)
    rotations=[generator_rotations(h,gamma,values) for h,gamma in pulses]
    exact=[generator_rotations(h,gamma,np.asarray(scores)) for h,gamma in pulses]
    contracts={};diagnostics={}
    for start in range(k):
        f=np.broadcast_to(eye,(len(values),dim,dim)).copy()
        target=np.broadcast_to(eye,(len(scores),dim,dim)).copy()
        for j in range(start,k):
            f=rotations[j]@f;target=exact[j]@target
            blocks=np.einsum('xy,yab->xab',laws,f)
            certificate=branch_certificate(blocks,target)
            contracts[start,j+1]=certificate.contract
            diagnostics[start,j+1]=certificate
            if j<k-1:f=gates[j]@f;target=gates[j]@target
    return contracts,diagnostics


def sine_variance_bound(phase_bits):
    if phase_bits<2:raise ValueError("at least two phase bits required")
    kappa=np.cos(np.pi/2**phase_bits)
    return float(2*(1-kappa)/kappa**2)


def uniform_moment_certificate(phase_bits,pulses,interventions=None,grid_points=513):
    """Uniform in input branches, with derivative-grid interpolation margins.

    F^(r) has norm <= A**r, A=sum |gamma|. Hence a derivative grid with
    spacing h bounds sup||F'|| and sup||F''|| by observed maxima plus
    A² h/2 and A³ h/2. The remainder is analytic, not an unchecked grid fit.
    """
    if grid_points<2:raise ValueError("at least two grid points required")
    dim=np.asarray(pulses[0][0]).shape[0];eye=np.eye(dim);k=len(pulses)
    gates=[eye]*(k-1) if interventions is None else list(interventions)
    if len(gates)!=k-1:raise ValueError("incorrect intervention count")
    extent=1/np.cos(np.pi/2**phase_bits)
    values=np.linspace(-extent,extent,grid_points)
    f=np.broadcast_to(eye,(grid_points,dim,dim)).copy()
    first=np.zeros_like(f);second=np.zeros_like(f)
    for j,(h,gamma) in enumerate(pulses):
        u=generator_rotations(h,gamma,values);g=-1j*gamma*np.asarray(h)
        du=g@u;ddu=g@du
        second=ddu@f+2*du@first+u@second
        first=du@f+u@first;f=u@f
        if j<k-1:f=gates[j]@f;first=gates[j]@first;second=gates[j]@second
    a=sum(abs(gamma) for _,gamma in pulses);spacing=2*extent/(grid_points-1)
    l1=float(np.max(np.linalg.svd(first,compute_uv=False)))+a*a*spacing/2
    l2=float(np.max(np.linalg.svd(second,compute_uv=False)))+a**3*spacing/2
    variance=sine_variance_bound(phase_bits)
    return {"certificate":min(2.,(l2+l1*l1)*variance),"l1_upper":l1,"l2_upper":l2,
            "variance_upper":variance,"grid_spacing":spacing,"interpolation_margin_l1":a*a*spacing/2,
            "interpolation_margin_l2":a**3*spacing/2,"enumerates_input_branches":False}
