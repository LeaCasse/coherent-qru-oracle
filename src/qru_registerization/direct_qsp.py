"""Complete, measurement-free GQSP comparator for exp(-i gamma s_x Z_T).

References: Motlagh & Wiebe, arXiv:2308.01501; Berntson & Suenderhauf,
arXiv:2406.04246. Rotations follow the GQSP convention also documented by
PennyLane 0.45.1. This NumPy implementation needs no PennyLane dependency.

Q=A Z A† Z has (Q+Q†)/2=-s I for a single system qubit. For U=Q tensor Z_T,
exp(i gamma (U+U†)/2) is exactly the desired retained action tensor I_S.
A bounded Laurent approximation is completed and recursively synthesized.
All completion outcomes are retained and traced; there is no postselection.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import lgamma
import time
import numpy as np
from numpy.polynomial import Polynomial
from scipy.special import jv
from scipy.linalg import block_diag


def bessel_tail_bound(gamma,degree):
    """2 sum_{n>d}|J_n(gamma)|: finite absolute sum plus analytic remainder.

    |J_n(t)| <= (|t|/2)^n/n! for real t. Bounds are evaluated in floating
    point, with an explicit small numerical margin, not interval arithmetic.
    """
    t=abs(float(gamma));last=max(degree+32,int(t)+40)
    finite=2*float(np.sum(abs(jv(np.arange(degree+1,last+1),t))))
    if t==0:return 0.
    first=last+1;ratio=t/(2*(first+1))
    remainder=2*np.exp(first*np.log(t/2)-lgamma(first+1))/(1-ratio)
    return finite+remainder+2e-15


def signal_rotation(theta,phi,lambd):
    return np.array([[np.exp(1j*(lambd+phi))*np.cos(theta),np.exp(1j*phi)*np.sin(theta)],
                     [np.exp(1j*lambd)*np.sin(theta),-np.cos(theta)]],complex)


def complementary_polynomial(p):
    """Spectral factor of 1-|p|² on the circle; audited after synthesis.

    Root finding is adequate for the deliberately moderate degrees studied.
    Coefficient residuals are checked rather than trusting root selection.
    """
    p=np.asarray(p,complex);degree=len(p)-1
    r=-np.convolve(p,p[::-1].conj());r[degree]+=1
    roots=Polynomial(r).roots()
    inside=roots[np.argsort(abs(roots))[:degree]]
    q=Polynomial.fromroots(inside).coef
    desired_energy=1-float(np.vdot(p,p).real)
    q*=np.sqrt(max(0.,desired_energy)/float(np.vdot(q,q).real))
    return q


def synthesize_rotations(p):
    q=complementary_polynomial(p);work=np.array([p,q],complex)
    angles=np.zeros((3,len(p)))
    for idx in range(len(p)-1,-1,-1):
        a,b=work[:,idx]
        theta=np.arctan2(abs(b),abs(a))
        phi=np.angle(a)-np.angle(b) if abs(b)>1e-15 else np.angle(a)
        lambd=np.angle(b) if idx==0 and abs(b)>1e-15 else 0.
        angles[:,idx]=theta,phi,lambd
        if idx:
            reduced=signal_rotation(theta,phi,0.).conj().T@work
            work=np.array([reduced[0,1:idx+1],reduced[1,:idx]])
    return angles


def implemented_polynomials(angles):
    r=signal_rotation(*angles[:,0]);p=np.array([r[0,0]]);q=np.array([r[1,0]])
    for col in angles[:,1:].T:
        r=signal_rotation(*col);zp=np.r_[0,p];qq=np.r_[q,0]
        p,q=r[0,0]*zp+r[0,1]*qq,r[1,0]*zp+r[1,1]*qq
    return p,q


@dataclass
class GQSPPlan:
    gamma: float
    degree: int
    angles: np.ndarray
    polynomial: np.ndarray
    implemented: np.ndarray
    analytic_tail: float
    normalization: float
    coefficient_residual: float
    completion_residual: float
    scalar_error_upper: float
    synthesis_seconds: float

    @property
    def channel_error_upper(self):return min(2.,4*self.scalar_error_upper)

    @property
    def literal_error_upper(self):return min(2.,np.sqrt(2*self.scalar_error_upper))

    @property
    def preparation_calls(self):return 6*self.degree

    def metadata(self):
        return {key:getattr(self,key) for key in ['gamma','degree','analytic_tail','normalization',
            'coefficient_residual','completion_residual','scalar_error_upper','synthesis_seconds']}|{
            'channel_error_upper':self.channel_error_upper,'literal_error_upper':self.literal_error_upper,
            'a_calls':self.preparation_calls,'controlled_signal_calls':2*self.degree,'uncontrolled_inverse_signal_calls':self.degree}


@lru_cache(maxsize=128)
def make_gqsp_plan(gamma,channel_tolerance,literal_tolerance=None):
    """Choose degree from a uniform contract, independently of QRU instances.

    Channel tolerance refers to the full retained-channel diamond norm.
    Optional literal tolerance bounds the zero-work isometry error.
    """
    if not np.isfinite(gamma) or gamma==0:raise ValueError("nonzero finite gamma required")
    if not 0<channel_tolerance<2:raise ValueError("channel tolerance must lie in (0,2)")
    target=channel_tolerance/4
    if literal_tolerance is not None:
        if literal_tolerance<=0:raise ValueError("positive literal tolerance required")
        target=min(target,literal_tolerance**2/2)
    started=time.perf_counter()
    degree=next(d for d in range(1,65) if bessel_tail_bound(gamma,d)<target/16)
    tail=bessel_tail_bound(gamma,degree)
    headroom=target/4;normalization=1+tail+headroom
    coeff=np.array([(1j)**int(n)*jv(int(n),gamma) for n in range(-degree,degree+1)])/normalization
    angles=synthesize_rotations(coeff)
    actual,q=implemented_polynomials(angles)
    residual=float(np.sum(abs(actual-coeff)))
    completion=np.convolve(actual,actual[::-1].conj())+np.convolve(q,q[::-1].conj())
    completion[len(actual)-1]-=1
    completion_error=float(np.sum(abs(completion)))
    bound=tail+(normalization-1)*(1+tail)/normalization+residual+2e-12
    if bound>target:
        raise RuntimeError(f"GQSP synthesis failed declared contract: {bound} > {target}; degree={degree}")
    return GQSPPlan(float(gamma),degree,angles,coeff,actual,tail,normalization,residual,
                    completion_error,float(bound),time.perf_counter()-started)


def gqsp_matrix(signal,plan):
    signal=np.asarray(signal,complex);dim=len(signal);eye=np.eye(dim)
    if not np.allclose(signal.conj().T@signal,eye,atol=1e-10):raise ValueError("unitary signal required")
    w=np.kron(signal_rotation(*plan.angles[:,0]),eye)
    controlled=block_diag(signal,eye)
    for col in plan.angles[:,1:].T:w=np.kron(signal_rotation(*col),eye)@controlled@w
    return np.kron(np.eye(2),np.linalg.matrix_power(signal.conj().T,plan.degree))@w


def branch_dilation(amplitude,plan):
    z=np.diag([1,-1]);q=amplitude@z@amplitude.conj().T@z
    return gqsp_matrix(np.kron(q,z),plan)


def gqsp_score_service(params,input_bits,x0,step,axis,plan,fibre_step=0.):
    """Gate-expanded service: no score lookup, no powered signal gates.

    Qiskit layout input bits, S, T, extra GQSP ancilla. All local signal
    rotations are arbitrary one-qubit gates; signal queries use A/A† and CZ.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from .structural import affine_input_oracle
    a=affine_input_oracle(params,input_bits,x0,step,axis,fibre_step=fibre_step)
    system=input_bits;target=system+1;signal=target+1
    circuit=QuantumCircuit(signal+1,name="GQSP_score_service")
    amap=[*range(input_bits),system]
    def rotate(col):circuit.append(UnitaryGate(signal_rotation(*col)),[signal])
    rotate(plan.angles[:,0])
    for col in plan.angles[:,1:].T:
        circuit.x(signal)
        circuit.cz(signal,system);circuit.compose(a.inverse(),qubits=amap,inplace=True)
        circuit.cz(signal,system);circuit.compose(a,qubits=amap,inplace=True)
        circuit.cz(signal,target);circuit.x(signal)
        rotate(col)
    for _ in range(plan.degree):
        circuit.compose(a.inverse(),qubits=amap,inplace=True);circuit.z(system)
        circuit.compose(a,qubits=amap,inplace=True);circuit.z(system);circuit.z(target)
    return circuit,{"input":list(range(input_bits)),"target":target,"work":[system,signal],"system":system}
