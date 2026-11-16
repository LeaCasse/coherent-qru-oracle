"""Fast exact isometry evaluation of a fused numeric-score service.

Only the zero-work input columns are propagated, using FFTs and two-by-two
query powers. No dense estimator or Choi matrix is constructed. This is a
classical validation algorithm, not a claim that powered queries are free.
"""
import numpy as np
from scipy.linalg import block_diag
from .interface import phase_taper_amplitudes,directional_amplitude_unitary
from .qru import qru_unitary,rz
from .certification import generator_rotations,sine_variance_bound


def target_protocol(values,pulses,interventions=None):
    dim=np.asarray(pulses[0][0]).shape[0];eye=np.eye(dim)
    gates=[eye]*(len(pulses)-1) if interventions is None else list(interventions)
    f=np.broadcast_to(eye,(len(values),dim,dim)).copy()
    for j,(h,gamma) in enumerate(pulses):
        f=generator_rotations(h,gamma,values)@f
        if j<len(gates):f=gates[j]@f
    return f


def fused_numeric_kraus(params,xs,axis,phase_bits,pulses,interventions=None,initialization="prepared",fibre=None):
    if initialization not in ('prepared','zero'):raise ValueError("invalid initialization")
    n=2**phase_bits;t=np.asarray(pulses[0][0]).shape[0];m=len(xs)
    window=phase_taper_amplitudes(phase_bits,'sine');z=np.diag([1.,-1.])
    values=-np.cos(2*np.pi*np.arange(n)/n)/np.cos(np.pi/n)
    controls=target_protocol(values,pulses,interventions)
    phases=np.zeros(m) if fibre is None else np.asarray(fibre,float)
    ks=np.zeros((2*n,m*t,m*t),complex);scores=[]
    v=-window.copy();v[0]+=1;denominator=float(v@v)
    for x,(datum,phi) in enumerate(zip(xs,phases,strict=True)):
        a=directional_amplitude_unitary(qru_unitary(params,float(datum))@rz(float(phi)),axis)
        scores.append(float(-np.vdot(a[:,0],z@a[:,0]).real))
        initial=a if initialization=='prepared' else np.eye(2)
        q=a@z@a.conj().T@z
        powers=np.empty((n,2,2),complex);powers[0]=np.eye(2)
        for j in range(1,n):powers[j]=q@powers[j-1]
        forward=np.einsum('nij,j->ni',powers,initial[:,0])*window[:,None]
        estimated=np.fft.fft(forward,axis=0)/np.sqrt(n)
        controlled=estimated[:,:,None,None]*controls[:,None,:,:]
        back_fourier=np.fft.ifft(controlled,axis=0)*np.sqrt(n)
        reversed_queries=np.einsum('nji,njkl->nikl',powers.conj(),back_fourier)
        reversed_window=reversed_queries-2*v[:,None,None,None]*np.einsum('n,nijk->ijk',v,reversed_queries)[None,:,:,:]/denominator
        output=np.einsum('ji,njkl->nikl',initial.conj(),reversed_window)
        ks[:,x*t:(x+1)*t,x*t:(x+1)*t]=output.reshape(2*n,t,t)
    ideal=block_diag(*target_protocol(np.asarray(scores),pulses,interventions))
    return ks,ideal,np.asarray(scores)


def choose_phase_bits(angular_budget,channel_tolerance,literal_tolerance=None,max_bits=14):
    """Uniform calibrated-sine certificate, independent of QRU scores."""
    for m in range(2,max_bits+1):
        variance=sine_variance_bound(m)
        if 2*angular_budget**2*variance<=channel_tolerance and (literal_tolerance is None or abs(angular_budget)*np.sqrt(variance)<=literal_tolerance):return m
    raise ValueError("precision exceeds declared maximum phase width")
