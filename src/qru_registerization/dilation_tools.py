"""Architecture-independent finite channel validation for explicit dilations."""
import numpy as np
from scipy.linalg import block_diag
from .process import kraus_choi,choi_from_superoperator,diamond_choi
from .channel import schur_diamond,trace_norm
from .memory_contract import clean_contract
from .certification import branch_certificate


def branch_kraus(dilations,target_dimension):
    """Input-preserving dilations in (environment,target) order per branch."""
    matrices=np.asarray(dilations,complex);m=len(matrices);t=target_dimension
    env=matrices.shape[1]//t
    blocks=matrices.reshape(m,env,t,env,t)
    return np.array([block_diag(*blocks[:,a,:,0,:]) for a in range(env)])


def sequence_channel(dilations,ideals,target_dimension,interventions=None,mode="shared"):
    """Retain input coherences; interventions may mix input and target.

    Every environment outcome contributes. fresh composes reduced channels;
    shared propagates the full isometry through the same environment.
    """
    matrices=[np.asarray(w,complex) for w in dilations]
    if not matrices or mode not in ("shared","fresh"):raise ValueError("invalid sequence")
    m=len(matrices[0]);t=target_dimension;d=m*t;env=matrices[0].shape[-1]//t
    gates=[np.eye(d)]*(len(matrices)-1) if interventions is None else list(interventions)
    if len(gates)!=len(matrices)-1:raise ValueError("incorrect intervention count")
    ideal=np.eye(d,dtype=complex);zero=np.eye(d,dtype=complex)
    superoperator=np.eye(d*d,dtype=complex)
    iso=np.zeros((m,env,t,d),complex)
    for x in range(m):iso[x,0,:,x*t:(x+1)*t]=np.eye(t)
    for j,(w,ideal_step) in enumerate(zip(matrices,ideals,strict=True)):
        if mode=="shared":
            for x in range(m):iso[x]=(w[x]@iso[x].reshape(env*t,d)).reshape(env,t,d)
        else:
            ks=branch_kraus(w,t)
            superoperator=sum(np.kron(k.conj(),k) for k in ks)@superoperator
            zero=ks[0]@zero
        ideal=ideal_step@ideal
        if j<len(gates):
            r=np.asarray(gates[j]);ideal=r@ideal
            if mode=="shared":
                retained=iso.transpose(0,2,1,3).reshape(d,env*d)
                iso=(r@retained).reshape(m,t,env,d).transpose(0,2,1,3)
            else:superoperator=np.kron(r.conj(),r)@superoperator;zero=r@zero
    if mode=="shared":
        ks=iso.transpose(1,0,2,3).reshape(env,d,d)
        return kraus_choi(ks),ks[0],ideal,ks
    return choi_from_superoperator(superoperator),zero,ideal,None


def channel_metrics(choi,zero,ideal,kraus=None,diamond=False):
    d=len(ideal);vec=ideal.ravel(order='F');delta=choi-np.outer(vec,vec.conj())
    contract=clean_contract(zero,ideal)
    cross=zero.conj().T@ideal
    literal=max(0.,float(np.linalg.eigvalsh(2*np.eye(d)-cross-cross.conj().T)[-1]))**.5
    output={"choi_error":trace_norm(delta)/d,"clean_error":contract.clean_error,
       "leakage":contract.leakage,"certificate":contract.channel_bound,"literal_error":literal}
    if diamond:
        if kraus is not None and max(np.max(abs(k-np.diag(np.diag(k)))) for k in kraus)<1e-10 and np.max(abs(ideal-np.diag(np.diag(ideal))))<1e-10:
            diagonal=np.array([np.diag(k) for k in kraus]).T
            gram=diagonal@diagonal.conj().T;target=np.diag(ideal)
            output.update(schur_diamond(gram-np.outer(target,target.conj())))
        else:output.update(diamond_choi(delta))
    return output


def circuit_channel(circuit,layout):
    """Independent statevector extraction in Qiskit's little-endian order."""
    from qiskit.quantum_info import Statevector
    r=len(layout['input']);m=2**r;tbit=layout['target'];work=layout['work']
    env=2**len(work);ks=np.zeros((env,2*m,2*m),complex)
    work_indices=np.array([sum(((a>>j)&1)<<q for j,q in enumerate(work)) for a in range(env)])
    for x in range(m):
        for t in range(2):
            state=np.zeros(2**circuit.num_qubits,complex);state[x+(t<<tbit)]=1
            out=Statevector(state).evolve(circuit).data
            for xo in range(m):
                for to in range(2):ks[:,2*xo+to,2*x+t]=out[xo+(to<<tbit)+work_indices]
    return kraus_choi(ks),ks


def unitary_segment_certificates(dilations,ideals,target_dimension,target_interventions=None):
    """All preserved-input segment contracts by propagating clean columns only.

    Applies to QSP as well as QPE and uses no Choi matrix, Kraus stack or SDP.
    Work dimension is explicit. Common target-only interventions are required.
    """
    steps=[np.asarray(w,complex) for w in dilations];k=len(steps)
    m=len(steps[0]);t=target_dimension;size=steps[0].shape[-1]
    if size%t:raise ValueError("incompatible work and target dimensions")
    gates=[np.eye(t)]*(k-1) if target_interventions is None else list(target_interventions)
    if len(gates)!=k-1:raise ValueError("incorrect target intervention count")
    initial=np.zeros((m,size,t),complex);initial[:,:t,:]=np.eye(t)
    local_ideals=[np.array([u[x*t:(x+1)*t,x*t:(x+1)*t] for x in range(m)]) for u in ideals]
    contracts={}
    for a in range(k):
        iso=initial.copy();ideal=np.broadcast_to(np.eye(t),(m,t,t)).copy()
        for j in range(a,k):
            iso=steps[j]@iso;ideal=local_ideals[j]@ideal
            contracts[a,j+1]=branch_certificate(iso[:,:t,:],ideal).contract
            if j<k-1:
                lifted=np.kron(np.eye(size//t),gates[j])
                iso=lifted@iso;ideal=gates[j]@ideal
    return contracts
