"""Independent matrix checks; no simulator package or experimental rerun."""
import json
import os
from pathlib import Path
import numpy as np
I=np.eye(2,dtype=complex);X=np.array([[0,1],[1,0]],complex);Y=np.array([[0,-1j],[1j,0]]);Z=np.diag([1,-1]).astype(complex);H=np.array([[1,1],[1,-1]],complex)/np.sqrt(2)
def rot(p,t):return np.cos(t/2)*I-1j*np.sin(t/2)*p
def embed(u,q,n):
 out=np.zeros((2**n,2**n),complex)
 for c in range(2**n):
  bit=(c>>q)&1
  for b in [0,1]:out[(c&~(1<<q))|(b<<q),c]=u[b,bit]
 return out
def cx(c,t,n):
 out=np.zeros((2**n,2**n),complex)
 for k in range(2**n):out[k^(1<<t) if (k>>c)&1 else k,k]=1
 return out
rng=np.random.default_rng(1337);errors=[];load=[];rzerr=[]
for _ in range(16):
 a,b,c,d=rng.normal(size=4);A=rot(Z,c)@rot(Y,b)@rot(X,a)
 Q=-A@(-Z)@A.conj().T@Z
 def ctrl(u):return np.block([[I,np.zeros((2,2))],[np.zeros((2,2)),u]])
 phase=np.kron(np.diag([1,-1]),I)
 controlled=phase@ctrl(A)@ctrl(X)@ctrl(Z)@ctrl(X)@ctrl(A.conj().T)@ctrl(Z)
 desired=np.block([[I,np.zeros((2,2))],[np.zeros((2,2)),Q]])
 errors.append(float(np.max(abs(controlled-desired))))
 assert np.max(abs(X@Z@X+Z))<1e-12
 base,step=rng.normal(size=2)
 for v in range(4):
  v0,v1=v&1,(v>>1)&1
  structural=rot(Z,c)@rot(Y,2*b*step*v1)@rot(Y,b*step*v0)@rot(Y,b*base+d)@rot(X,a)
  literal=rot(Z,c)@rot(Y,b*(base+step*(v0+2*v1))+d)@rot(X,a)
  load.append(float(np.max(abs(structural-literal))))
 g=float(rng.normal());gamma=.8
 rzerr.append(float(np.max(abs(rot(Z,2*gamma*g)-np.diag(np.exp(-1j*gamma*g*np.array([1,-1])))))))
state=np.eye(8,dtype=complex)[:,0]
for u,q in [(X,0),(H,1),(Z,1)]:state=embed(u,q,3)@state
state=cx(1,2,3)@state
n=np.arange(8);F=np.exp(2j*np.pi*np.outer(n,n)/8)/np.sqrt(8);state=F@state
for u,q in [(H,2),(X,2)]:state=embed(u,q,3)@state
sine=np.sqrt(2/4)*np.sin(np.pi*np.arange(4)/4);target=np.r_[sine,np.zeros(4)]
phase=np.vdot(target,state);sine_error=float(np.max(abs(state-phase*target)))
result={'controlled_grover_error':max(errors),'reflection_error':float(np.max(abs(X@Z@X+Z))),
        'affine_layer_error':max(load),'rotation_angle_error':max(rzerr),
        'sine_state_error_up_to_global_phase':sine_error,'sine_ancilla_one_probability':float(np.sum(abs(state[4:])**2)),
        'source':'independent NumPy matrix checks of depicted gate order; not a rerun of paper experiments'}
assert max(v for v in result.values() if isinstance(v,float))<1e-12
(Path(__file__).resolve().parents[1]/'outputs/circuit_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
