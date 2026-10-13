"""Check the paper's protocol-level moment and exact-return identities."""
import numpy as np
from qru_registerization.qru import random_qru_params
from qru_registerization.process import process_channel, process_metrics, rotation

def test_noncommuting_protocol_average_and_channel_bound():
    p=random_qru_params(2,seed=604,scale=.6)
    pulses=[([0,0,1],.8),([1,0,0],.8)]
    r=process_channel(p,[-.7,.9],[.3,-.4,np.sqrt(.75)],pulses,phase_bits=4)
    for x in range(2):
        mean=sum(weight*(rotation([1,0,0],.8*g)@rotation([0,0,1],.8*g)) for weight,g in zip(r.distributions[x],r.estimates))
        np.testing.assert_allclose(r.zero_kraus[2*x:2*x+2,2*x:2*x+2],mean,atol=1e-12)
    delta=r.estimates[None,:]-r.scores[:,None]
    bias=np.max(abs(np.sum(r.distributions*delta,axis=1)))
    mu=np.max(np.sum(r.distributions*delta**2,axis=1))
    variance=np.max(np.sum(r.distributions*r.estimates[None,:]**2,axis=1)-(r.distributions@r.estimates)**2)
    budget=1.6
    bound=min(2.,2*budget*bias+budget**2*(mu+variance))
    assert process_metrics(r)['diamond_upper']<=bound+1e-5

def test_exact_return_identity_for_nonbasis_target():
    p=random_qru_params(2,seed=604,scale=.6)
    r=process_channel(p,[-.7,.9],[.3,-.4,np.sqrt(.75)],[([0,0,1],.8),([1,0,0],.8)],phase_bits=3)
    target=np.array([1,1j])/np.sqrt(2)
    for x in range(2):
        v=np.zeros(4,complex);v[2*x:2*x+2]=target
        dilation_error=sum(np.linalg.norm(k@v-(r.ideal@v if a==0 else 0))**2 for a,k in enumerate(r.kraus))
        ideal=rotation([1,0,0],.8*r.scores[x])@rotation([0,0,1],.8*r.scores[x])
        expected=sum(weight*np.linalg.norm((rotation([1,0,0],.8*g)@rotation([0,0,1],.8*g)-ideal)@target)**2 for weight,g in zip(r.distributions[x],r.estimates))
        np.testing.assert_allclose(dilation_error,expected,atol=1e-12)
