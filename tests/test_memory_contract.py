"""Composition contracts against independently expanded unitary dilations."""
import numpy as np
import pytest
from scipy.linalg import expm
from qru_registerization.memory_contract import clean_contract,shared_bound,select_partition,partitions,partition_bound,WorkContract
from qru_registerization.process import diamond_choi,kraus_choi


def test_general_interventions_and_distinct_dilations():
    rng=np.random.default_rng(5613);d=2;e=3
    for _ in range(4):
        product=np.eye(e*d,dtype=complex);ideal=np.eye(d,dtype=complex);contracts=[]
        for j in range(3):
            h=rng.normal(size=(e*d,e*d))+1j*rng.normal(size=(e*d,e*d));h=(h+h.conj().T)/2
            w=expm(-.03j*h)
            q,_=np.linalg.qr(rng.normal(size=(d,d))+1j*rng.normal(size=(d,d)))
            contracts.append(clean_contract(w[:d,:d],np.eye(d)))
            product=w@np.kron(np.eye(e),q)@product;ideal=q@ideal
        ks=product[:,:d].reshape(e,d,d);v=ideal.ravel(order='F')
        error=diamond_choi(kraus_choi(ks)-np.outer(v,v.conj()))
        assert error['diamond_upper']<=shared_bound(contracts)+1e-6


def test_quadratic_order_and_leading_constant():
    x=np.array([[0,1],[1,0]]);z=np.diag([1,-1]);k=4
    ratios=[]
    for delta in [.02,.01,.005]:
        w=expm(-1j*delta*np.kron(x,z));c=clean_contract(w[:2,:2],np.eye(2))
        exact=2*np.sin(k*delta)**2
        assert exact<=shared_bound([c]*k)+1e-12
        ratios.append(exact/shared_bound([c]*k))
    assert ratios[-1]>.999 and ratios[-1]>ratios[0]


def test_budget_selection_uses_certificates():
    n=4;cs={};cost={}
    for a in range(n):
        for b in range(a+1,n+1):
            cs[a,b]=WorkContract(.01*(b-a)**2,0.);cost[a,b]=10
    assert len(list(partitions(n)))==8
    ends,bound,used=select_partition(n,cs,cost,20)
    assert ends==(2,4) and used==20 and np.isclose(bound,.16)
    assert bound==min(partition_bound(p,cs) for p in partitions(n) if len(p)<=2)
    with pytest.raises(ValueError):select_partition(n,cs,cost,9)


def test_invalid_clean_contract():
    with pytest.raises(ValueError):clean_contract(2*np.eye(2),np.eye(2))
