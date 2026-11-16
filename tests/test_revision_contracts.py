import numpy as np
import pytest
from scipy.linalg import block_diag
from qiskit.quantum_info import Operator
from qru_registerization.qru import random_qru_params,qru_unitary,rz
from qru_registerization.interface import directional_amplitude_unitary,phase_taper_amplitudes
from qru_registerization.structural import affine_estimator
from qru_registerization.channel import estimator_matrix
from qru_registerization.process import process_channel
from qru_registerization.process import diamond_choi
from qru_registerization.memory_contract import select_partition,select_partition_exhaustive,WorkContract
from qru_registerization.certification import preserved_segment_certificates,uniform_moment_certificate
from qru_registerization.direct_qsp import make_gqsp_plan,gqsp_matrix,branch_dilation,gqsp_score_service
from qru_registerization.dilation_tools import branch_kraus,circuit_channel,sequence_channel,unitary_segment_certificates
from qru_registerization.numeric_service import fused_numeric_kraus

AXIS=[.3,-.4,np.sqrt(.75)];Z=np.diag([1.,-1.]);X=np.array([[0.,1.],[1.,0.]])
PARAMS=random_qru_params(2,seed=604,scale=.6)


@pytest.mark.parametrize('m',[2,3])
def test_reflections_preserve_entire_dilation(m):
    kw=dict(phase_bits=m,taper='sine',fibre_step=1.3)
    old=affine_estimator(PARAMS,1,-.7,1.6,AXIS,**kw)
    new=affine_estimator(PARAMS,1,-.7,1.6,AXIS,query_implementation='reflections',**kw)
    np.testing.assert_allclose(Operator(old).data,Operator(new).data,atol=3e-12)


def test_zero_initialization_word_law_and_fibre_invariance():
    amps=phase_taper_amplitudes(3,'sine')
    a=directional_amplitude_unitary(qru_unitary(PARAMS,.9),AXIS)
    e=estimator_matrix(a,amps);f=estimator_matrix(a,amps,'zero')
    np.testing.assert_allclose(np.sum(abs(e[:,0].reshape(8,2))**2,axis=1),np.sum(abs(f[:,0].reshape(8,2))**2,axis=1),atol=1e-12)
    np.testing.assert_allclose(f,estimator_matrix(a@rz(2.1),amps,'zero'),atol=2e-12)


def test_dp_matches_exhaustive_for_general_costs_and_saturation():
    rng=np.random.default_rng(916)
    for n in [3,5,7]:
        for scale in [.02,2.]:
            contracts={(a,b):WorkContract(float(rng.uniform(0,scale)),float(rng.uniform(0,.2))) for a in range(n) for b in range(a+1,n+1)}
            costs={ab:int(rng.integers(1,5)) for ab in contracts}
            for budget in range(costs[0,n],20):
                new=select_partition(n,contracts,costs,budget);old=select_partition_exhaustive(n,contracts,costs,budget)
                assert new[0]==old[0] and new[2]==old[2]
                assert new[1]==pytest.approx(old[1],abs=1e-12)
    # Different partial errors become tied only after a later segment caps
    # the full bound. The discarded higher-error path is lexicographically first.
    contracts={(a,b):WorkContract(1.,0.) for a in range(3) for b in range(a+1,4)}
    contracts[0,1]=WorkContract(.05,0.)
    contracts[1,2]=WorkContract(.05,0.)
    contracts[0,2]=WorkContract(.05,0.)
    costs={ab:100 for ab in contracts}
    for ab in [(0,1),(1,2),(2,3)]:costs[ab]=1
    costs[0,2]=2
    assert select_partition(3,contracts,costs,3)==select_partition_exhaustive(3,contracts,costs,3)==((1,2,3),2.,3)


def test_branch_contracts_match_independent_dense_noncommuting_channels():
    scores=np.array([process_channel(PARAMS,[x],AXIS,[([0,0,1],.3)],phase_bits=3).scores[0] for x in [-.7,.9]])
    pulses=[(Z,.3),(X,.4),(Z,-.2)]
    contracts,diag=preserved_segment_certificates(scores,3,pulses)
    for a in range(3):
        for b in range(a+1,4):
            axes=[([0,0,1] if np.allclose(h,Z) else [1,0,0],g) for h,g in pulses[a:b]]
            result=process_channel(PARAMS,[-.7,.9],AXIS,axes,phase_bits=3)
            np.testing.assert_allclose(block_diag(*diag[a,b].zero_blocks),result.zero_kraus,atol=3e-12)
            uniform=uniform_moment_certificate(3,pulses[a:b],grid_points=129)
            assert contracts[a,b].channel_bound<=uniform['certificate']+1e-10


@pytest.mark.parametrize('gamma',[.2,.8,1.6,6.4])
def test_completed_gqsp_matches_polynomial_and_declared_contract(gamma):
    plan=make_gqsp_plan(gamma,.001)
    angles=np.linspace(0,2*np.pi,257)
    for theta in angles:
        u=np.array([[np.exp(1j*theta)]])
        actual=gqsp_matrix(u,plan)[0,0]
        target=np.exp(1j*gamma*np.cos(theta))
        assert abs(actual-target)<=plan.scalar_error_upper+1e-12
    assert plan.channel_error_upper<=.001


def test_gqsp_expanded_circuit_matches_all_kraus_and_fibre_invariance():
    plan=make_gqsp_plan(.8,.01);ws=[]
    for x in [-.7,.9]:
        a=directional_amplitude_unitary(qru_unitary(PARAMS,x),AXIS)
        ws.append(branch_dilation(a,plan))
    expected=branch_kraus(ws,2)
    c,layout=gqsp_score_service(PARAMS,1,-.7,1.6,AXIS,plan)
    _,actual=circuit_channel(c,layout)
    np.testing.assert_allclose(actual,expected,atol=3e-11)
    changed,l=gqsp_score_service(PARAMS,1,-.7,1.6,AXIS,plan,fibre_step=2.1)
    _,ks=circuit_channel(changed,l)
    np.testing.assert_allclose(ks,actual,atol=3e-11)


@pytest.mark.parametrize('initialization',['prepared','zero'])
def test_fft_isometry_matches_dense_estimator_with_interventions(initialization):
    gates=[np.kron(np.eye(2),np.array([[0,1],[1,0]]))]
    dense=process_channel(PARAMS,[-.7,.9],AXIS,[([0,0,1],.3),([1,0,0],.4)],gates,phase_bits=3,initialization=initialization,fibre=[0,1.3])
    ks,ideal,_=fused_numeric_kraus(PARAMS,[-.7,.9],AXIS,3,[(Z,.3),(X,.4)],[X],initialization=initialization,fibre=[0,1.3])
    np.testing.assert_allclose(ks,dense.kraus,atol=3e-12)
    np.testing.assert_allclose(ideal,dense.ideal,atol=1e-12)


def test_memory_bounded_sdp_solver_has_correct_full_norm():
    d=4;v=np.eye(d).ravel(order='F')
    result=diamond_choi(np.diag(v)-np.outer(v,v),solver='SCS')
    assert abs(result['diamond_lower']-1.5)<2e-5
    assert result['diamond_upper']>=result['diamond_lower']-1e-8


def test_workspace_phase_keeps_local_channel_but_changes_shared_context():
    from qru_registerization.process import kraus_choi
    from qru_registerization.certification import generator_rotations
    plan=make_gqsp_plan(.8/32,.01/32)
    amps=[directional_amplitude_unitary(qru_unitary(PARAMS,x),AXIS) for x in [-.7,.9]]
    steps=np.array([branch_dilation(a,plan) for a in amps]);kick=np.kron(Z,np.eye(4))
    changed=np.array([kick@w for w in steps])
    np.testing.assert_allclose(kraus_choi(branch_kraus(steps,2)),kraus_choi(branch_kraus(changed,2)),atol=1e-13)
    scores=np.array([-np.vdot(a[:,0],Z@a[:,0]).real for a in amps])
    ideal=block_diag(*generator_rotations(Z,.8/32,scores))
    original=sequence_channel([steps]*32,[ideal]*32,2)
    phased=sequence_channel([changed]*32,[ideal]*32,2)
    assert np.linalg.norm(original[0]-phased[0])>1e-4
    contracts=unitary_segment_certificates([changed]*4,[ideal]*4,2)
    dense=sequence_channel([changed]*4,[ideal]*4,2)
    from qru_registerization.memory_contract import clean_contract
    direct=clean_contract(dense[1],dense[2])
    assert contracts[0,4].channel_bound==pytest.approx(direct.channel_bound,abs=1e-12)
