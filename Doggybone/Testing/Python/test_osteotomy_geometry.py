import importlib.util, numpy as np
import os
P=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','DoggyboneLib','osteotomy_logic.py')
spec=importlib.util.spec_from_file_location('ol',P); ol=importlib.util.module_from_spec(spec); spec.loader.exec_module(ol)

def apply(M,p):
    H=ol.matrix_to_numpy(M); q=np.r_[p,1.0]; return (H@q)[:3]

# Pure torsion: every point on shaft line must stay fixed even with a remote hinge.
shaft_p=np.array([10.,20.,30.]); shaft=np.array([0.,0.,1.]); hinge=np.array([55.,20.,30.]); ap=np.array([0.,1.,0.])
M=ol.anatomical_correction_matrix(shaft_p,shaft,27.,hinge,ap,0.)
for t in (-100.,0.,85.):
    p=shaft_p+t*shaft
    assert np.linalg.norm(apply(M,p)-p)<1e-9
# Old shared-pivot geometry would move shaft_p when hinge is remote; verify new one does not.
assert np.linalg.norm(apply(M,shaft_p)-shaft_p)<1e-9

# Pure wedge: every point on hinge/AP line remains fixed.
M=ol.anatomical_correction_matrix(shaft_p,shaft,0.,hinge,ap,13.)
for t in (-40.,0.,60.):
    p=hinge+t*ap
    assert np.linalg.norm(apply(M,p)-p)<1e-9

# Full correction is rigid: rotation block remains orthonormal with det +1.
M=ol.anatomical_correction_matrix(shaft_p,shaft,31.,hinge,ap,-17.)
R=ol.matrix_to_numpy(M)[:3,:3]
assert np.allclose(R.T@R,np.eye(3),atol=1e-10)
assert abs(np.linalg.det(R)-1.0)<1e-10
print('All osteotomy geometry tests passed.')
