import numpy as np
import scipy as sp
import igl
import math
from scipy.sparse import csr_matrix

def get_uniform_laplacian_1(faces):
    A = igl.adjacency_matrix(faces)
    N = np.asarray(A.sum(axis=1)).reshape(-1)
    D = sp.sparse.diags(N)

    N_1 = np.divide(np.ones_like(N).astype(np.float32), N.astype(np.float32), out=np.zeros_like(N).astype(np.float32), where=N!=0)
    
    D_inv = sp.sparse.diags(np.divide(np.ones_like(N).astype(np.float32), N.astype(np.float32), out=np.ones_like(N).astype(np.float32), where=N!=0))
    
    return (D-A) @ D_inv
    # return D - A

def get_uniform_laplacian_1int(faces):
    A = igl.adjacency_matrix(faces)
    N = np.asarray(A.sum(axis=1)).reshape(-1)
    D = sp.sparse.diags(N)
    return D - A


def get_uniform_laplacian_theta(V, F):
    """
    Get uniform cotangent laplacian
    
    In the paper, denoted as L_u
    """
    A = igl.adjacency_matrix(F)
    N = np.asarray(A.sum(axis=0)).reshape(-1)
    D = sp.sparse.diags(N)

    theta = sp.sparse.diags(np.reciprocal(np.tan(math.pi * (0.5 - np.reciprocal(N.astype(np.float32))))))
    
    return theta @ (A-D)


class LapEditor:
    def __init__(self, Lf, v, picked_points = np.array([0, 1])):
        n_v = Lf.shape[0]
        n_p = picked_points.shape[0]
        A_v = np.zeros((n_p, n_v))
        for i, p in enumerate(picked_points):
            A_v[i, p] = 1
        A_v = csr_matrix(A_v)

        A = sp.sparse.vstack([Lf, A_v]).tocsr()
        b = np.zeros((n_v + n_p, 3))
        for i, p in enumerate(picked_points):
            b[n_v + i, :] = v[p, :]
        AT = A.transpose()
        ATA = AT @ A

        self.n_v = n_v
        self.b = b
        self.AT = AT
        # sp.sparse.linalg.use_solver(use_umfpack=False)
        self.solve = sp.sparse.linalg.factorized(ATA)

    def update_anchor(self, v, picked_points):
        for i, p in enumerate(picked_points):
            self.b[self.n_v + i, :] = v[p, :]

    def recon(self, delta):
        self.b[:self.n_v] = delta
        rhs = self.AT @ self.b
        v_new = self.solve(rhs)
        return v_new


