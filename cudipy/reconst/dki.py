import numpy as np
import cupy as cp

from dipy.reconst.dki import DiffusionKurtosisFit

def dki_gpu_fit(self, data):
    data_thres = np.maximum(data, self.min_signal)

    # Set up least squares problem
    _lt_indices = cp.array([[0, 1, 3], [1, 2, 4], [3, 4, 5]])
    A = cp.array(self.design_matrix)
    A_inv = cp.array(self.inverse_design_matrix)
    y = cp.array(np.log(data_thres))

    # DKI ordinary linear least square solution
    result = cp.dot(A_inv, y)

    # Define weights as diag(yn**2)
    if self.weights:
        W = cp.diag(cp.exp(2 * cp.dot(A, result)))
        AT_W = cp.dot(A.T, W)
        inv_AT_W_A = cp.linalg.pinv(cp.dot(AT_W, A))
        AT_W_LS = cp.dot(AT_W, y)
        result = cp.dot(inv_AT_W_A, AT_W_LS)

    # Write output
    # Extracting the diffusion tensor parameters from solution
    DT_elements = result[:6]
    DT_elements[..., _lt_indices]

    eigenvals, eigenvecs = cp.linalg.eigh(DT_elements)

    if eigenvals.ndim == 1:
        # this is a lot faster when dealing with a single voxel
        order = eigenvals.argsort()[::-1]
        eigenvecs = eigenvecs[:, order]
        eigenvals = eigenvals[order]
    else:
        # temporarily flatten eigenvals and eigenvecs to make sorting easier
        shape = eigenvals.shape[:-1]
        eigenvals = eigenvals.reshape(-1, 3)
        eigenvecs = eigenvecs.reshape(-1, 3, 3)
        size = eigenvals.shape[0]
        order = eigenvals.argsort()[:, ::-1]
        xi, yi = cp.ogrid[:size, :3, :3][:2]
        eigenvecs = eigenvecs[xi, yi, order[:, None, :]]
        xi = cp.ogrid[:size, :3][0]
        eigenvals = eigenvals[xi, order]
        eigenvecs = eigenvecs.reshape(shape + (3, 3))
        eigenvals = eigenvals.reshape(shape + (3,))
    eigenvals = eigenvals.clip(min=self.min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = eigenvals.mean(0) ** 2
    KT_elements = result[6:21] / MD_square if MD_square else 0.0 * result[6:21]
    S0 = cp.exp(-result[[-1]])

    # Write output
    dki_params = cp.concatenate(
        (eigenvals, eigenvecs[0], eigenvecs[1], eigenvecs[2], KT_elements, S0), axis=0
    ).get()
    params = dki_params[..., 0:-1]
    if self.return_S0_hat:
        S0_params = dki_params[..., -1]
    else:
        S0_params = None

    return DiffusionKurtosisFit(self, params, model_S0=S0_params)
