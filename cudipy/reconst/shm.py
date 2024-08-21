from dipy.reconst.shm import QballModel as QballModelCPU
import cupy as cp

def calculate_max_order(n_coeffs):
    r"""Calculate the maximal harmonic order (l), given that you know the
    number of parameters that were estimated."""
    L1 = (-3 + cp.sqrt(1 + 8 * n_coeffs)) / 2.0
    if L1 == cp.floor(L1) and not cp.mod(L1, 2):
        return int(L1)

def anisotropic_power(sh_coeffs, norm_factor=0.00001, power=2, non_negative=True):
    r"""Calculate anisotropic power map with a given SH coefficient matrix."""
    
    dim = sh_coeffs.shape[:-1]
    n_coeffs = sh_coeffs.shape[-1]
    max_order = calculate_max_order(n_coeffs)
    ap = cp.zeros(dim)
    n_start = 1
    
    for L in range(2, max_order + 2, 2):
        n_stop = n_start + (2 * L + 1)
        ap_i = cp.mean(cp.abs(sh_coeffs[..., n_start:n_stop]) ** power, -1)
        ap += ap_i
        n_start = n_stop

    if ap.ndim < 1:
        ap = cp.reshape(ap, (1,))
        
    log_ap = cp.zeros_like(ap)
    positive_mask = ap > 0
    log_ap[positive_mask] = cp.log(ap[positive_mask]) - cp.log(norm_factor)

    if non_negative:
        log_ap = cp.maximum(log_ap, 0)
        
    return log_ap

def normalize_data(data, where_b0, min_signal=1e-5):
    """Normalizes the data with respect to the mean b0"""
    out = cp.array(data, dtype="float32", copy=True)
    out = cp.clip(out, min_signal, None)
    b0 = out[..., where_b0].mean(-1)
    out /= b0[..., None]
    return out

class QballModel(QballModelCPU):
    """Implementation of regularized Qball reconstruction method.

    References
    ----------
    .. [1] Descoteaux, M., et al. 2007. Regularized, fast, and robust
           analytical Q-ball imaging.
    """
    def _get_shm_coef(self, data, mask=None):
        """Returns the coefficients of the model"""
        return cp.dot(data[..., self._where_dwi], cp.asarray(self._fit_matrix.T))

    def fit(self, data, mask=None):
        """Fits the model to diffusion data and returns the model fit"""
        # Normalize the data and fit coefficients
        if not self.assume_normed:
            data = normalize_data(data, self._where_b0s, self.min_signal)

        # Compute coefficients using abstract method
        coef = self._get_shm_coef(data)

        # Apply the mask to the coefficients
        if mask is not None:
            mask = cp.asarray(mask, dtype=bool)
            coef *= mask[..., None]
        return coef