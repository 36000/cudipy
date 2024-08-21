import numpy as np
import math

from scipy.optimize import minimize

from numba import cuda, float64, float32

from dipy.reconst.mcsd import MSDeconvFit

# GPU implementation of primal-dual Interior Point method
# for convex quadratic constrained optimization (QP)
# Specifically, ||Rx-d||_2 where Ax>=b
# parallelized to solve 10s-100s of thousands of QPs
# ultimately used to fit MSMT CSD


# constants
cp=0.9
max_iter=1000
tol=1e-6
tau=0.95


@cuda.jit(max_registers=64)
def parallel_qp_fit(Rt, R_pinv, G, A, b, x0, y0, l0, data, results, lt_ifx,
                    c, y, l, dx, dy, dl, rhs1, rhs2, Z, schur, cgr, cgp, cgAp):
    '''
    Solves 1/2*x^t*G*x+(Rt*d)^t*x given Ax>=b
    In MSMT, G, R, A, b are the same across voxels,
    but there are different d for different voxels.
    This fact is not used currently. So this is a more general
    batched CLS QP solver.

    Let:
    c=(Rt*d)^t
    L = diag(l)
    Y = diag(y)
    mu as centering parameter that tends to 0 with iteration number.

    Set up with interior points, this is reformulated to:
    | G  0 -A.T | |dx|   |0 |   | G*x-A.T*l+c |
    | A -I  0   | |dy| = |0 | - | A*x-y-b     |
    | 0  L  Y   | |dl|   |mu|   | YL          |

    This reduces to:
    |G -A.T| |dx| = | -G*x-A.T*l+c          |
    |A Y\L | |dl|   | -(A*x-y-b) + (-y+mu/l)|
    with dy = A*dx+(A*x-y-b)

    Solving for dx, dl:
    (G+A.T*(Y\L)*A)*dx = -G*x-A.T*l+c
    dl = (Y\L)*(-(A*x-y-b) + (-y+mu/l) - A*dx)

    So, the tricky part is solving for dx.
    However, note that G+A.T*(Y\L)*A is hermitian positive semidefinite.
    So, it can be solved using conjugate gradients.
    This is done every iteration and is the longest part of the calculation.
    '''
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z

    d = data[bx, by, bz]
    x = results[bx, by, bz]

    c = c[bx, by, bz]
    
    y = y[bx, by, bz]
    l = l[bx, by, bz]
    dx = dx[bx, by, bz]
    dy = dy[bx, by, bz]
    dl = dl[bx, by, bz]
    rhs1 = rhs1[bx, by, bz]
    rhs2 = rhs2[bx, by, bz]
    Z = Z[bx, by, bz]

    schur = schur[bx, by, bz]
    cgr = cgr[bx, by, bz]
    cgp = cgp[bx, by, bz]
    cgAp = cgAp[bx, by, bz]

    mu = 1.0
    m, n = A.shape

    if max(d) == 0:
        for l in block_range(results.shape[3]):
            x[l] = 0.0
        cuda.syncthreads()
        return

    # first check if naive R^+d happens to satisfy the constraints
    for ii in block_range(R_pinv.shape[0]):
        x[ii] = 0.0
        for jj in range(R_pinv.shape[1]):
            x[ii] += R_pinv[ii, jj] * d[jj]
    cuda.syncthreads()

    fails_ineq = 0
    for ii in range(A.shape[0]):
        __tmp = 0.0
        for jj in range(A.shape[1]):
            __tmp += A[ii, jj] * x[jj]
        if (__tmp - b[ii]) < -tol:
            fails_ineq = 1
            break
    cuda.syncthreads()
    if fails_ineq == 0:
        return

    # now calc c
    for ii in block_range(Rt.shape[0]):
        c[ii] = 0.0
        for jj in range(Rt.shape[1]):
            c[ii] += Rt[ii, jj] * d[jj]
    for ii in block_range(n):
        x[ii] = x0[ii]
    for ii in block_range(m):
        y[ii] = y0[ii]
    for ii in block_range(m):
        l[ii] = l0[ii]
    cuda.syncthreads()

    for __iter in range(max_iter):
        if __iter != 0:
            mu *= cp

        # z = l \ y
        for i in block_range(m):
            Z[i] = safe_divide(l[i], y[i])
        cuda.syncthreads()

        # rhs[n:n+m] = -(A.dot(x0) - y0 - b) + (-y + sigma*mu/l)
        for i in block_range(m):
            A_dot_x_i = 0.0
            for j in range(n):
                A_dot_x_i += A[i, j] * x[j]
            
            rp_i = A_dot_x_i - y[i] - b[i]
            rhs2[i] = -rp_i + (-y[i] + safe_divide(mu, l[i]))
            dy[i] = rp_i
        cuda.syncthreads()

        # rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
        for i in block_range(n):
            G_dot_x_i = 0.0
            for j in range(n):
                G_dot_x_i += G[i, j] * x[j]

            A_T_dot_l_i = 0.0
            for j in range(m):
                A_T_dot_l_i += A[j, i] * l[j]

            rhs1[i] = -(G_dot_x_i - A_T_dot_l_i + c[i])

            # rhs1 + A.T@(z*b)
            for j in range(m):
                rhs1[i] += A[j, i]*rhs2[j]*Z[j]
        cuda.syncthreads()

        # dx = np.linalg.solve(G+A.T@np.diag(Z)@A, rhs1)
        for idx in block_range(lt_ifx.shape[0]):
            i, j = lt_ifx[idx]
            __tmp = G[j, i]
            for k in range(m):
                __tmp += A[k, i] * A[k, j] * Z[k]
            schur[j, i] = __tmp
            schur[i, j] = __tmp
        cuda.syncthreads()

        solved_inverse = cg(
            schur, rhs1, dx,
            cgr, cgp, cgAp,
            n, tol)
        cuda.syncthreads()
        if solved_inverse == 0:
            for ii in block_range(n):
                x[ii] = 0
            cuda.syncthreads()
            return

        # dl = z*(rhs2-A@dx)
        for i in block_range(m):
            dl[i] = rhs2[i]
            for j in range(n):
                dl[i] -= A[i, j]*dx[j]
            dl[i] *= Z[i]
        cuda.syncthreads()

        # dy = A@dx+(A.dot(x0) - y0 - b)
        for i in block_range(m):
            for j in range(n):
                dy[i] += A[i, j]*dx[j]
        cuda.syncthreads()

        # calculate step size
        beta = 1.0
        sigma = 1.0
        for ii in range(m):
            if dy[ii] < 0 and dy[ii]*sigma < -y[ii]:
                sigma = -y[ii]/dy[ii]
            if dl[ii] < 0 and dl[ii]*beta < -l[ii]:
                beta = -l[ii]/dl[ii]
        beta *= tau
        sigma *= tau
        alpha = min(beta, sigma)

        cuda.syncthreads()
        # print(np.asarray(x))
        # print(alpha)
        # print(v)

        # time to step
        for ii in block_range(n):
            x[ii] += alpha*dx[ii]
        for ii in block_range(m):
            y[ii] += alpha*dy[ii]
            l[ii] += alpha*dl[ii]
        cuda.syncthreads()

        tdx = 0
        tdy = 0
        tdl = 0
        for ii in range(n):
            tdx += abs(dx[ii])
        for ii in range(m):
            tdy += abs(dy[ii])
            tdl += abs(dl[ii])
        cuda.syncthreads()

        if (alpha*tdx < n*tol) and (alpha*tdy < m*tol) and (alpha*tdl < m*tol):
            return

    # failed to solve
    for ii in block_range(n):
        x[ii] = 0
    cuda.syncthreads()

# @cuda.jit(device=True, inline=True)
# def cholesky_solve(A, b, x, L, n):
#     for ii in block_range(n):
#         for jj in range(n):
#             L[jj, ii] = 0
#     cuda.syncthreads()

#     for jj in range(n):
#         __tmp = math.sqrt(A[jj, jj] - L[jj, jj])
#         if cuda.threadIdx.x == 0:
#             L[jj, jj] = __tmp
#         __tmp = 1.0 / __tmp
#         for ii in block_range_part(jj+1, n):
#             __sum = 0.0
#             for kk in range(jj):
#                 __sum += L[ii, kk] * L[jj, kk]
#             __val = (__tmp * (A[ii, jj] - __sum))
#             L[ii, jj] = __val
#             L[ii, ii] += __val*__val
#         cuda.syncthreads()

#     if cuda.threadIdx.x == 0:
#         for i in range(n):
#             x[i] = b[i]
#             for j in range(i):
#                 x[i] -= L[i, j] * x[j]
#             x[i] /= L[i, i]
#         for i in range(n - 1, -1, -1):
#             for j in range(i + 1, n):
#                 x[i] -= L[j, i] * x[j]
#             x[i] /= L[i, i]
#     cuda.syncthreads()


@cuda.jit(device=True, inline=True)
def cg(A, b, x,
       r, p, Ap,
       n, tol):
    # r = b - np.dot(A, x)
    for ii in block_range(n):
        r[ii] = b[ii]
        for jj in range(n):
            r[ii] -= A[ii, jj] * x[jj]
        r[ii] = safe_divide(r[ii], A[ii, ii])
    cuda.syncthreads()

    # Apply the Jacobi preconditioner: p = M^{-1} * r
    for ii in block_range(n):
        p[ii] = r[ii]
    cuda.syncthreads()

    #rs_old = np.dot(r.T, r)
    rs_old = dotA(r, p, A)

    for _ in range(max_iter):
        # Ap = np.dot(A, p)
        for ii in block_range(n):
            Ap[ii] = 0
            for jj in range(n):
                Ap[ii] += A[ii, jj] * p[jj]
        cuda.syncthreads()

        # alpha = rs_old / np.dot(p.T, Ap)
        alpha = dot(p, Ap)
        alpha = rs_old / alpha

        # x = x + alpha * p
        # r = r - alpha * Ap
        for ii in block_range(n):
            x[ii] += alpha * p[ii]
            r[ii] -= alpha * Ap[ii] / A[ii, ii]
        cuda.syncthreads()

        rs_new = dotA(r, r, A)

        if rs_new < tol:
            return 1

        # p = r + (rs_new / rs_old) * p
        rs_old = rs_new / rs_old
        for ii in block_range(n):
            p[ii] = r[ii] + rs_old * p[ii]
        cuda.syncthreads()

        rs_old = rs_new

    return rs_new * rs_new < tol  # close enough


@cuda.jit(device=True, inline=True)
def safe_divide(a, b):
    if abs(b) < tol:
        if b < 0:
            return -tol
        else:
            return tol
    else:
        return a / b

@cuda.jit(device=True, inline=True)
def dot(x, y):
    rv = 0
    for ii in range(x.shape[0]):
        rv += x[ii]*y[ii]
    cuda.syncthreads()
    return rv

@cuda.jit(device=True, inline=True)
def dotA(x, y, A):
    rv = 0
    for ii in range(x.shape[0]):
        rv += x[ii]*y[ii]*A[ii, ii]
    cuda.syncthreads()
    return rv

@cuda.jit(device=True, inline=True)
def block_range(__stop):
    '''
    Assumes blocks are of shape (x, 1, 1)
    '''
    return range(cuda.threadIdx.x, __stop, cuda.blockDim.x)

# @cuda.jit(device=True, inline=True)
# def block_range_part(__start, __stop):
#     return range(__start+cuda.threadIdx.x, __stop, cuda.blockDim.x)

def find_analytic_center(A, b, x0):
    """Find the analytic center using scipy.optimize.minimize"""
    m, n = A.shape

    cons = [{'type': 'ineq', 'fun': lambda x: A @ x - b}]

    result = minimize(
        lambda x: 1e-3*np.linalg.norm(x)**2-np.sum(np.log(A @ x - b)), x0,
        constraints=cons, method='SLSQP')

    return result.x

def init_point(Q, A, b, x0):
    m, n = A.shape

    l0 = np.ones(m)
    y0 = np.ones(m)

    M = np.zeros((n+m+m, n+m+m))
    M[:n, :n] = Q
    M[:n, n+m:] = -A.T
    M[n:n+m, :n] = A
    M[n:n+m, n:n+m] = -np.eye(m)
    M[n+m:, n:n+m] = np.diag(l0)
    M[n+m:, n+m:] = np.diag(y0)

    rhs_vec = np.empty(n+m+m)
    rhs_vec[:n] = -(Q.dot(x0) - A.T.dot(l0))
    rhs_vec[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs_vec[n+m:] = -(y0 * l0)

    solution = np.linalg.solve(M, rhs_vec)
    dx = solution[:n]
    dy = solution[n:n+m]
    dl = solution[n+m:]
    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0 + dl))

    return y0, l0

def gen_lt_idx(n):
    return np.vstack(np.tril_indices(n)).T

def fit(self, data):
    m, n = self.fitter._reg.shape
    coeff = np.zeros((*data.shape[:3], n))

    R = self.fitter._X
    A = self.fitter._reg
    b = np.zeros(A.shape[0])

    for i in range(A.shape[0]):
        A[i, :] /= np.linalg.norm(A[i, :])

    Q = R.T @ R
    x0 = np.linalg.pinv(A) @ np.ones(A.shape[0])
    x0 = find_analytic_center(A, b, x0)
    y0, l0 = init_point(Q, A, b, x0)
    
    Rt = cuda.to_device(-R.T)
    R_pinv = cuda.to_device(np.linalg.pinv(R))
    Q = cuda.to_device(Q)
    A = cuda.to_device(A)
    b = cuda.to_device(b)
    x0 = cuda.to_device(x0)
    y0 = cuda.to_device(y0)
    l0 = cuda.to_device(l0)

    c = cuda.device_array((*data.shape[:3], n))
    y = cuda.device_array((*data.shape[:3], m))
    l = cuda.device_array((*data.shape[:3], m))
    dx = cuda.device_array((*data.shape[:3], n))
    dy = cuda.device_array((*data.shape[:3], m))
    dl = cuda.device_array((*data.shape[:3], m))
    rhs1 = cuda.device_array((*data.shape[:3], n))
    rhs2 = cuda.device_array((*data.shape[:3], m))
    Z = cuda.device_array((*data.shape[:3], m))

    schur = cuda.device_array((*data.shape[:3], n, n))
    cgr = cuda.device_array((*data.shape[:3], n))
    cgp = cuda.device_array((*data.shape[:3], n))
    cgAp = cuda.device_array((*data.shape[:3], n))

    d = cuda.device_array((*data.shape[:3], m))
    x = cuda.device_array((*data.shape[:3], m))

    data = cuda.to_device(data)
    coeff = cuda.to_device(coeff)
    lt_ifx = cuda.to_device(gen_lt_idx(n))

    parallel_qp_fit[
    #    (2, 2, 2), 64,
        data.shape[:3], 64,
        0, 0](
            Rt, R_pinv, Q, A, b, x0, y0, l0, data, coeff, lt_ifx, 
            c, y, l, dx, dy, dl, rhs1, rhs2, Z, schur, cgr, cgp, cgAp)

    cuda.current_context().synchronize()
    coeff = coeff.copy_to_host()
    
    return MSDeconvFit(self, coeff, None)
