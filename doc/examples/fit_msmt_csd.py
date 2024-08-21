import os
import os.path as op
import numpy as np
import cupy as cp

from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.data import get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.reconst.mcsd import (
    MultiShellDeconvModel,
    mask_for_response_msmt,
    multi_shell_fiber_response,
    response_from_mask_msmt,
)

from cudipy.segment.mask import median_otsu
from cudipy.segment.tissue import TissueClassifierHMRF
from cudipy.reconst.quadratic_program import fit as fit_gpu
from cudipy.reconst.shm import QballModel, anisotropic_power


import nibabel as nib

# inspired from: https://github.com/dipy/dipy/blob/master/doc/examples/reconst_mcsd.py 

fraw, fbval, fbvec, t1_fname = get_fnames("cfin_multib")

sphere = get_sphere("symmetric724")

data, affine = load_nifti(fraw)
bvals, bvecs = read_bvals_bvecs(fbval, fbvec)
gtab = gradient_table(bvals, bvecs)

bvals = gtab.bvals
bvecs = gtab.bvecs

sel_b = np.logical_or(np.logical_or(bvals == 0, bvals == 1000), bvals == 2000)
data = data[..., sel_b]
data_cp = cp.asarray(data)

gtab = gradient_table(bvals[sel_b], bvecs[sel_b])


_, mask = median_otsu(data_cp, median_radius=2, numpass=1, vol_idx=[0, 1])

# Use Qball to get 3T seg
qball_model = QballModel(gtab, 8)
shm_coeff = qball_model.fit(data_cp, mask=mask)
ap = anisotropic_power(shm_coeff)
nclass = 3
beta = 0.1
hmrf = TissueClassifierHMRF()
_, final_segmentation, PVE = hmrf.classify(ap, nclass, beta)
final_segmentation = final_segmentation.get()

csf = np.where(final_segmentation == 1, 1, 0)
gm = np.where(final_segmentation == 2, 1, 0)
wm = np.where(final_segmentation == 3, 1, 0)
print(f"HMRF CSF volume size: {np.sum(csf)}")
print(f"HMRF GM volume size: {np.sum(gm)}")
print(f"HMRF WM volume size: {np.sum(wm)}")

# get responses for each T
mask_wm, mask_gm, mask_csf = mask_for_response_msmt(
    gtab,
    data,
    roi_radii=10,
    wm_fa_thr=0.7,
    gm_fa_thr=0.3,
    csf_fa_thr=0.15,
    gm_md_thr=0.001,
    csf_md_thr=0.0032,
)
mask_wm *= wm
mask_gm *= gm
mask_csf *= csf
nvoxels_wm = np.sum(mask_wm)
nvoxels_gm = np.sum(mask_gm)
nvoxels_csf = np.sum(mask_csf)
print(f"Final CSF volume size: {nvoxels_csf}")
print(f"Final GM volume size: {nvoxels_gm}")
print(f"Final WM volume size: {nvoxels_wm}")

response_wm, response_gm, response_csf = response_from_mask_msmt(
    gtab, data, mask_wm, mask_gm, mask_csf
)
print("Responses")
print(response_wm)
print(response_gm)
print(response_csf)

ubvals = unique_bvals_tolerance(gtab.bvals)
response_mcsd = multi_shell_fiber_response(
    8,
    ubvals,
    response_wm,
    response_gm,
    response_csf,
)

mcsd_model = MultiShellDeconvModel(gtab, response_mcsd)
data_cp[~mask] = 0
print("Solving QPs, this could take some time...")
mcsd_fit = fit_gpu(mcsd_model, data_cp)
print("Done!")

nib.save(nib.Nifti1Image(mcsd_fit.shm_coeff, affine), "msmt_warp_shm.nii.gz")
print(np.histogram(mcsd_fit.shm_coeff[..., 0]))
