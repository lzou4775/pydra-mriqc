from fileformats.generic import File
import logging
from pydra.tasks.mriqc.utils.misc import _flatten_dict
import nibabel as nb
import numpy as np
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate(
    {
        "return": {
            "bdiffs": dict,
            "efc": dict,
            "fa_degenerate": float,
            "fa_nans": float,
            "fber": dict,
            "fd": dict,
            "ndc": float,
            "sigma": dict,
            "spikes": dict,
            "snr_cc": dict,
            "summary": dict,
            "out_qc": dict,
        }
    }
)
def DiffusionQC(
    in_file: File,
    in_b0: File,
    in_shells: ty.List[File],
    in_shells_bval: list,
    in_bval_file: File,
    in_bvec: list,
    in_bvec_rotated: list,
    in_bvec_diff: list,
    in_fa: File,
    in_fa_nans: File,
    in_fa_degenerate: File,
    in_cfa: File,
    in_md: File,
    brain_mask: File,
    wm_mask: File,
    cc_mask: File,
    spikes_mask: File,
    noise_floor: float,
    direction: ty.Any,
    in_fd: File,
    fd_thres: float,
    in_fwhm: list,
    qspace_neighbors: list,
    piesno_sigma: float,
) -> ty.Tuple[
    dict, dict, float, float, dict, dict, float, dict, dict, dict, dict, dict
]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.diffusion.diffusion_qc import DiffusionQC

    """
    from mriqc.qc import anatomical as aqc
    from mriqc.qc import diffusion as dqc

    # from mriqc.qc import functional as fqc

    # Get the mean EPI data and get it ready
    b0nii = nb.load(in_b0)
    b0data = np.round(
        np.nan_to_num(np.asanyarray(b0nii.dataobj)),
        3,
    )
    b0data[b0data < 0] = 0

    # Get the FA data and get it ready OE: enable when used
    # fanii = nb.load(in_fa)
    # fadata = np.round(
    #     np.nan_to_num(np.asanyarray(fanii.dataobj)),
    #     3,
    # )

    # Get brain mask data
    msknii = nb.load(brain_mask)
    mskdata = np.round(  # Protect the thresholding with a rounding for stability
        msknii.get_fdata(),
        3,
    )
    if np.sum(mskdata) < 100:
        raise RuntimeError(
            "Detected less than 100 voxels belonging to the brain mask. "
            "MRIQC failed to process this dataset."
        )

    # Get wm mask data
    wmnii = nb.load(wm_mask)
    wmdata = np.round(  # Protect the thresholding with a rounding for stability
        np.asanyarray(wmnii.dataobj),
        3,
    )

    # Get cc mask data
    ccnii = nb.load(cc_mask)
    ccdata = np.round(  # Protect the thresholding with a rounding for stability
        np.asanyarray(ccnii.dataobj),
        3,
    )

    # Get DWI data after splitting them by shell (DSI's data is clustered)
    shelldata = [
        np.round(
            np.asanyarray(nb.load(s).dataobj),
            4,
        )
        for s in in_shells
    ]

    # Summary stats
    rois = {
        "fg": mskdata,
        "bg": 1.0 - mskdata,
        "wm": wmdata,
    }
    stats = aqc.summary_stats(b0data, rois)
    summary = stats

    # CC mask SNR and std
    snr_cc, cc_sigma = dqc.cc_snr(
        in_b0=b0data,
        dwi_shells=shelldata,
        cc_mask=ccdata,
        b_values=in_shells_bval,
        b_vectors=in_bvec,
    )

    fa_nans_mask = np.asanyarray(nb.load(in_fa_nans).dataobj) > 0.0
    fa_nans = round(float(1e6 * fa_nans_mask[mskdata > 0.5].mean()), 2)

    fa_degenerate_mask = np.asanyarray(nb.load(in_fa_degenerate).dataobj) > 0.0
    fa_degenerate = round(
        float(1e6 * fa_degenerate_mask[mskdata > 0.5].mean()),
        2,
    )

    # Get spikes-mask data
    spmask = np.asanyarray(nb.load(spikes_mask).dataobj) > 0.0
    spikes = dqc.spike_ppm(spmask)

    # FBER
    fber = {
        f"shell{i + 1:02d}": aqc.fber(bdata, mskdata.astype(np.uint8))
        for i, bdata in enumerate(shelldata)
    }

    # EFC
    efc = {f"shell{i + 1:02d}": aqc.efc(bdata) for i, bdata in enumerate(shelldata)}

    # FD
    fd_data = np.loadtxt(in_fd, skiprows=1)
    num_fd = (fd_data > fd_thres).sum()
    fd = {
        "mean": round(float(fd_data.mean()), 4),
        "num": int(num_fd),
        "perc": float(num_fd * 100 / (len(fd_data) + 1)),
    }

    # NDC
    dwidata = np.round(
        np.nan_to_num(nb.load(in_file).get_fdata()),
        3,
    )
    ndc = dqc.neighboring_dwi_correlation(
        dwidata,
        neighbor_indices=qspace_neighbors,
        mask=mskdata > 0.5,
    )

    # Sigmas
    sigma = {
        "cc": round(float(cc_sigma), 4),
        "piesno": round(piesno_sigma, 4),
        "pca": round(noise_floor, 4),
    }

    # rotated b-vecs deviations
    diffs = np.array(in_bvec_diff)
    bdiffs = {
        "mean": round(float(diffs[diffs > 1e-4].mean()), 4),
        "median": round(float(np.median(diffs[diffs > 1e-4])), 4),
        "max": round(float(diffs[diffs > 1e-4].max()), 4),
        "min": round(float(diffs[diffs > 1e-4].min()), 4),
    }

    out_qc = _flatten_dict(self._results)

    return (
        bdiffs,
        efc,
        fa_degenerate,
        fa_nans,
        fber,
        fd,
        ndc,
        sigma,
        spikes,
        snr_cc,
        summary,
        out_qc,
    )


# Nipype methods converted into functions
