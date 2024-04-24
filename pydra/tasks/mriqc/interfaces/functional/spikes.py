from fileformats.generic import File
import logging
import nibabel as nb
import numpy as np
from os import path as op
from pathlib import Path
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate(
    {"return": {"out_tsz": File, "out_spikes": File, "num_spikes": int}}
)
def Spikes(
    in_file: File,
    in_mask: File,
    invert_mask: bool,
    no_zscore: bool,
    detrend: bool,
    spike_thresh: float,
    skip_frames: int,
    out_tsz: Path,
    out_spikes: Path,
) -> ty.Tuple[File, File, int]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.functional.spikes import Spikes

    """
    func_nii = nb.load(in_file)
    func_data = func_nii.get_fdata()
    func_shape = func_data.shape
    ntsteps = func_shape[-1]
    tr = func_nii.header.get_zooms()[-1]
    nskip = skip_frames

    if detrend:
        from nilearn.signal import clean

        data = func_data.reshape(-1, ntsteps)
        clean_data = clean(data[:, nskip:].T, t_r=tr, standardize=False).T
        new_shape = (
            func_shape[0],
            func_shape[1],
            func_shape[2],
            clean_data.shape[-1],
        )
        func_data = np.zeros(func_shape)
        func_data[..., nskip:] = clean_data.reshape(new_shape)

    mask_data = np.bool_(nb.load(in_mask).dataobj)
    mask_data[..., :nskip] = 0
    mask_data = np.stack([mask_data] * ntsteps, axis=-1)

    if not invert_mask:
        brain = np.ma.array(func_data, mask=(mask_data != 1))
    else:
        mask_data[..., :skip_frames] = 1
        brain = np.ma.array(func_data, mask=(mask_data == 1))

    if no_zscore:
        ts_z = find_peaks(brain)
        total_spikes = []
    else:
        total_spikes, ts_z = find_spikes(brain, spike_thresh)
    total_spikes = list(set(total_spikes))

    out_tsz = op.abspath(out_tsz)
    out_tsz = out_tsz
    np.savetxt(out_tsz, ts_z)

    out_spikes = op.abspath(out_spikes)
    out_spikes = out_spikes
    np.savetxt(out_spikes, total_spikes)
    num_spikes = len(total_spikes)

    return out_tsz, out_spikes, num_spikes


# Nipype methods converted into functions


def _robust_zscore(data):
    return (data - np.atleast_2d(np.median(data, axis=1)).T) / np.atleast_2d(
        data.std(axis=1)
    ).T


def find_peaks(data):
    t_z = [data[:, :, i, :].mean(axis=0).mean(axis=0) for i in range(data.shape[2])]
    return t_z


def find_spikes(data, spike_thresh):
    data -= np.median(np.median(np.median(data, axis=0), axis=0), axis=0)
    slice_mean = np.median(np.median(data, axis=0), axis=0)
    t_z = _robust_zscore(slice_mean)
    spikes = np.abs(t_z) > spike_thresh
    spike_inds = np.transpose(spikes.nonzero())
    # mask out the spikes and recompute z-scores using variance uncontaminated with spikes.
    # This will catch smaller spikes that may have been swamped by big
    # ones.
    data.mask[:, :, spike_inds[:, 0], spike_inds[:, 1]] = True
    slice_mean2 = np.median(np.median(data, axis=0), axis=0)
    t_z = _robust_zscore(slice_mean2)

    spikes = np.logical_or(spikes, np.abs(t_z) > spike_thresh)
    spike_inds = [tuple(i) for i in np.transpose(spikes.nonzero())]
    return spike_inds, t_z
