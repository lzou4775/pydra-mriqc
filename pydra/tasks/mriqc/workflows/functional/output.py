import attrs
import logging
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.specs import BaseSpec, SpecInfo
from pydra.engine.task import FunctionTask
import pydra.mark
from pydra.tasks.mriqc.interfaces import DerivativesDataSink
import typing as ty


logger = logging.getLogger(__name__)


def init_func_report_wf(
    brainmask=attrs.NOTHING,
    epi_mean=attrs.NOTHING,
    epi_parc=attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    fd_thres=attrs.NOTHING,
    hmc_epi=attrs.NOTHING,
    hmc_fd=attrs.NOTHING,
    in_dvars=attrs.NOTHING,
    in_fft=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    in_spikes=attrs.NOTHING,
    in_stddev=attrs.NOTHING,
    meta_sidecar=attrs.NOTHING,
    mni_report=attrs.NOTHING,
    name="func_report_wf",
    name_source=attrs.NOTHING,
    outliers=attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fft_spikes_detector=False,
    wf_species="human",
):
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.functional.output import init_func_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_func_report_wf()

    """
    from pydra.tasks.nireports.interfaces import FMRISummary, PlotMosaic, PlotSpikes
    from pydra.tasks.niworkflows.interfaces.morphology import (
        BinaryDilation,
        BinarySubtraction,
    )
    from pydra.tasks.mriqc.interfaces.functional import Spikes

    # from mriqc.interfaces.reports import IndividualReport
    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(
        name=name,
        input_spec=[
            "brainmask",
            "epi_mean",
            "epi_parc",
            "fd_thres",
            "hmc_epi",
            "hmc_fd",
            "in_dvars",
            "in_fft",
            "in_ras",
            "in_spikes",
            "in_stddev",
            "meta_sidecar",
            "mni_report",
            "name_source",
            "outliers",
        ],
        meta_sidecar=meta_sidecar,
        name_source=name_source,
        in_spikes=in_spikes,
        hmc_epi=hmc_epi,
        outliers=outliers,
        epi_parc=epi_parc,
        mni_report=mni_report,
        epi_mean=epi_mean,
        in_stddev=in_stddev,
        in_fft=in_fft,
        fd_thres=fd_thres,
        in_dvars=in_dvars,
        hmc_fd=hmc_fd,
        brainmask=brainmask,
        in_ras=in_ras,
    )

    verbose = exec_verbose_reports
    mem_gb = wf_biggest_file_gb
    reportlets_dir = exec_work_dir / "reportlets"

    # Set FD threshold

    workflow.add(
        FunctionTask(
            input_spec=SpecInfo(
                name="FunctionIn",
                bases=(BaseSpec,),
                fields=[("in_file", ty.Any), ("in_mask", ty.Any)],
            ),
            output_spec=SpecInfo(
                name="FunctionOut",
                bases=(BaseSpec,),
                fields=[("out_file", ty.Any), ("out_plot", ty.Any)],
            ),
            func=spikes_mask,
            in_ras=workflow.lzin.in_ras,
            name="spmask",
        )
    )
    workflow.add(
        Spikes(
            no_zscore=True,
            detrend=False,
            in_ras=workflow.lzin.in_ras,
            in_mask=workflow.spmask.lzout.out_file,
            name="spikes_bg",
        )
    )
    # Generate crown mask
    # Create the crown mask
    workflow.add(BinaryDilation(brainmask=workflow.lzin.brainmask, name="dilated_mask"))
    workflow.add(
        BinarySubtraction(
            brainmask=workflow.lzin.brainmask,
            in_base=workflow.dilated_mask.lzout.out_mask,
            name="subtract_mask",
        )
    )
    workflow.add(
        FunctionTask(
            func=_carpet_parcellation,
            epi_parc=workflow.lzin.epi_parc,
            crown_mask=workflow.subtract_mask.lzout.out_mask,
            name="parcels",
        )
    )
    workflow.add(
        FMRISummary(
            hmc_epi=workflow.lzin.hmc_epi,
            hmc_fd=workflow.lzin.hmc_fd,
            fd_thres=workflow.lzin.fd_thres,
            in_dvars=workflow.lzin.in_dvars,
            outliers=workflow.lzin.outliers,
            in_segm=workflow.parcels.lzout.out,
            in_spikes_bg=workflow.spikes_bg.lzout.out_tsz,
            name="bigplot",
        )
    )
    # fmt: off

    @pydra.mark.task
    def inputnode_meta_sidecar_callable(in_: str):
        return _get_tr(in_)

    workflow.add(inputnode_meta_sidecar_callable(in_=workflow.lzin.meta_sidecar, name="inputnode_meta_sidecar"))

    workflow.bigplot.inputs.tr = workflow.inputnode_meta_sidecar.lzout.out
    # fmt: on
    workflow.add(
        PlotMosaic(
            out_file="plot_func_mean_mosaic1.svg",
            cmap="Greys_r",
            epi_mean=workflow.lzin.epi_mean,
            name="mosaic_mean",
        )
    )
    workflow.add(
        PlotMosaic(
            out_file="plot_func_stddev_mosaic2_stddev.svg",
            cmap="viridis",
            in_stddev=workflow.lzin.in_stddev,
            name="mosaic_stddev",
        )
    )
    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            epi_mean=workflow.lzin.epi_mean,
            brainmask=workflow.lzin.brainmask,
            name="mosaic_zoom",
        )
    )
    workflow.add(
        PlotMosaic(
            only_noise=True,
            cmap="viridis_r",
            epi_mean=workflow.lzin.epi_mean,
            name="mosaic_noise",
        )
    )
    if wf_species.lower() in ("rat", "mouse"):
        workflow.mosaic_mean.inputs.view = ["coronal", "axial"]
        workflow.mosaic_stddev.inputs.view = ["coronal", "axial"]
        workflow.mosaic_zoom.inputs.view = ["coronal", "axial"]
        workflow.mosaic_noise.inputs.view = ["coronal", "axial"]

    # fmt: off
    # fmt: on
    if wf_fft_spikes_detector:
        workflow.add(
            PlotSpikes(
                out_file="plot_spikes.svg",
                cmap="viridis",
                title="High-Frequency spikes",
                name="mosaic_spikes",
            )
        )
        workflow.add(
            DerivativesDataSink(
                base_directory=reportlets_dir,
                desc="spikes",
                datatype="figures",
                dismiss_entities=("part",),
                name="ds_report_spikes",
            )
        )
        # fmt: off
        workflow.ds_report_spikes.inputs.source_file = workflow.lzin.name_source
        workflow.mosaic_spikes.inputs.in_file = workflow.lzin.in_ras
        workflow.mosaic_spikes.inputs.in_spikes = workflow.lzin.in_spikes
        workflow.mosaic_spikes.inputs.in_fft = workflow.lzin.in_fft
        workflow.ds_report_spikes.inputs.in_file = workflow.mosaic_spikes.lzout.out_file
        # fmt: on
    if not verbose:
        return workflow
    # Verbose-reporting goes here
    from pydra.tasks.nireports.interfaces import PlotContours
    from pydra.tasks.niworkflows.utils.connections import pop_file as _pop

    workflow.add(
        PlotContours(
            display_mode="y" if wf_species.lower() in ("rat", "mouse") else "z",
            levels=[0.5],
            colors=["r"],
            cut_coords=10,
            out_file="bmask",
            brainmask=workflow.lzin.brainmask,
            name="plot_bmask",
        )
    )
    workflow.add(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc="brainmask",
            datatype="figures",
            dismiss_entities=("part", "echo"),
            name_source=workflow.lzin.name_source,
            name="ds_report_bmask",
        )
    )
    workflow.add(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc="norm",
            datatype="figures",
            dismiss_entities=("part", "echo"),
            mni_report=workflow.lzin.mni_report,
            name_source=workflow.lzin.name_source,
            name="ds_report_norm",
        )
    )
    # fmt: off

    @pydra.mark.task
    def inputnode_epi_mean_callable(in_: str):
        return _pop(in_)

    workflow.add(inputnode_epi_mean_callable(in_=workflow.lzin.epi_mean, name="inputnode_epi_mean"))

    workflow.plot_bmask.inputs.in_file = workflow.inputnode_epi_mean.lzout.out

    @pydra.mark.task
    def plot_bmask_out_file_callable(in_: str):
        return _pop(in_)

    workflow.add(plot_bmask_out_file_callable(in_=workflow.plot_bmask.lzout.out_file, name="plot_bmask_out_file"))

    workflow.ds_report_bmask.inputs.in_file = workflow.plot_bmask_out_file.lzout.out
    # fmt: on
    return workflow


def _carpet_parcellation(segmentation, crown_mask):
    """Generate the union of two masks."""
    from pathlib import Path

    import nibabel as nb
    import numpy as np

    img = nb.load(segmentation)

    lut = np.zeros((256,), dtype="uint8")
    lut[100:201] = 1  # Ctx GM
    lut[30:99] = 2  # dGM
    lut[1:11] = 3  # WM+CSF
    lut[255] = 4  # Cerebellum
    # Apply lookup table
    seg = lut[np.asanyarray(img.dataobj, dtype="uint16")]
    seg[np.asanyarray(nb.load(crown_mask).dataobj, dtype=int) > 0] = 5

    outimg = img.__class__(seg.astype("uint8"), img.affine, img.header)
    outimg.set_data_dtype("uint8")
    out_file = Path("segments.nii.gz").absolute()
    outimg.to_filename(out_file)
    return str(out_file)


def _get_tr(meta_dict):
    if isinstance(meta_dict, (list, tuple)):
        meta_dict = meta_dict[0]

    return meta_dict.get("RepetitionTime", None)


def spikes_mask(in_file, in_mask=None, out_file=None):
    """Calculate a mask in which check for :abbr:`EM (electromagnetic)` spikes."""
    import os.path as op

    import nibabel as nb
    import numpy as np
    from nilearn.image import mean_img
    from nilearn.plotting import plot_roi
    from scipy import ndimage as nd

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_spmask{ext}")
        out_plot = op.abspath(f"{fname}_spmask.pdf")

    in_4d_nii = nb.load(in_file)
    orientation = nb.aff2axcodes(in_4d_nii.affine)

    if in_mask:
        mask_data = np.asanyarray(nb.load(in_mask).dataobj)
        a = np.where(mask_data != 0)
        bbox = (
            np.max(a[0]) - np.min(a[0]),
            np.max(a[1]) - np.min(a[1]),
            np.max(a[2]) - np.min(a[2]),
        )
        longest_axis = np.argmax(bbox)

        # Input here is a binarized and intersected mask data from previous section
        dil_mask = nd.binary_dilation(
            mask_data, iterations=int(mask_data.shape[longest_axis] / 9)
        )

        rep = list(mask_data.shape)
        rep[longest_axis] = -1
        new_mask_2d = dil_mask.max(axis=longest_axis).reshape(rep)

        rep = [1, 1, 1]
        rep[longest_axis] = mask_data.shape[longest_axis]
        new_mask_3d = np.logical_not(np.tile(new_mask_2d, rep))
    else:
        new_mask_3d = np.zeros(in_4d_nii.shape[:3]) == 1

    if orientation[0] in ("L", "R"):
        new_mask_3d[0:2, :, :] = True
        new_mask_3d[-3:-1, :, :] = True
    else:
        new_mask_3d[:, 0:2, :] = True
        new_mask_3d[:, -3:-1, :] = True

    mask_nii = nb.Nifti1Image(
        new_mask_3d.astype(np.uint8), in_4d_nii.affine, in_4d_nii.header
    )
    mask_nii.to_filename(out_file)

    plot_roi(mask_nii, mean_img(in_4d_nii), output_file=out_plot)
    return out_file, out_plot
