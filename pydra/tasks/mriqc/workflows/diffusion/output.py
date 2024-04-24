import attrs
import logging
from pathlib import Path
from pydra.engine import Workflow
from pydra.engine.task import FunctionTask
import pydra.mark
from pydra.tasks.mriqc.interfaces import DerivativesDataSink
from pydra.tasks.nireports.interfaces.dmri import DWIHeatmap
from pydra.tasks.nireports.interfaces.reporting.base import (
    SimpleBeforeAfterRPT as SimpleBeforeAfter,
)


logger = logging.getLogger(__name__)


def init_dwi_report_wf(
    brain_mask=attrs.NOTHING,
    epi_mean=attrs.NOTHING,
    epi_parc=attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    fd_thres=attrs.NOTHING,
    hmc_epi=attrs.NOTHING,
    hmc_fd=attrs.NOTHING,
    in_avgmap=attrs.NOTHING,
    in_bdict=attrs.NOTHING,
    in_dvars=attrs.NOTHING,
    in_epi=attrs.NOTHING,
    in_fa=attrs.NOTHING,
    in_fft=attrs.NOTHING,
    in_md=attrs.NOTHING,
    in_parcellation=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    in_shells=attrs.NOTHING,
    in_spikes=attrs.NOTHING,
    in_stdmap=attrs.NOTHING,
    meta_sidecar=attrs.NOTHING,
    mni_report=attrs.NOTHING,
    name="dwi_report_wf",
    name_source=attrs.NOTHING,
    noise_floor=attrs.NOTHING,
    outliers=attrs.NOTHING,
    wf_biggest_file_gb=1,
    wf_fd_thres=0.2,
    wf_fft_spikes_detector=False,
    wf_species="human",
):
    """
    Write out individual reportlets.

    .. workflow::

        from mriqc.workflows.diffusion.output import init_dwi_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_dwi_report_wf()

    """
    from pydra.tasks.nireports.interfaces import FMRISummary, PlotMosaic, PlotSpikes
    from pydra.tasks.niworkflows.interfaces.morphology import (
        BinaryDilation,
        BinarySubtraction,
    )

    # from mriqc.interfaces.reports import IndividualReport
    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(
        name=name,
        input_spec=[
            "brain_mask",
            "epi_mean",
            "epi_parc",
            "fd_thres",
            "hmc_epi",
            "hmc_fd",
            "in_avgmap",
            "in_bdict",
            "in_dvars",
            "in_epi",
            "in_fa",
            "in_fft",
            "in_md",
            "in_parcellation",
            "in_ras",
            "in_shells",
            "in_spikes",
            "in_stdmap",
            "meta_sidecar",
            "mni_report",
            "name_source",
            "noise_floor",
            "outliers",
        ],
        name_source=name_source,
        mni_report=mni_report,
        in_bdict=in_bdict,
        epi_mean=epi_mean,
        in_md=in_md,
        in_parcellation=in_parcellation,
        in_epi=in_epi,
        in_fft=in_fft,
        outliers=outliers,
        in_shells=in_shells,
        meta_sidecar=meta_sidecar,
        fd_thres=fd_thres,
        hmc_fd=hmc_fd,
        in_dvars=in_dvars,
        brain_mask=brain_mask,
        in_stdmap=in_stdmap,
        noise_floor=noise_floor,
        epi_parc=epi_parc,
        in_fa=in_fa,
        in_avgmap=in_avgmap,
        in_spikes=in_spikes,
        hmc_epi=hmc_epi,
        in_ras=in_ras,
    )

    verbose = exec_verbose_reports
    mem_gb = wf_biggest_file_gb
    reportlets_dir = exec_work_dir / "reportlets"

    # Set FD threshold
    # inputnode.inputs.fd_thres = wf_fd_thres
    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            in_fa=workflow.lzin.in_fa,
            brain_mask=workflow.lzin.brain_mask,
            name="mosaic_fa",
        )
    )
    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            in_md=workflow.lzin.in_md,
            brain_mask=workflow.lzin.brain_mask,
            name="mosaic_md",
        )
    )
    workflow.add(
        SimpleBeforeAfter(
            fixed_params={"cmap": "viridis"},
            moving_params={"cmap": "Greys_r"},
            before_label="Average",
            after_label="Standard Deviation",
            dismiss_affine=True,
            in_avgmap=workflow.lzin.in_avgmap,
            in_stdmap=workflow.lzin.in_stdmap,
            brain_mask=workflow.lzin.brain_mask,
            name="mosaic_snr",
        )
    )
    workflow.add(
        PlotMosaic(
            only_noise=True,
            cmap="viridis_r",
            in_avgmap=workflow.lzin.in_avgmap,
            epi_mean=workflow.lzin.epi_mean,
            name="mosaic_noise",
        )
    )
    if wf_species.lower() in ("rat", "mouse"):
        workflow.mosaic_noise.inputs.view = ["coronal", "axial"]
        workflow.mosaic_fa.inputs.view = ["coronal", "axial"]
        workflow.mosaic_md.inputs.view = ["coronal", "axial"]

    def _gen_entity(inlist):
        return ["00000"] + [f"{int(round(bval, 0)):05d}" for bval in inlist]

    # fmt: off

    @pydra.mark.task
    def inputnode_in_shells_callable(in_: str):
        return _gen_entity(in_)

    workflow.add(inputnode_in_shells_callable(in_=workflow.inputnode.lzout.in_shells, name="inputnode_in_shells"))

    workflow.set_output([('bval', workflow.inputnode_in_shells.lzout.out)])

    @pydra.mark.task
    def inputnode_in_shells_callable(in_: str):
        return _gen_entity(in_)

    workflow.add(inputnode_in_shells_callable(in_=workflow.inputnode.lzout.in_shells, name="inputnode_in_shells"))

    workflow.set_output([('bval', workflow.inputnode_in_shells.lzout.out)])
    # fmt: on
    workflow.add(
        FunctionTask(
            func=_get_wm, in_parcellation=workflow.lzin.in_parcellation, name="get_wm"
        )
    )
    workflow.add(
        DWIHeatmap(
            scalarmap_label="Shell-wise Fractional Anisotropy (FA)",
            in_epi=workflow.lzin.in_epi,
            in_fa=workflow.lzin.in_fa,
            in_bdict=workflow.lzin.in_bdict,
            noise_floor=workflow.lzin.noise_floor,
            mask_file=workflow.get_wm.lzout.out,
            name="plot_heatmap",
        )
    )

    # fmt: off
    # fmt: on
    # if True:
    #     return workflow
    # Generate crown mask
    # Create the crown mask
    workflow.add(
        BinaryDilation(brain_mask=workflow.lzin.brain_mask, name="dilated_mask")
    )
    workflow.add(
        BinarySubtraction(
            brain_mask=workflow.lzin.brain_mask,
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
    if wf_fft_spikes_detector:
        workflow.add(
            PlotSpikes(
                out_file="plot_spikes.svg",
                cmap="viridis",
                title="High-Frequency spikes",
                name="mosaic_spikes",
            )
        )

        # fmt: off
        workflow.set_output([('source_file', workflow.inputnode.lzout.name_source)])
        workflow.mosaic_spikes.inputs.in_file = workflow.lzin.in_ras
        workflow.mosaic_spikes.inputs.in_spikes = workflow.lzin.in_spikes
        workflow.mosaic_spikes.inputs.in_fft = workflow.lzin.in_fft
        workflow.set_output([('in_file', workflow.mosaic_spikes.lzout.out_file)])
        # fmt: on
    if not verbose:
        return workflow
    # Verbose-reporting goes here
    from pydra.tasks.nireports.interfaces import PlotContours

    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            epi_mean=workflow.lzin.epi_mean,
            brain_mask=workflow.lzin.brain_mask,
            name="mosaic_zoom",
        )
    )
    workflow.add(
        PlotContours(
            display_mode="y" if wf_species.lower() in ("rat", "mouse") else "z",
            levels=[0.5],
            colors=["r"],
            cut_coords=10,
            out_file="bmask",
            epi_mean=workflow.lzin.epi_mean,
            brain_mask=workflow.lzin.brain_mask,
            name="plot_bmask",
        )
    )
    workflow.add(
        DerivativesDataSink(
            base_directory=reportlets_dir,
            desc="zoomed",
            datatype="figures",
            name_source=workflow.lzin.name_source,
            in_file=workflow.mosaic_zoom.lzout.out_file,
            name="ds_report_zoomed",
        )
    )

    # fmt: off
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
    return meta_dict.get("RepetitionTime", None)


def _get_wm(in_file, radius=2):
    from pathlib import Path

    import nibabel as nb
    import numpy as np
    from pydra.tasks.mriqc.nipype_ports.utils.filemanip import fname_presuffix
    from scipy import ndimage as ndi
    from skimage.morphology import ball

    parc = nb.load(in_file)
    hdr = parc.header.copy()
    data = np.array(parc.dataobj, dtype=hdr.get_data_dtype())
    wm_mask = ndi.binary_erosion((data == 1) | (data == 2), ball(radius))

    hdr.set_data_dtype(np.uint8)
    out_wm = fname_presuffix(in_file, suffix="wm", newpath=str(Path.cwd()))
    parc.__class__(
        wm_mask.astype(np.uint8),
        parc.affine,
        hdr,
    ).to_filename(out_wm)
    return out_wm
