import attrs
import logging
from pathlib import Path
from pydra.engine import Workflow


logger = logging.getLogger(__name__)


def init_anat_report_wf(
    airmask=attrs.NOTHING,
    artmask=attrs.NOTHING,
    brainmask=attrs.NOTHING,
    exec_verbose_reports=False,
    exec_work_dir=None,
    headmask=attrs.NOTHING,
    in_ras=attrs.NOTHING,
    mni_report=attrs.NOTHING,
    name: str = "anat_report_wf",
    name_source=attrs.NOTHING,
    noisefit=attrs.NOTHING,
    segmentation=attrs.NOTHING,
    wf_species="human",
):
    """
    Generate the components of the individual report.

    .. workflow::

        from mriqc.workflows.anatomical.output import init_anat_report_wf
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_anat_report_wf()

    """
    from pydra.tasks.nireports.interfaces import PlotMosaic

    # from mriqc.interfaces.reports import IndividualReport
    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(
        name=name,
        input_spec=[
            "airmask",
            "artmask",
            "brainmask",
            "headmask",
            "in_ras",
            "mni_report",
            "name_source",
            "noisefit",
            "segmentation",
        ],
        name_source=name_source,
        headmask=headmask,
        artmask=artmask,
        mni_report=mni_report,
        airmask=airmask,
        brainmask=brainmask,
        segmentation=segmentation,
        noisefit=noisefit,
        in_ras=in_ras,
    )

    verbose = exec_verbose_reports
    reportlets_dir = exec_work_dir / "reportlets"

    workflow.add(
        PlotMosaic(
            cmap="Greys_r",
            in_ras=workflow.lzin.in_ras,
            brainmask=workflow.lzin.brainmask,
            name="mosaic_zoom",
        )
    )
    workflow.add(
        PlotMosaic(
            only_noise=True,
            cmap="viridis_r",
            in_ras=workflow.lzin.in_ras,
            name="mosaic_noise",
        )
    )
    if wf_species.lower() in ("rat", "mouse"):
        workflow.mosaic_zoom.inputs.view = ["coronal", "axial"]
        workflow.mosaic_noise.inputs.view = ["coronal", "axial"]

    # fmt: off
    # fmt: on
    if not verbose:
        return workflow
    from pydra.tasks.nireports.interfaces import PlotContours

    display_mode = "y" if wf_species.lower() in ("rat", "mouse") else "z"
    workflow.add(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5, 1.5, 2.5],
            cut_coords=10,
            colors=["r", "g", "b"],
            in_ras=workflow.lzin.in_ras,
            segmentation=workflow.lzin.segmentation,
            name="plot_segm",
        )
    )

    workflow.add(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=["r"],
            cut_coords=10,
            out_file="bmask",
            in_ras=workflow.lzin.in_ras,
            brainmask=workflow.lzin.brainmask,
            name="plot_bmask",
        )
    )

    workflow.add(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=["r"],
            cut_coords=10,
            out_file="artmask",
            saturate=True,
            in_ras=workflow.lzin.in_ras,
            artmask=workflow.lzin.artmask,
            name="plot_artmask",
        )
    )

    # NOTE: humans switch on these two to coronal view.
    display_mode = "y" if wf_species.lower() in ("rat", "mouse") else "x"
    workflow.add(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=["r"],
            cut_coords=6,
            out_file="airmask",
            in_ras=workflow.lzin.in_ras,
            airmask=workflow.lzin.airmask,
            name="plot_airmask",
        )
    )

    workflow.add(
        PlotContours(
            display_mode=display_mode,
            levels=[0.5],
            colors=["r"],
            cut_coords=6,
            out_file="headmask",
            in_ras=workflow.lzin.in_ras,
            headmask=workflow.lzin.headmask,
            name="plot_headmask",
        )
    )

    # fmt: off
    # fmt: on
    return workflow
