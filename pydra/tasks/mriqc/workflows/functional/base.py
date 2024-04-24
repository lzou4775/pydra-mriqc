import logging
from pydra.tasks.mriqc.workflows.functional.output import init_func_report_wf
import nibabel as nb

from pydra.tasks.niworkflows.utils.connections import pop_file as _pop
from pathlib import Path
from pydra.engine import Workflow
import pydra.mark
from pydra.tasks.niworkflows.utils.connections import pop_file as _pop


logger = logging.getLogger(__name__)


def fmri_bmsk_workflow(name="fMRIBrainMask"):
    """
    Compute a brain mask for the input :abbr:`fMRI (functional MRI)` dataset.

    .. workflow::

        from mriqc.workflows.functional.base import fmri_bmsk_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_bmsk_workflow()


    """
    from pydra.tasks.afni.auto import Automask

    workflow = Workflow(name=name, input_spec=["in_file"])

    workflow.add(
        Automask(outputtype="NIFTI_GZ", in_file=workflow.lzin.in_file, name="afni_msk")
    )
    # Connect brain mask extraction
    # fmt: off
    workflow.set_output([('out_file', workflow.afni_msk.lzout.out_file)])
    # fmt: on
    return workflow


def epi_mni_align(
    exec_ants_float=False,
    exec_debug=False,
    name="SpatialNormalization",
    nipype_nprocs=10,
    nipype_omp_nthreads=10,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    Estimate the transform that maps the EPI space into MNI152NLin2009cAsym.

    The input epi_mean is the averaged and brain-masked EPI timeseries

    Returns the EPI mean resampled in MNI space (for checking out registration) and
    the associated "lobe" parcellation in EPI space.

    .. workflow::

        from mriqc.workflows.functional.base import epi_mni_align
        from mriqc.testing import mock_config
        with mock_config():
            wf = epi_mni_align()

    """
    from pydra.tasks.ants.auto import ApplyTransforms, N4BiasFieldCorrection
    from pydra.tasks.niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )
    from templateflow.api import get as get_template

    # Get settings
    workflow = Workflow(name=name, input_spec=["epi_mask", "epi_mean"])

    testing = exec_debug
    n_procs = nipype_nprocs
    ants_nthreads = nipype_omp_nthreads

    workflow.add(
        N4BiasFieldCorrection(
            dimension=3, copy_header=True, epi_mean=workflow.lzin.epi_mean, name="n4itk"
        )
    )
    workflow.add(
        RobustMNINormalization(
            explicit_masking=False,
            flavor="testing" if testing else "precise",
            float=exec_ants_float,
            generate_report=True,
            moving="boldref",
            num_threads=ants_nthreads,
            reference="boldref",
            template=wf_template_id,
            moving_image=workflow.n4itk.lzout.output_image,
            name="norm",
        )
    )
    if wf_species.lower() == "human":
        workflow.norm.inputs.reference_image = str(
            get_template(wf_template_id, resolution=2, suffix="boldref")
        )
        workflow.norm.inputs.reference_mask = str(
            get_template(
                wf_template_id,
                resolution=2,
                desc="brain",
                suffix="mask",
            )
        )
    # adapt some population-specific settings
    else:
        from nirodents.workflows.brainextraction import _bspline_grid

        workflow.n4itk.inputs.shrink_factor = 1
        workflow.n4itk.inputs.n_iterations = [50] * 4
        workflow.norm.inputs.reference_image = str(
            get_template(wf_template_id, suffix="T2w")
        )
        workflow.norm.inputs.reference_mask = str(
            get_template(
                wf_template_id,
                desc="brain",
                suffix="mask",
            )[0]
        )
        workflow.add(niu.Function(function=_bspline_grid, name="bspline_grid"))
        # fmt: off
        workflow.bspline_grid.inputs.in_file = workflow.lzin.epi_mean
        workflow.n4itk.inputs.args = workflow.bspline_grid.lzout.out
        # fmt: on
    # Warp segmentation into EPI space
    workflow.add(
        ApplyTransforms(
            float=True,
            dimension=3,
            default_value=0,
            interpolation="MultiLabel",
            epi_mean=workflow.lzin.epi_mean,
            transforms=workflow.norm.lzout.inverse_composite_transform,
            name="invt",
        )
    )
    if wf_species.lower() == "human":
        workflow.invt.inputs.input_image = str(
            get_template(
                wf_template_id,
                resolution=1,
                desc="carpet",
                suffix="dseg",
            )
        )
    else:
        workflow.invt.inputs.input_image = str(
            get_template(
                wf_template_id,
                suffix="dseg",
            )[-1]
        )
    # fmt: off
    workflow.set_output([('epi_parc', workflow.invt.lzout.output_image)])
    workflow.set_output([('epi_mni', workflow.norm.lzout.warped_image)])
    workflow.set_output([('report', workflow.norm.lzout.out_report)])
    # fmt: on
    if wf_species.lower() == "human":
        workflow.norm.inputs.moving_mask = workflow.lzin.epi_mask
    return workflow


def hmc(
    name="fMRI_HMC",
    omp_nthreads=None,
    wf_biggest_file_gb=1,
    wf_deoblique=False,
    wf_despike=False,
):
    """
    Create a :abbr:`HMC (head motion correction)` workflow for fMRI.

    .. workflow::

        from mriqc.workflows.functional.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        FramewiseDisplacement,
    )
    from pydra.tasks.afni.auto import Despike, Refit, Volreg

    workflow = Workflow(name=name, input_spec=["fd_radius", "in_file"])

    mem_gb = wf_biggest_file_gb

    # calculate hmc parameters
    workflow.add(
        Volreg(
            args="-Fourier -twopass", zpad=4, outputtype="NIFTI_GZ", name="estimate_hm"
        )
    )
    # Compute the frame-wise displacement
    workflow.add(
        FramewiseDisplacement(
            normalize=False,
            parameter_source="AFNI",
            fd_radius=workflow.lzin.fd_radius,
            in_file=workflow.estimate_hm.lzout.oned_file,
            name="fdnode",
        )
    )
    # Apply transforms to other echos
    workflow.add(
        niu.Function(
            function=_apply_transforms,
            input_names=["in_file", "in_xfm"],
            in_xfm=workflow.estimate_hm.lzout.oned_matrix_save,
            name="apply_hmc",
        )
    )
    # fmt: off
    workflow.set_output([('out_file', workflow.apply_hmc.lzout.out)])
    workflow.set_output([('mpars', workflow.estimate_hm.lzout.oned_file)])
    workflow.set_output([('out_fd', workflow.fdnode.lzout.out_file)])
    # fmt: on
    if not (wf_despike or wf_deoblique):
        # fmt: off

        @pydra.mark.task
        def inputnode_in_file_callable(in_: str):
            return _pop(in_)

        workflow.add(inputnode_in_file_callable(in_=workflow.lzin.in_file, name="inputnode_in_file"))

        workflow.estimate_hm.inputs.in_file = workflow.inputnode_in_file.lzout.out
        workflow.apply_hmc.inputs.in_file = workflow.lzin.in_file
        # fmt: on
        return workflow
    # despiking, and deoblique
    workflow.add(Refit(deoblique=True, name="deoblique_node"))
    workflow.add(Despike(outputtype="NIFTI_GZ", name="despike_node"))
    if wf_despike and wf_deoblique:
        # fmt: off
        workflow.despike_node.inputs.in_file = workflow.lzin.in_file
        workflow.deoblique_node.inputs.in_file = workflow.despike_node.lzout.out_file

        @pydra.mark.task
        def deoblique_node_out_file_callable(in_: str):
            return _pop(in_)

        workflow.add(deoblique_node_out_file_callable(in_=workflow.deoblique_node.lzout.out_file, name="deoblique_node_out_file"))

        workflow.estimate_hm.inputs.in_file = workflow.deoblique_node_out_file.lzout.out
        workflow.apply_hmc.inputs.in_file = workflow.deoblique_node.lzout.out_file
        # fmt: on
    elif wf_despike:
        # fmt: off
        workflow.despike_node.inputs.in_file = workflow.lzin.in_file

        @pydra.mark.task
        def despike_node_out_file_callable(in_: str):
            return _pop(in_)

        workflow.add(despike_node_out_file_callable(in_=workflow.despike_node.lzout.out_file, name="despike_node_out_file"))

        workflow.estimate_hm.inputs.in_file = workflow.despike_node_out_file.lzout.out
        workflow.apply_hmc.inputs.in_file = workflow.despike_node.lzout.out_file
        # fmt: on
    elif wf_deoblique:
        # fmt: off
        workflow.deoblique_node.inputs.in_file = workflow.lzin.in_file

        @pydra.mark.task
        def deoblique_node_out_file_callable(in_: str):
            return _pop(in_)

        workflow.add(deoblique_node_out_file_callable(in_=workflow.deoblique_node.lzout.out_file, name="deoblique_node_out_file"))

        workflow.estimate_hm.inputs.in_file = workflow.deoblique_node_out_file.lzout.out
        workflow.apply_hmc.inputs.in_file = workflow.deoblique_node.lzout.out_file
        # fmt: on
    else:
        raise NotImplementedError
    return workflow


def _apply_transforms(in_file, in_xfm):
    from pathlib import Path

    from nitransforms.linear import load

    from mriqc.utils.bids import derive_bids_fname

    realigned = load(in_xfm, fmt="afni", reference=in_file, moving=in_file).apply(
        in_file
    )
    out_file = derive_bids_fname(
        in_file,
        entity="desc-realigned",
        newpath=Path.cwd(),
        absolute=True,
    )

    realigned.to_filename(out_file)
    return str(out_file)


def compute_iqms(
    exec_dsname="<unset>",
    exec_output_dir=None,
    name="ComputeIQMs",
    wf_biggest_file_gb=1,
    wf_fft_spikes_detector=False,
):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.functional.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import ComputeDVARS
    from pydra.tasks.afni.auto import OutlierCount, QualityIndex
    from pydra.tasks.mriqc.interfaces import (
        DerivativesDataSink,
        FunctionalQC,
        GatherTimeseries,
        IQMFileSink,
    )
    from pydra.tasks.mriqc.interfaces.reports import AddProvenance
    from pydra.tasks.mriqc.interfaces.transitional import GCOR
    from pydra.tasks.mriqc.workflows.utils import _tofloat, get_fwhmx

    workflow = Workflow(
        name=name,
        input_spec=[
            "acquisition",
            "brainmask",
            "epi_mean",
            "exclude_index",
            "fd_thres",
            "hmc_epi",
            "hmc_fd",
            "in_file",
            "in_ras",
            "in_tsnr",
            "metadata",
            "mpars",
            "reconstruction",
            "run",
            "session",
            "subject",
            "task",
        ],
    )

    mem_gb = wf_biggest_file_gb

    # Set FD threshold

    # Compute DVARS
    workflow.add(
        ComputeDVARS(
            save_plot=False,
            save_all=True,
            hmc_epi=workflow.lzin.hmc_epi,
            brainmask=workflow.lzin.brainmask,
            name="dvnode",
        )
    )
    # AFNI quality measures
    workflow.add(
        get_fwhmx(
            epi_mean=workflow.lzin.epi_mean,
            brainmask=workflow.lzin.brainmask,
            name="fwhm",
        )
    )
    workflow.fwhm.inputs.acf = True  # Only AFNI >= 16
    workflow.add(
        OutlierCount(
            fraction=True,
            out_file="outliers.out",
            hmc_epi=workflow.lzin.hmc_epi,
            brainmask=workflow.lzin.brainmask,
            name="outliers",
        )
    )
    workflow.add(
        QualityIndex(automask=True, hmc_epi=workflow.lzin.hmc_epi, name="quality")
    )
    workflow.add(
        GCOR(
            hmc_epi=workflow.lzin.hmc_epi,
            brainmask=workflow.lzin.brainmask,
            name="gcor",
        )
    )
    workflow.add(
        FunctionalQC(
            epi_mean=workflow.lzin.epi_mean,
            brainmask=workflow.lzin.brainmask,
            hmc_epi=workflow.lzin.hmc_epi,
            hmc_fd=workflow.lzin.hmc_fd,
            fd_thres=workflow.lzin.fd_thres,
            in_tsnr=workflow.lzin.in_tsnr,
            in_dvars=workflow.dvnode.lzout.out_all,
            name="measures",
        )
    )
    workflow.add(
        GatherTimeseries(
            mpars_source="AFNI",
            outliers=workflow.outliers.lzout.out_file,
            quality=workflow.quality.lzout.out_file,
            dvars=workflow.dvnode.lzout.out_all,
            hmc_fd=workflow.lzin.hmc_fd,
            mpars=workflow.lzin.mpars,
            name="timeseries",
        )
    )
    # fmt: off

    @pydra.mark.task
    def fwhm_fwhm_callable(in_: str):
        return _tofloat(in_)

    workflow.add(fwhm_fwhm_callable(in_=workflow.fwhm.lzout.fwhm, name="fwhm_fwhm"))

    workflow.measures.inputs.in_fwhm = workflow.fwhm_fwhm.lzout.out
    workflow.set_output([('out_dvars', workflow.dvnode.lzout.out_all)])
    workflow.set_output([('outliers', workflow.outliers.lzout.out_file)])
    # fmt: on
    workflow.add(
        AddProvenance(modality="bold", in_file=workflow.lzin.in_file, name="addprov")
    )
    # Save to JSON file
    workflow.add(
        IQMFileSink(
            modality="bold",
            out_dir=str(exec_output_dir),
            dataset=exec_dsname,
            in_file=workflow.lzin.in_file,
            exclude_index=workflow.lzin.exclude_index,
            metadata=workflow.lzin.metadata,
            provenance=workflow.addprov.lzout.out_prov,
            root=workflow.measures.lzout.out_qc,
            name="datasink",
        )
    )
    # Save timeseries TSV file

    # fmt: off

    @pydra.mark.task
    def inputnode_subject_callable(in_: str):
        return _pop(in_)

    workflow.add(inputnode_subject_callable(in_=workflow.lzin.subject, name="inputnode_subject"))

    workflow.datasink.inputs.subject_id = workflow.inputnode_subject.lzout.out

    @pydra.mark.task
    def inputnode_session_callable(in_: str):
        return _pop(in_)

    workflow.add(inputnode_session_callable(in_=workflow.lzin.session, name="inputnode_session"))

    workflow.datasink.inputs.session_id = workflow.inputnode_session.lzout.out

    @pydra.mark.task
    def inputnode_task_callable(in_: str):
        return _pop(in_)

    workflow.add(inputnode_task_callable(in_=workflow.lzin.task, name="inputnode_task"))

    workflow.datasink.inputs.task_id = workflow.inputnode_task.lzout.out

    @pydra.mark.task
    def inputnode_acquisition_callable(in_: str):
        return _pop(in_)

    workflow.add(inputnode_acquisition_callable(in_=workflow.lzin.acquisition, name="inputnode_acquisition"))

    workflow.datasink.inputs.acq_id = workflow.inputnode_acquisition.lzout.out

    @pydra.mark.task
    def inputnode_reconstruction_callable(in_: str):
        return _pop(in_)

    workflow.add(inputnode_reconstruction_callable(in_=workflow.lzin.reconstruction, name="inputnode_reconstruction"))

    workflow.datasink.inputs.rec_id = workflow.inputnode_reconstruction.lzout.out

    @pydra.mark.task
    def inputnode_run_callable(in_: str):
        return _pop(in_)

    workflow.add(inputnode_run_callable(in_=workflow.lzin.run, name="inputnode_run"))

    workflow.datasink.inputs.run_id = workflow.inputnode_run.lzout.out

    @pydra.mark.task
    def outliers_out_file_callable(in_: str):
        return _parse_tout(in_)

    workflow.add(outliers_out_file_callable(in_=workflow.outliers.lzout.out_file, name="outliers_out_file"))

    workflow.datasink.inputs.aor = workflow.outliers_out_file.lzout.out

    @pydra.mark.task
    def gcor_out_callable(in_: str):
        return _tofloat(in_)

    workflow.add(gcor_out_callable(in_=workflow.gcor.lzout.out, name="gcor_out"))

    workflow.datasink.inputs.gcor = workflow.gcor_out.lzout.out

    @pydra.mark.task
    def quality_out_file_callable(in_: str):
        return _parse_tqual(in_)

    workflow.add(quality_out_file_callable(in_=workflow.quality.lzout.out_file, name="quality_out_file"))

    workflow.datasink.inputs.aqi = workflow.quality_out_file.lzout.out
    workflow.set_output([('out_file', workflow.datasink.lzout.out_file)])
    # fmt: on
    # FFT spikes finder
    if wf_fft_spikes_detector:
        from pydra.tasks.mriqc.workflows.utils import slice_wise_fft

        workflow.add(
            niu.Function(
                input_names=["in_file"],
                output_names=["n_spikes", "out_spikes", "out_fft"],
                function=slice_wise_fft,
                name="spikes_fft",
            )
        )
        # fmt: off
        workflow.spikes_fft.inputs.in_file = workflow.lzin.in_ras
        workflow.set_output([('out_spikes', workflow.spikes_fft.lzout.out_spikes)])
        workflow.set_output([('out_fft', workflow.spikes_fft.lzout.out_fft)])
        workflow.datasink.inputs.spikes_num = workflow.spikes_fft.lzout.n_spikes
        # fmt: on
    return workflow


def _parse_tout(in_file):
    if isinstance(in_file, (list, tuple)):
        return (
            [_parse_tout(f) for f in in_file]
            if len(in_file) > 1
            else _parse_tout(in_file[0])
        )

    import numpy as np

    data = np.loadtxt(in_file)  # pylint: disable=no-member
    return data.mean()


def _parse_tqual(in_file):
    if isinstance(in_file, (list, tuple)):
        return (
            [_parse_tqual(f) for f in in_file]
            if len(in_file) > 1
            else _parse_tqual(in_file[0])
        )

    import numpy as np

    with open(in_file) as fin:
        lines = fin.readlines()
    return np.mean([float(line.strip()) for line in lines if not line.startswith("++")])


def fmri_qc_workflow(
    exec_ants_float=False,
    exec_bids_database_dir=None,
    exec_datalad_get=True,
    exec_debug=False,
    exec_dsname="<unset>",
    exec_float32=True,
    exec_no_sub=False,
    exec_output_dir=None,
    exec_upload_strict=False,
    exec_verbose_reports=False,
    exec_webapi_token="<secret_token>",
    exec_webapi_url="https://mriqc.nimh.nih.gov:443/api/v1",
    exec_work_dir=None,
    name="funcMRIQC",
    nipype_nprocs=10,
    nipype_omp_nthreads=10,
    wf_biggest_file_gb=1,
    wf_deoblique=False,
    wf_despike=False,
    wf_fd_radius=50,
    wf_fft_spikes_detector=False,
    wf_inputs=None,
    wf_min_len_bold=5,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    Initialize the (f)MRIQC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.functional.base import fmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = fmri_qc_workflow()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        NonSteadyStateDetector,
        TSNR,
    )
    from pydra.tasks.afni.auto import TStat
    from pydra.tasks.niworkflows.interfaces.bids import ReadSidecarJSON
    from pydra.tasks.niworkflows.interfaces.header import SanitizeImage
    from pydra.tasks.mriqc.interfaces.functional import SelectEcho

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(name=name, input_spec=["in_file"])

    mem_gb = wf_biggest_file_gb
    dataset = wf_inputs.get("bold", [])
    if exec_datalad_get:
        from pydra.tasks.mriqc.utils.misc import _datalad_get

        _datalad_get(dataset)
    full_files = []
    for bold_path in dataset:
        try:
            bold_len = nb.load(bold_path).shape[3]
        except nb.filebasedimages.ImageFileError:
            bold_len = wf_min_len_bold
        except IndexError:  # shape has only 3 elements
            bold_len = 0
        if bold_len >= wf_min_len_bold:
            full_files.append(bold_path)
        else:
            logger.warn(
                f"Dismissing {bold_path} for processing: insufficient number of "
                f"timepoints ({bold_len}) to execute the workflow."
            )
    message = "Building {modality} MRIQC workflow {detail}.".format(
        modality="functional",
        detail=(
            f"for {len(full_files)} BOLD runs."
            if len(full_files) > 2
            else f"({' and '.join('<%s>' % v for v in dataset)})."
        ),
    )
    logger.info(message)
    if set(dataset) - set(full_files):
        wf_inputs["bold"] = full_files
        
    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation

    # Get metadata
    workflow.add(
        ReadSidecarJSON(
            index_db=exec_bids_database_dir, in_file=workflow.lzin.in_file, name="meta"
        )
    )
    workflow.add(
        SelectEcho(
            in_file=workflow.lzin.in_file,
            metadata=workflow.meta.lzout.out_dict,
            name="pick_echo",
        )
    )
    workflow.add(
        NonSteadyStateDetector(
            in_file=workflow.pick_echo.lzout.out_file, name="non_steady_state_detector"
        )
    )
    workflow.add(
        SanitizeImage(
            max_32bit=exec_float32,
            in_file=workflow.lzin.in_file,
            n_volumes_to_discard=workflow.non_steady_state_detector.lzout.n_volumes_to_discard,
            name="sanitize",
        )
    )
    # Workflow --------------------------------------------------------
    # 1. HMC: head motion correct
    workflow.add(
        hmc(
            omp_nthreads=nipype_omp_nthreads,
            wf_biggest_file_gb=wf_biggest_file_gb,
            wf_deoblique=wf_deoblique,
            wf_despike=wf_despike,
        )(in_file=workflow.sanitize.lzout.out_file, name="hmcwf")
    )
    # Set HMC settings
    workflow.hmcwf.inputs.inputnode.fd_radius = wf_fd_radius
    # 2. Compute mean fmri
    workflow.add(
        TStat(
            options="-mean",
            outputtype="NIFTI_GZ",
            in_file=workflow.hmcwf.lzout.out_file,
            name="mean",
        )
    )
    # Compute TSNR using nipype implementation
    workflow.add(TSNR(in_file=workflow.hmcwf.lzout.out_file, name="tsnr"))
    # EPI to MNI registration
    workflow.add(
        epi_mni_align(
            exec_ants_float=exec_ants_float,
            exec_debug=exec_debug,
            nipype_nprocs=nipype_nprocs,
            nipype_omp_nthreads=nipype_omp_nthreads,
            wf_species=wf_species,
            wf_template_id=wf_template_id,
        )(name="ema")
    )
    # 7. Compute IQMs
    workflow.add(
        compute_iqms(
            exec_dsname=exec_dsname,
            exec_output_dir=exec_output_dir,
            wf_biggest_file_gb=wf_biggest_file_gb,
            wf_fft_spikes_detector=wf_fft_spikes_detector,
        )(
            metadata=workflow.meta.lzout.out_dict,
            subject=workflow.meta.lzout.subject,
            session=workflow.meta.lzout.session,
            task=workflow.meta.lzout.task,
            acquisition=workflow.meta.lzout.acquisition,
            reconstruction=workflow.meta.lzout.reconstruction,
            run=workflow.meta.lzout.run,
            in_file=workflow.lzin.in_file,
            in_ras=workflow.sanitize.lzout.out_file,
            epi_mean=workflow.mean.lzout.out_file,
            hmc_epi=workflow.hmcwf.lzout.out_file,
            hmc_fd=workflow.hmcwf.lzout.out_fd,
            mpars=workflow.hmcwf.lzout.mpars,
            in_tsnr=workflow.tsnr.lzout.tsnr_file,
            exclude_index=workflow.non_steady_state_detector.lzout.n_volumes_to_discard,
            name="iqmswf",
        )
    )
    # Reports
    workflow.add(
        init_func_report_wf(
            exec_verbose_reports=exec_verbose_reports,
            exec_work_dir=exec_work_dir,
            wf_biggest_file_gb=wf_biggest_file_gb,
            wf_fft_spikes_detector=wf_fft_spikes_detector,
            wf_species=wf_species,
        )(
            in_file=workflow.lzin.in_file,
            in_ras=workflow.sanitize.lzout.out_file,
            epi_mean=workflow.mean.lzout.out_file,
            in_stddev=workflow.tsnr.lzout.stddev_file,
            hmc_fd=workflow.hmcwf.lzout.out_fd,
            hmc_epi=workflow.hmcwf.lzout.out_file,
            epi_parc=workflow.ema.lzout.epi_parc,
            mni_report=workflow.ema.lzout.report,
            in_iqms=workflow.iqmswf.lzout.out_file,
            in_dvars=workflow.iqmswf.lzout.out_dvars,
            outliers=workflow.iqmswf.lzout.outliers,
            meta_sidecar=workflow.meta.lzout.out_dict,
            name="func_report_wf",
        )
    )
    # fmt: off

    @pydra.mark.task
    def mean_out_file_callable(in_: str):
        return _pop(in_)

    workflow.add(mean_out_file_callable(in_=workflow.mean.lzout.out_file, name="mean_out_file"))

    workflow.ema.inputs.epi_mean = workflow.mean_out_file.lzout.out
    workflow.set_output([('out_fd', workflow.hmcwf.lzout.out_fd)])
    # fmt: on
    if wf_fft_spikes_detector:
        # fmt: off
        workflow.func_report_wf.inputs.in_spikes = workflow.iqmswf.lzout.out_spikes
        workflow.func_report_wf.inputs.in_fft = workflow.iqmswf.lzout.out_fft
        # fmt: on
    # population specific changes to brain masking
    if wf_species == "human":
        from pydra.tasks.mriqc.workflows.shared import (
            synthstrip_wf as fmri_bmsk_workflow,
        )

        workflow.add(
            fmri_bmsk_workflow(omp_nthreads=nipype_omp_nthreads)(name="skullstrip_epi")
        )
        # fmt: off

        @pydra.mark.task
        def mean_out_file_callable(in_: str):
            return _pop(in_)

        workflow.add(mean_out_file_callable(in_=workflow.mean.lzout.out_file, name="mean_out_file"))

        workflow.skullstrip_epi.inputs.in_files = workflow.mean_out_file.lzout.out
        workflow.ema.inputs.epi_mask = workflow.skullstrip_epi.lzout.out_mask
        workflow.iqmswf.inputs.brainmask = workflow.skullstrip_epi.lzout.out_mask
        workflow.func_report_wf.inputs.brainmask = workflow.skullstrip_epi.lzout.out_mask
        # fmt: on
    else:
        from pydra.tasks.mriqc.workflows.anatomical.base import _binarize

        workflow.add(
            niu.Function(
                input_names=["in_file", "threshold"],
                output_names=["out_file"],
                function=_binarize,
                name="binarise_labels",
            )
        )
        # fmt: off
        workflow.binarise_labels.inputs.in_file = workflow.ema.lzout.epi_parc
        workflow.iqmswf.inputs.brainmask = workflow.binarise_labels.lzout.out_file
        workflow.func_report_wf.inputs.brainmask = workflow.binarise_labels.lzout.out_file
        # fmt: on
    # Upload metrics
    if not exec_no_sub:
        from pydra.tasks.mriqc.interfaces.webapi import UploadIQMs

        workflow.add(
            UploadIQMs(
                endpoint=exec_webapi_url,
                auth_token=exec_webapi_token,
                strict=exec_upload_strict,
                name="upldwf",
            )
        )
        # fmt: off
        workflow.upldwf.inputs.in_iqms = workflow.iqmswf.lzout.out_file
        # fmt: on
    return workflow
