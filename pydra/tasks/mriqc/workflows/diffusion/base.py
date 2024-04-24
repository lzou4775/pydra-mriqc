import logging
import numpy as np
from pathlib import Path
from pydra.engine import Workflow
import pydra.mark


logger = logging.getLogger(__name__)


def dmri_qc_workflow(
    exec_ants_float=False,
    exec_bids_database_dir="/Users/lekangzoukang/Data/ds000114/sub-02/ses-test/anat/sub-02-ses-test-T1w.nii.gz ",
    exec_datalad_get=True,
    exec_debug=False,
    exec_dsname="<unset>",
    exec_float32=True,
    exec_layout=None,
    exec_output_dir=None,
    exec_verbose_reports=False,
    exec_work_dir=None,
    name="dwiMRIQC",
    nipype_nprocs=10,
    nipype_omp_nthreads=10,
    wf_biggest_file_gb=1,
    wf_fd_radius=50,
    wf_fd_thres=0.2,
    wf_fft_spikes_detector=False,
    wf_inputs=None,  # dataset leave it as it is at this moment
    wf_min_len_dwi=7,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    Initialize the dMRI-QC workflow.

    .. workflow::

        import os.path as op
        from mriqc.workflows.diffusion.base import dmri_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = dmri_qc_workflow()

    """
    from pydra.tasks.afni.auto import Volreg
    from pydra.tasks.mrtrix3.v3_0 import DwiDenoise
    from pydra.tasks.niworkflows.interfaces.header import SanitizeImage
    from pydra.tasks.niworkflows.interfaces.images import RobustAverage
    from pydra.tasks.mriqc.interfaces.diffusion import (
        CCSegmentation,
        CorrectSignalDrift,
        DiffusionModel,
        ExtractOrientations,
        NumberOfShells,
        PIESNO,
        ReadDWIMetadata,
        SpikingVoxelsMask,
        WeightedStat,
    )

    from pydra.tasks.mriqc.workflows.shared import synthstrip_wf as dmri_bmsk_workflow

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(name=name, input_spec=["in_file"])

    ### Dataset set to run before running with the workflow

    # dataset = wf_inputs.get("dwi", [])
    # full_data = []
    # for dwi_path in dataset:
    #     bval = exec_layout.get_bval(dwi_path)
    #     if bval and Path(bval).exists() and len(np.loadtxt(bval)) > wf_min_len_dwi:
    #         full_data.append(dwi_path)
    #     else:
    #         logger.warn(
    #             f"Dismissing {dwi_path} for processing. b-values are missing or "
    #             "insufficient in number to execute the workflow."
    #         )
    # if set(dataset) - set(full_data):
    #     wf_inputs["dwi"] = full_data

    ### Full_data not defined because of the previous section comment (*** will use later)

    # message = "Building {modality} MRIQC workflow {detail}.".format(
    #     modality="diffusion",
    #     detail=(
    #         f"for {len(full_data)} NIfTI files."
    #         if len(full_data) > 2
    #         else f"({' and '.join('<%s>' % v for v in full_data)})."
    #     ),
    # )
    # logger.info(message)
    # if exec_datalad_get:
    #     from pydra.tasks.mriqc.utils.misc import _datalad_get

    #     _datalad_get(full_data)

    # Define workflow, inputs and outputs
    # 0. Get data, put it in RAS orientation

    workflow.add(
        SanitizeImage(
            n_volumes_to_discard=0,
            max_32bit=exec_float32,
            in_file=workflow.lzin.in_file,
            name="sanitize",
        )
    )
    # Workflow --------------------------------------------------------
    # Read metadata & bvec/bval, estimate number of shells, extract and split B0s
    workflow.add(
        ReadDWIMetadata(
            index_db=exec_bids_database_dir,
            in_file=workflow.lzin.in_file,
            name="load_bmat",
        )
    )
    workflow.add(
        NumberOfShells(in_bvals=workflow.load_bmat.lzout.out_bval_file, name="shells")
    )
    workflow.add(
        ExtractOrientations(in_file=workflow.sanitize.lzout.out_file, name="get_lowb")
    )
    # Generate B0 reference
    workflow.add(
        RobustAverage(
            mc_method=None, in_file=workflow.sanitize.lzout.out_file, name="dwi_ref"
        )
    )
    workflow.add(
        Volreg(
            args="-Fourier -twopass",
            zpad=4,
            outputtype="NIFTI_GZ",
            in_file=workflow.get_lowb.lzout.out_file,
            basefile=workflow.dwi_ref.lzout.out_file,
            name="hmc_b0",
        )
    )
    # Calculate brainmask
    workflow.add(
        dmri_bmsk_workflow(omp_nthreads=nipype_omp_nthreads)(
            in_files=workflow.dwi_ref.lzout.out_file, name="dmri_bmsk"
        )
    )
    # HMC: head motion correct
    workflow.add(
        hmc_workflow(wf_fd_radius=wf_fd_radius)(
            in_bvec=workflow.load_bmat.lzout.out_bvec_file, name="hmcwf"
        )
    )
    workflow.add(
        ExtractOrientations(
            in_bvec_file=workflow.load_bmat.lzout.out_bvec_file,
            indices=workflow.shells.lzout.b_indices,
            in_file=workflow.hmcwf.lzout.out_file,
            name="get_hmc_shells",
        )
    )
    # Split shells and compute some stats
    workflow.add(
        WeightedStat(in_weights=workflow.shells.lzout.b_masks, name="averages")
    )
    workflow.add(
        WeightedStat(
            stat="std", in_weights=workflow.shells.lzout.b_masks, name="stddev"
        )
    )
    workflow.add(
        DWIDenoise(
            noise="noisemap.nii.gz",
            nthreads=nipype_omp_nthreads,
            mask=workflow.dmri_bmsk.lzout.out_mask,
            name="dwidenoise",
        )
    )
    workflow.add(
        CorrectSignalDrift(
            full_epi=workflow.sanitize.lzout.out_file,
            bval_file=workflow.load_bmat.lzout.out_bval_file,
            in_file=workflow.hmc_b0.lzout.out_file,
            brainmask_file=workflow.dmri_bmsk.lzout.out_mask,
            name="drift",
        )
    )
    workflow.add(
        SpikingVoxelsMask(
            in_file=workflow.sanitize.lzout.out_file,
            b_masks=workflow.shells.lzout.b_masks,
            brain_mask=workflow.dmri_bmsk.lzout.out_mask,
            name="sp_mask",
        )
    )
    # Fit DTI/DKI model
    workflow.add(
        DiffusionModel(
            bvals=workflow.shells.lzout.out_data,
            n_shells=workflow.shells.lzout.n_shells,
            bvec_file=workflow.load_bmat.lzout.out_bvec_file,
            in_file=workflow.dwidenoise.lzout.out_file,
            brain_mask=workflow.dmri_bmsk.lzout.out_mask,
            name="dwimodel",
        )
    )
    # Calculate CC mask
    workflow.add(
        CCSegmentation(
            in_fa=workflow.dwimodel.lzout.out_fa,
            in_cfa=workflow.dwimodel.lzout.out_cfa,
            name="cc_mask",
        )
    )
    # Run PIESNO noise estimation
    workflow.add(PIESNO(in_file=workflow.sanitize.lzout.out_file, name="piesno"))
    # EPI to MNI registration
    workflow.add(
        epi_mni_align(
            exec_ants_float=exec_ants_float,
            exec_debug=exec_debug,
            nipype_nprocs=nipype_nprocs,
            nipype_omp_nthreads=nipype_omp_nthreads,
            wf_species=wf_species,
            wf_template_id=wf_template_id,
        )(
            epi_mean=workflow.dwi_ref.lzout.out_file,
            epi_mask=workflow.dmri_bmsk.lzout.out_mask,
            name="spatial_norm",
        )
    )
    # Compute IQMs
    workflow.add(
        compute_iqms(
            exec_bids_database_dir=exec_bids_database_dir,
            exec_dsname=exec_dsname,
            exec_output_dir=exec_output_dir,
        )(
            in_file=workflow.lzin.in_file,
            b_values_file=workflow.load_bmat.lzout.out_bval_file,
            qspace_neighbors=workflow.load_bmat.lzout.qspace_neighbors,
            spikes_mask=workflow.sp_mask.lzout.out_mask,
            piesno_sigma=workflow.piesno.lzout.sigma,
            framewise_displacement=workflow.hmcwf.lzout.out_fd,
            in_bvec_rotated=workflow.hmcwf.lzout.out_bvec,
            in_bvec_diff=workflow.hmcwf.lzout.out_bvec_diff,
            in_fa=workflow.dwimodel.lzout.out_fa,
            in_cfa=workflow.dwimodel.lzout.out_cfa,
            in_fa_nans=workflow.dwimodel.lzout.out_fa_nans,
            in_fa_degenerate=workflow.dwimodel.lzout.out_fa_degenerate,
            in_md=workflow.dwimodel.lzout.out_md,
            brain_mask=workflow.dmri_bmsk.lzout.out_mask,
            cc_mask=workflow.cc_mask.lzout.out_mask,
            wm_mask=workflow.cc_mask.lzout.wm_finalmask,
            n_shells=workflow.shells.lzout.n_shells,
            b_values_shells=workflow.shells.lzout.b_values,
            in_shells=workflow.get_hmc_shells.lzout.out_file,
            in_bvec=workflow.get_hmc_shells.lzout.out_bvec,
            in_noise=workflow.dwidenoise.lzout.noise,
            name="iqms_wf",
        )
    )
    # Generate outputs

    # fmt: off

    @pydra.mark.task
    def shells_b_masks_callable(in_: str):
        return _first(in_)

    workflow.add(shells_b_masks_callable(in_=workflow.shells.lzout.b_masks, name="shells_b_masks"))

    workflow.dwi_ref.inputs.t_mask = workflow.shells_b_masks.lzout.out

    @pydra.mark.task
    def shells_b_indices_callable(in_: str):
        return _first(in_)

    workflow.add(shells_b_indices_callable(in_=workflow.shells.lzout.b_indices, name="shells_b_indices"))

    workflow.get_lowb.inputs.indices = workflow.shells_b_indices.lzout.out

    @pydra.mark.task
    def shells_b_indices_callable(in_: str):
        return _first(in_)

    workflow.add(shells_b_indices_callable(in_=workflow.shells.lzout.b_indices, name="shells_b_indices"))

    workflow.drift.inputs.b0_ixs = workflow.shells_b_indices.lzout.out
    workflow.hmcwf.inputs.in_file = workflow.drift.lzout.out_full_file
    workflow.averages.inputs.in_file = workflow.drift.lzout.out_full_file
    workflow.stddev.inputs.in_file = workflow.drift.lzout.out_full_file

    @pydra.mark.task
    def averages_out_file_callable(in_: str):
        return _first(in_)

    workflow.add(averages_out_file_callable(in_=workflow.averages.lzout.out_file, name="averages_out_file"))

    workflow.hmcwf.inputs.reference = workflow.averages_out_file.lzout.out
    workflow.dwidenoise.inputs.in_file = workflow.drift.lzout.out_full_file

    @pydra.mark.task
    def averages_out_file_callable(in_: str):
        return _first(in_)

    workflow.add(averages_out_file_callable(in_=workflow.averages.lzout.out_file, name="averages_out_file"))

    workflow.iqms_wf.inputs.in_b0 = workflow.averages_out_file.lzout.out
    # fmt: on
    return workflow


def hmc_workflow(name="dMRI_HMC", wf_fd_radius=50):
    """
    Create a :abbr:`HMC (head motion correction)` workflow for dMRI.

    .. workflow::

        from mriqc.workflows.diffusion.base import hmc
        from mriqc.testing import mock_config
        with mock_config():
            wf = hmc()

    """
    from pydra.tasks.mriqc.nipype_ports.algorithms.confounds import (
        FramewiseDisplacement,
    )
    from pydra.tasks.afni.auto import Volreg
    from pydra.tasks.mriqc.interfaces.diffusion import RotateVectors

    workflow = Workflow(name=name, input_spec=["in_bvec", "in_file", "reference"])

    # calculate hmc parameters
    workflow.add(
        Volreg(
            args="-Fourier -twopass",
            zpad=4,
            outputtype="NIFTI_GZ",
            in_file=workflow.lzin.in_file,
            reference=workflow.lzin.reference,
            name="hmc",
        )
    )
    workflow.add(
        RotateVectors(
            in_bvec=workflow.lzin.in_bvec,
            reference=workflow.lzin.reference,
            transforms=workflow.hmc.lzout.oned_matrix_save,
            name="bvec_rot",
        )
    )
    # Compute the frame-wise displacement
    workflow.add(
        FramewiseDisplacement(
            normalize=False,
            parameter_source="AFNI",
            radius=wf_fd_radius,
            in_file=workflow.hmc.lzout.oned_file,
            name="fdnode",
        )
    )
    # fmt: off
    workflow.set_output([('out_file', workflow.hmc.lzout.out_file)])
    workflow.set_output([('out_fd', workflow.fdnode.lzout.out_file)])
    workflow.set_output([('out_bvec', workflow.bvec_rot.lzout.out_bvec)])
    workflow.set_output([('out_bvec_diff', workflow.bvec_rot.lzout.out_diff)])
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

        from mriqc.workflows.diffusion.base import epi_mni_align
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


def compute_iqms(
    exec_bids_database_dir=None,
    exec_dsname="<unset>",
    exec_output_dir=None,
    name="ComputeIQMs",
):
    """
    Initialize the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.diffusion.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.niworkflows.interfaces.bids import ReadSidecarJSON
    from pydra.tasks.mriqc.interfaces import IQMFileSink
    from pydra.tasks.mriqc.interfaces.diffusion import DiffusionQC
    from pydra.tasks.mriqc.interfaces.reports import AddProvenance

    # from mriqc.workflows.utils import _tofloat, get_fwhmx
    workflow = Workflow(
        name=name,
        input_spec=[
            "b_values_file",
            "b_values_shells",
            "brain_mask",
            "cc_mask",
            "framewise_displacement",
            "in_b0",
            "in_bvec",
            "in_bvec_diff",
            "in_bvec_rotated",
            "in_cfa",
            "in_fa",
            "in_fa_degenerate",
            "in_fa_nans",
            "in_file",
            "in_md",
            "in_noise",
            "in_shells",
            "n_shells",
            "piesno_sigma",
            "qspace_neighbors",
            "spikes_mask",
            "wm_mask",
        ],
    )

    # niu function needs re-definition
    workflow.add(
        niu.Function(
            function=_estimate_sigma,
            in_noise=workflow.lzin.in_noise,
            brain_mask=workflow.lzin.brain_mask,
            name="estimate_sigma",
        )
    )
    workflow.add(
        ReadSidecarJSON(
            index_db=exec_bids_database_dir, in_file=workflow.lzin.in_file, name="meta"
        )
    )
    workflow.add(
        DiffusionQC(
            in_file=workflow.lzin.in_file,
            b_values_file=workflow.lzin.b_values_file,
            b_values_shells=workflow.lzin.b_values_shells,
            in_shells=workflow.lzin.in_shells,
            in_bvec=workflow.lzin.in_bvec,
            in_bvec_rotated=workflow.lzin.in_bvec_rotated,
            in_bvec_diff=workflow.lzin.in_bvec_diff,
            in_b0=workflow.lzin.in_b0,
            brain_mask=workflow.lzin.brain_mask,
            wm_mask=workflow.lzin.wm_mask,
            cc_mask=workflow.lzin.cc_mask,
            spikes_mask=workflow.lzin.spikes_mask,
            in_fa=workflow.lzin.in_fa,
            in_md=workflow.lzin.in_md,
            in_cfa=workflow.lzin.in_cfa,
            in_fa_nans=workflow.lzin.in_fa_nans,
            in_fa_degenerate=workflow.lzin.in_fa_degenerate,
            framewise_displacement=workflow.lzin.framewise_displacement,
            qspace_neighbors=workflow.lzin.qspace_neighbors,
            piesno_sigma=workflow.lzin.piesno_sigma,
            noise_floor=workflow.estimate_sigma.lzout.out,
            name="measures",
        )
    )
    workflow.add(
        AddProvenance(modality="dwi", in_file=workflow.lzin.in_file, name="addprov")
    )
    # Save to JSON file
    workflow.add(
        IQMFileSink(
            modality="dwi",
            out_dir=str(exec_output_dir),
            dataset=exec_dsname,
            in_file=workflow.lzin.in_file,
            n_shells=workflow.lzin.n_shells,
            b_values_shells=workflow.lzin.b_values_shells,
            provenance=workflow.addprov.lzout.out_prov,
            subject_id=workflow.meta.lzout.subject,
            session_id=workflow.meta.lzout.session,
            task_id=workflow.meta.lzout.task,
            acq_id=workflow.meta.lzout.acquisition,
            rec_id=workflow.meta.lzout.reconstruction,
            run_id=workflow.meta.lzout.run,
            root=workflow.measures.lzout.out_qc,
            name="datasink",
        )
    )
    # fmt: off

    @pydra.mark.task
    def inputnode_b_values_file_callable(in_: str):
        return _bvals_report(in_)

    workflow.add(inputnode_b_values_file_callable(in_=workflow.lzin.b_values_file, name="inputnode_b_values_file"))

    workflow.datasink.inputs.bValues = workflow.inputnode_b_values_file.lzout.out

    @pydra.mark.task
    def meta_out_dict_callable(in_: str):
        return _filter_metadata(in_)

    workflow.add(meta_out_dict_callable(in_=workflow.meta.lzout.out_dict, name="meta_out_dict"))

    workflow.datasink.inputs.metadata = workflow.meta_out_dict.lzout.out
    workflow.set_output([('out_file', workflow.datasink.lzout.out_file)])
    workflow.set_output([('meta_sidecar', workflow.meta.lzout.out_dict)])
    workflow.set_output([('noise_floor', workflow.estimate_sigma.lzout.out)])
    # fmt: on
    return workflow


def _bvals_report(in_file):
    import numpy as np

    bvals = [
        round(float(val), 2) for val in np.unique(np.round(np.loadtxt(in_file), 2))
    ]

    if len(bvals) > 10:
        return "Likely DSI"

    return bvals


def _estimate_sigma(in_file, mask):
    import nibabel as nb
    import numpy as np

    msk = nb.load(mask).get_fdata() > 0.5
    return round(
        float(np.median(nb.load(in_file).get_fdata()[msk])),
        6,
    )


def _filter_metadata(
    in_dict,
    keys=(
        "global",
        "dcmmeta_affine",
        "dcmmeta_reorient_transform",
        "dcmmeta_shape",
        "dcmmeta_slice_dim",
        "dcmmeta_version",
        "time",
    ),
):
    """Drop large and partially redundant objects generated by dcm2niix."""

    for key in keys:
        in_dict.pop(key, None)

    return in_dict


def _first(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]

    return inlist
