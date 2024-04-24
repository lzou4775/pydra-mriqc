import logging
from pathlib import Path
from pydra.engine import Workflow
import pydra.mark
from pydra.tasks.mriqc.interfaces import (
    ArtifactMask,
    ComputeQI2,
    ConformImage,
    IQMFileSink,
    RotationMask,
    StructuralQC,
)
from pydra.tasks.mriqc.interfaces.reports import AddProvenance
from pydra.tasks.mriqc.workflows.anatomical.output import init_anat_report_wf
from pydra.tasks.mriqc.workflows.utils import get_fwhmx
from pydra.tasks.niworkflows.interfaces.fixes import (
    FixHeaderApplyTransforms as ApplyTransforms,
)
from templateflow.api import get as get_template


logger = logging.getLogger(__name__)


def anat_qc_workflow(
    exec_ants_float=False,
    exec_bids_database_dir=None,
    exec_datalad_get=True,
    exec_debug=False,
    exec_dsname="<unset>",
    exec_no_sub=False,
    exec_output_dir=None,
    exec_upload_strict=False,
    exec_verbose_reports=False,
    exec_webapi_token="<secret_token>",
    exec_webapi_url="https://mriqc.nimh.nih.gov:443/api/v1",
    exec_work_dir=None,
    name="anatMRIQC",
    nipype_omp_nthreads=10,
    wf_inputs=None,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """
    One-subject-one-session-one-run pipeline to extract the NR-IQMs from
    anatomical images

    .. workflow::

        import os.path as op
        from mriqc.workflows.anatomical.base import anat_qc_workflow
        from mriqc.testing import mock_config
        with mock_config():
            wf = anat_qc_workflow()

    """
    from pydra.tasks.mriqc.workflows.shared import synthstrip_wf

    if exec_work_dir is None:
        exec_work_dir = Path.cwd()

    workflow = Workflow(name=name, input_spec=["in_file"])

    dataset = wf_inputs.get("t1w", []) + wf_inputs.get("t2w", [])
    message = "Building {modality} MRIQC workflow {detail}.".format(
        modality="anatomical",
        detail=(
            f"for {len(dataset)} NIfTI files."
            if len(dataset) > 2
            else f"({' and '.join('<%s>' % v for v in dataset)})."
        ),
    )
    logger.info(message)
    if exec_datalad_get:
        from pydra.tasks.mriqc.utils.misc import _datalad_get

        _datalad_get(dataset)
    # Initialize workflow
    # Define workflow, inputs and outputs
    # 0. Get data

    # 1. Reorient anatomical image
    workflow.add(
        ConformImage(check_dtype=False, in_file=workflow.lzin.in_file, name="to_ras")
    )
    # 2. species specific skull-stripping
    if wf_species.lower() == "human":
        workflow.add(
            synthstrip_wf(omp_nthreads=nipype_omp_nthreads)(
                in_files=workflow.to_ras.lzout.out_file, name="skull_stripping"
            )
        )
        ss_bias_field = "outputnode.bias_image"
    else:
        from nirodents.workflows.brainextraction import init_rodent_brain_extraction_wf

        skull_stripping = init_rodent_brain_extraction_wf(template_id=wf_template_id)
        ss_bias_field = "final_n4.bias_image"
    # 3. Head mask
    workflow.add(
        headmsk_wf(omp_nthreads=nipype_omp_nthreads, wf_species=wf_species)(name="hmsk")
    )
    # 4. Spatial Normalization, using ANTs
    workflow.add(
        spatial_normalization(
            exec_ants_float=exec_ants_float,
            exec_debug=exec_debug,
            nipype_omp_nthreads=nipype_omp_nthreads,
            wf_species=wf_species,
            wf_template_id=wf_template_id,
        )(name="norm")
    )
    # 5. Air mask (with and without artifacts)
    workflow.add(
        airmsk_wf()(
            ind2std_xfm=workflow.norm.lzout.ind2std_xfm,
            in_file=workflow.to_ras.lzout.out_file,
            head_mask=workflow.hmsk.lzout.out_file,
            name="amw",
        )
    )
    # 6. Brain tissue segmentation
    workflow.add(
        init_brain_tissue_segmentation(nipype_omp_nthreads=nipype_omp_nthreads)(
            std_tpms=workflow.norm.lzout.out_tpms,
            in_file=workflow.hmsk.lzout.out_denoised,
            name="bts",
        )
    )
    # 7. Compute IQMs
    workflow.add(
        compute_iqms(
            exec_bids_database_dir=exec_bids_database_dir,
            exec_dsname=exec_dsname,
            exec_output_dir=exec_output_dir,
            wf_species=wf_species,
        )(
            in_file=workflow.lzin.in_file,
            std_tpms=workflow.norm.lzout.out_tpms,
            in_ras=workflow.to_ras.lzout.out_file,
            airmask=workflow.amw.lzout.air_mask,
            hatmask=workflow.amw.lzout.hat_mask,
            artmask=workflow.amw.lzout.art_mask,
            rotmask=workflow.amw.lzout.rot_mask,
            segmentation=workflow.bts.lzout.out_segm,
            pvms=workflow.bts.lzout.out_pvms,
            headmask=workflow.hmsk.lzout.out_file,
            name="iqmswf",
        )
    )
    # Reports
    workflow.add(
        init_anat_report_wf(
            exec_verbose_reports=exec_verbose_reports,
            exec_work_dir=exec_work_dir,
            wf_species=wf_species,
        )(
            in_file=workflow.lzin.in_file,
            mni_report=workflow.norm.lzout.out_report,
            in_ras=workflow.to_ras.lzout.out_file,
            headmask=workflow.hmsk.lzout.out_file,
            airmask=workflow.amw.lzout.air_mask,
            artmask=workflow.amw.lzout.art_mask,
            rotmask=workflow.amw.lzout.rot_mask,
            segmentation=workflow.bts.lzout.out_segm,
            noisefit=workflow.iqmswf.lzout.noisefit,
            in_iqms=workflow.iqmswf.lzout.out_file,
            name="anat_report_wf",
        )
    )
    # Connect all nodes
    # fmt: off

    @pydra.mark.task
    def inputnode_in_file_callable(in_: str):
        return _get_mod(in_)

    workflow.add(inputnode_in_file_callable(in_=workflow.lzin.in_file, name="inputnode_in_file"))

    workflow.norm.inputs.modality = workflow.inputnode_in_file.lzout.out
    workflow.hmsk.inputs.in_file = workflow.skull_stripping.lzout.out_corrected
    workflow.hmsk.inputs.brainmask = workflow.skull_stripping.lzout.out_mask
    workflow.bts.inputs.brainmask = workflow.skull_stripping.lzout.out_mask
    workflow.norm.inputs.moving_image = workflow.skull_stripping.lzout.out_corrected
    workflow.norm.inputs.moving_mask = workflow.skull_stripping.lzout.out_mask
    workflow.hmsk.inputs.in_tpms = workflow.norm.lzout.out_tpms
    workflow.amw.inputs.in_mask = workflow.skull_stripping.lzout.out_mask
    workflow.iqmswf.inputs.inu_corrected = workflow.skull_stripping.lzout.out_corrected
    workflow.iqmswf.inputs.in_inu = getattr(workflow.skull_stripping.lzout, ss_bias_field)
    workflow.iqmswf.inputs.brainmask = workflow.skull_stripping.lzout.out_mask
    workflow.anat_report_wf.inputs.inu_corrected = workflow.skull_stripping.lzout.out_corrected
    workflow.anat_report_wf.inputs.brainmask = workflow.skull_stripping.lzout.out_mask
    workflow.set_output([('out_json', workflow.iqmswf.lzout.out_file)])
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
        workflow.anat_report_wf.inputs.api_id = workflow.upldwf.lzout.api_id
        # fmt: on
    return workflow


def airmsk_wf(name="AirMaskWorkflow"):
    """
    Calculate air, artifacts and "hat" masks to evaluate noise in the background.

    This workflow mostly addresses the implementation of Step 1 in [Mortamet2009]_.
    This work proposes to look at the signal distribution in the background, where
    no signals are expected, to evaluate the spread of the noise.
    It is in the background where [Mortamet2009]_ proposed to also look at the presence
    of ghosts and artifacts, where they are very easy to isolate.

    However, [Mortamet2009]_ proposes not to look at the background around the face
    because of the likely signal leakage through the phase-encoding axis sourcing from
    eyeballs (and their motion).
    To avoid that, [Mortamet2009]_ proposed atlas-based identification of two landmarks
    (nasion and cerebellar projection on to the occipital bone).
    MRIQC, for simplicity, used a such a mask created in MNI152NLin2009cAsym space and
    projected it on to the individual.
    Such a solution is inadequate because it doesn't drop full in-plane slices as there
    will be a large rotation of the individual's tilt of the head with respect to the
    template.
    The new implementation (23.1.x series) follows [Mortamet2009]_ more closely,
    projecting the two landmarks from the template space and leveraging
    *NiTransforms* to do that.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical.base import airmsk_wf
        with mock_config():
            wf = airmsk_wf()

    """
    workflow = Workflow(name=name, input_spec=["head_mask", "in_file", "ind2std_xfm"])

    workflow.add(RotationMask(in_file=workflow.lzin.in_file, name="rotmsk"))
    workflow.add(
        ArtifactMask(
            in_file=workflow.lzin.in_file,
            head_mask=workflow.lzin.head_mask,
            ind2std_xfm=workflow.lzin.ind2std_xfm,
            name="qi1",
        )
    )
    # fmt: off
    workflow.set_output([('hat_mask', workflow.qi1.lzout.out_hat_msk)])
    workflow.set_output([('air_mask', workflow.qi1.lzout.out_air_msk)])
    workflow.set_output([('art_mask', workflow.qi1.lzout.out_art_msk)])
    workflow.set_output([('rot_mask', workflow.rotmsk.lzout.out_file)])
    # fmt: on
    return workflow


def headmsk_wf(name="HeadMaskWorkflow", omp_nthreads=1, wf_species="human"):
    """
    Computes a head mask as in [Mortamet2009]_.

    .. workflow::

        from mriqc.testing import mock_config
        from mriqc.workflows.anatomical.base import headmsk_wf
        with mock_config():
            wf = headmsk_wf()

    """
    from pydra.tasks.niworkflows.interfaces.nibabel import ApplyMask

    workflow = Workflow(name=name, input_spec=["brainmask", "in_file", "in_tpms"])

    def _select_wm(inlist):
        return [f for f in inlist if "WM" in f][0]

    workflow.add(
        niu.Function(
            input_names=["in_file", "wm_tpm"],
            output_names=["out_file"],
            function=_enhance,
            in_file=workflow.lzin.in_file,
            name="enhance",
        )
    )
    workflow.add(
        niu.Function(
            input_names=["in_file", "brainmask", "sigma"],
            output_names=["out_file"],
            function=image_gradient,
            brainmask=workflow.lzin.brainmask,
            in_file=workflow.enhance.lzout.out_file,
            name="gradient",
        )
    )
    workflow.add(
        niu.Function(
            input_names=["in_file", "brainmask", "aniso", "thresh"],
            output_names=["out_file"],
            function=gradient_threshold,
            brainmask=workflow.lzin.brainmask,
            in_file=workflow.gradient.lzout.out_file,
            name="thresh",
        )
    )
    if wf_species != "human":
        workflow.gradient.inputs.sigma = 3.0
        workflow.thresh.inputs.aniso = True
        workflow.thresh.inputs.thresh = 4.0
    workflow.add(
        ApplyMask(
            brainmask=workflow.lzin.brainmask,
            in_file=workflow.enhance.lzout.out_file,
            name="apply_mask",
        )
    )
    # fmt: off

    @pydra.mark.task
    def inputnode_in_tpms_callable(in_: str):
        return _select_wm(in_)

    workflow.add(inputnode_in_tpms_callable(in_=workflow.lzin.in_tpms, name="inputnode_in_tpms"))

    workflow.enhance.inputs.wm_tpm = workflow.inputnode_in_tpms.lzout.out
    workflow.set_output([('out_file', workflow.thresh.lzout.out_file)])
    workflow.set_output([('out_denoised', workflow.apply_mask.lzout.out_file)])
    # fmt: on
    return workflow


def init_brain_tissue_segmentation(
    name="brain_tissue_segmentation", nipype_omp_nthreads=10
):
    """
    Setup a workflow for brain tissue segmentation.

    .. workflow::

        from mriqc.workflows.anatomical.base import init_brain_tissue_segmentation
        from mriqc.testing import mock_config
        with mock_config():
            wf = init_brain_tissue_segmentation()

    """
    from pydra.tasks.ants.auto import Atropos

    workflow = Workflow(name=name, input_spec=["brainmask", "in_file", "std_tpms"])

    def _format_tpm_names(in_files, fname_string=None):
        import glob
        from pathlib import Path
        import nibabel as nb

        out_path = Path.cwd().absolute()
        # copy files to cwd and rename iteratively
        for count, fname in enumerate(in_files):
            img = nb.load(fname)
            extension = "".join(Path(fname).suffixes)
            out_fname = f"priors_{1 + count:02}{extension}"
            nb.save(img, Path(out_path, out_fname))
        if fname_string is None:
            fname_string = f"priors_%02d{extension}"
        out_files = [
            str(prior)
            for prior in glob.glob(str(Path(out_path, f"priors*{extension}")))
        ]
        # return path with c-style format string for Atropos
        file_format = str(Path(out_path, fname_string))
        return file_format, out_files

    workflow.add(
        niu.Function(
            input_names=["in_files"],
            output_names=["file_format"],
            function=_format_tpm_names,
            execution={"keep_inputs": True, "remove_unnecessary_outputs": False},
            std_tpms=workflow.lzin.std_tpms,
            name="format_tpm_names",
        )
    )
    workflow.add(
        Atropos(
            initialization="PriorProbabilityImages",
            number_of_tissue_classes=3,
            prior_weighting=0.1,
            mrf_radius=[1, 1, 1],
            mrf_smoothing_factor=0.01,
            save_posteriors=True,
            out_classified_image_name="segment.nii.gz",
            output_posteriors_name_template="segment_%02d.nii.gz",
            num_threads=nipype_omp_nthreads,
            in_file=workflow.lzin.in_file,
            brainmask=workflow.lzin.brainmask,
            name="segment",
        )
    )
    # fmt: off

    @pydra.mark.task
    def format_tpm_names_file_format_callable(in_: str):
        return _pop(in_)

    workflow.add(format_tpm_names_file_format_callable(in_=workflow.format_tpm_names.lzout.file_format, name="format_tpm_names_file_format"))

    workflow.segment.inputs.prior_image = workflow.format_tpm_names_file_format.lzout.out
    workflow.set_output([('out_segm', workflow.segment.lzout.classified_image)])
    workflow.set_output([('out_pvms', workflow.segment.lzout.posteriors)])
    # fmt: on
    return workflow


def spatial_normalization(
    exec_ants_float=False,
    exec_debug=False,
    name="SpatialNormalization",
    nipype_omp_nthreads=10,
    wf_species="human",
    wf_template_id="MNI152NLin2009cAsym",
):
    """Create a simplified workflow to perform fast spatial normalization."""
    from pydra.tasks.niworkflows.interfaces.reportlets.registration import (
        SpatialNormalizationRPT as RobustMNINormalization,
    )

    # Have the template id handy
    workflow = Workflow(
        name=name, input_spec=["modality", "moving_image", "moving_mask"]
    )

    tpl_id = wf_template_id
    # Define workflow interface

    # Spatial normalization
    workflow.add(
        RobustMNINormalization(
            flavor=["testing", "fast"][exec_debug],
            num_threads=nipype_omp_nthreads,
            float=exec_ants_float,
            template=tpl_id,
            generate_report=True,
            moving_image=workflow.lzin.moving_image,
            moving_mask=workflow.lzin.moving_mask,
            modality=workflow.lzin.modality,
            name="norm",
        )
    )
    if wf_species.lower() == "human":
        workflow.norm.inputs.reference_mask = str(
            get_template(tpl_id, resolution=2, desc="brain", suffix="mask")
        )
    else:
        workflow.norm.inputs.reference_image = str(get_template(tpl_id, suffix="T2w"))
        workflow.norm.inputs.reference_mask = str(
            get_template(tpl_id, desc="brain", suffix="mask")[0]
        )
    # Project standard TPMs into T1w space
    workflow.add(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            interpolation="Gaussian",
            float=exec_ants_float,
            moving_image=workflow.lzin.moving_image,
            transforms=workflow.norm.lzout.inverse_composite_transform,
            name="tpms_std2t1w",
        )
    )
    workflow.tpms_std2t1w.inputs.input_image = [
        str(p)
        for p in get_template(
            wf_template_id,
            suffix="probseg",
            resolution=(1 if wf_species.lower() == "human" else None),
            label=["CSF", "GM", "WM"],
        )
    ]
    # fmt: off
    workflow.set_output([('ind2std_xfm', workflow.norm.lzout.composite_transform)])
    workflow.set_output([('out_report', workflow.norm.lzout.out_report)])
    workflow.set_output([('out_tpms', workflow.tpms_std2t1w.lzout.output_image)])
    # fmt: on
    return workflow


def compute_iqms(
    exec_bids_database_dir=None,
    exec_dsname="<unset>",
    exec_output_dir=None,
    name="ComputeIQMs",
    wf_species="human",
):
    """
    Setup the workflow that actually computes the IQMs.

    .. workflow::

        from mriqc.workflows.anatomical.base import compute_iqms
        from mriqc.testing import mock_config
        with mock_config():
            wf = compute_iqms()

    """
    from pydra.tasks.niworkflows.interfaces.bids import ReadSidecarJSON
    from pydra.tasks.mriqc.interfaces.anatomical import Harmonize
    from pydra.tasks.mriqc.workflows.utils import _tofloat

    workflow = Workflow(
        name=name,
        input_spec=[
            "airmask",
            "artmask",
            "brainmask",
            "hatmask",
            "headmask",
            "in_file",
            "in_inu",
            "in_ras",
            "inu_corrected",
            "pvms",
            "rotmask",
            "segmentation",
            "std_tpms",
        ],
    )

    # Extract metadata
    workflow.add(
        ReadSidecarJSON(
            index_db=exec_bids_database_dir, in_file=workflow.lzin.in_file, name="meta"
        )
    )
    # Add provenance
    workflow.add(
        AddProvenance(
            in_file=workflow.lzin.in_file,
            airmask=workflow.lzin.airmask,
            rotmask=workflow.lzin.rotmask,
            name="addprov",
        )
    )
    # AFNI check smoothing
    fwhm_interface = get_fwhmx()
    workflow.add(
        fwhm_interface(
            in_ras=workflow.lzin.in_ras, brainmask=workflow.lzin.brainmask, name="fwhm"
        )
    )
    # Harmonize
    workflow.add(Harmonize(inu_corrected=workflow.lzin.inu_corrected, name="homog"))
    if wf_species.lower() != "human":
        workflow.homog.inputs.erodemsk = False
        workflow.homog.inputs.thresh = 0.8
    # Mortamet's QI2
    workflow.add(
        ComputeQI2(
            in_ras=workflow.lzin.in_ras, hatmask=workflow.lzin.hatmask, name="getqi2"
        )
    )
    # Compute python-coded measures
    workflow.add(
        StructuralQC(
            human=wf_species.lower() == "human",
            in_inu=workflow.lzin.in_inu,
            in_ras=workflow.lzin.in_ras,
            airmask=workflow.lzin.airmask,
            headmask=workflow.lzin.headmask,
            artmask=workflow.lzin.artmask,
            rotmask=workflow.lzin.rotmask,
            segmentation=workflow.lzin.segmentation,
            pvms=workflow.lzin.pvms,
            std_tpms=workflow.lzin.std_tpms,
            in_noinu=workflow.homog.lzout.out_file,
            name="measures",
        )
    )
    workflow.add(
        IQMFileSink(
            out_dir=exec_output_dir,
            dataset=exec_dsname,
            in_file=workflow.lzin.in_file,
            subject_id=workflow.meta.lzout.subject,
            session_id=workflow.meta.lzout.session,
            task_id=workflow.meta.lzout.task,
            acq_id=workflow.meta.lzout.acquisition,
            rec_id=workflow.meta.lzout.reconstruction,
            run_id=workflow.meta.lzout.run,
            metadata=workflow.meta.lzout.out_dict,
            root=workflow.measures.lzout.out_qc,
            provenance=workflow.addprov.lzout.out_prov,
            qi_2=workflow.getqi2.lzout.qi2,
            name="datasink",
        )
    )

    def _getwm(inlist):
        return inlist[-1]

    # fmt: off

    @pydra.mark.task
    def inputnode_in_file_callable(in_: str):
        return _get_mod(in_)

    workflow.add(inputnode_in_file_callable(in_=workflow.lzin.in_file, name="inputnode_in_file"))

    workflow.datasink.inputs.modality = workflow.inputnode_in_file.lzout.out

    @pydra.mark.task
    def inputnode_in_file_callable(in_: str):
        return _get_mod(in_)

    workflow.add(inputnode_in_file_callable(in_=workflow.lzin.in_file, name="inputnode_in_file"))

    workflow.addprov.inputs.modality = workflow.inputnode_in_file.lzout.out

    @pydra.mark.task
    def inputnode_pvms_callable(in_: str):
        return _getwm(in_)

    workflow.add(inputnode_pvms_callable(in_=workflow.lzin.pvms, name="inputnode_pvms"))

    workflow.homog.inputs.wm_mask = workflow.inputnode_pvms.lzout.out

    @pydra.mark.task
    def fwhm_fwhm_callable(in_: str):
        return _tofloat(in_)

    workflow.add(fwhm_fwhm_callable(in_=workflow.fwhm.lzout.fwhm, name="fwhm_fwhm"))

    workflow.measures.inputs.in_fwhm = workflow.fwhm_fwhm.lzout.out
    workflow.set_output([('noisefit', workflow.getqi2.lzout.out_file)])
    workflow.set_output([('out_file', workflow.datasink.lzout.out_file)])
    # fmt: on
    return workflow


def _enhance(in_file, wm_tpm, out_file=None):
    import nibabel as nb
    import numpy as np

    from mriqc.workflows.utils import generate_filename

    imnii = nb.load(in_file)
    data = imnii.get_fdata(dtype=np.float32)
    range_max = np.percentile(data[data > 0], 99.98)
    excess = data > range_max

    wm_prob = nb.load(wm_tpm).get_fdata()
    wm_prob[wm_prob < 0] = 0  # Ensure no negative values
    wm_prob[excess] = 0  # Ensure no outliers are considered

    # Calculate weighted mean and standard deviation
    wm_mu = np.average(data, weights=wm_prob)
    wm_sigma = np.sqrt(np.average((data - wm_mu) ** 2, weights=wm_prob))

    # Resample signal excess pixels
    data[excess] = np.random.normal(loc=wm_mu, scale=wm_sigma, size=excess.sum())

    out_file = out_file or str(generate_filename(in_file, suffix="enhanced").absolute())
    nb.Nifti1Image(data, imnii.affine, imnii.header).to_filename(out_file)
    return out_file


def _get_mod(in_file):
    from pathlib import Path

    in_file = Path(in_file)
    extension = "".join(in_file.suffixes)
    return in_file.name.replace(extension, "").split("_")[-1]


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def gradient_threshold(in_file, brainmask, thresh=15.0, out_file=None, aniso=False):
    """Compute a threshold from the histogram of the magnitude gradient image"""
    import nibabel as nb
    import numpy as np
    from scipy import ndimage as sim

    from mriqc.workflows.utils import generate_filename

    if not aniso:
        struct = sim.iterate_structure(sim.generate_binary_structure(3, 2), 2)
    else:
        # Generate an anisotropic binary structure, taking into account slice thickness
        img = nb.load(in_file)
        zooms = img.header.get_zooms()
        dist = max(zooms)
        dim = img.header["dim"][0]

        x = np.ones((5) * np.ones(dim, dtype=np.int8))
        np.put(x, x.size // 2, 0)
        dist_matrix = np.round(sim.distance_transform_edt(x, sampling=zooms), 5)
        struct = dist_matrix <= dist

    imnii = nb.load(in_file)

    hdr = imnii.header.copy()
    hdr.set_data_dtype(np.uint8)

    data = imnii.get_fdata(dtype=np.float32)

    mask = np.zeros_like(data, dtype=np.uint8)
    mask[data > thresh] = 1
    mask = sim.binary_closing(mask, struct, iterations=2).astype(np.uint8)
    mask = sim.binary_erosion(mask, sim.generate_binary_structure(3, 2)).astype(
        np.uint8
    )

    segdata = np.asanyarray(nb.load(brainmask).dataobj) > 0
    segdata = sim.binary_dilation(segdata, struct, iterations=2, border_value=1).astype(
        np.uint8
    )
    mask[segdata] = 1

    # Remove small objects
    label_im, nb_labels = sim.label(mask)
    artmsk = np.zeros_like(mask)
    if nb_labels > 2:
        sizes = sim.sum(mask, label_im, list(range(nb_labels + 1)))
        ordered = sorted(zip(sizes, list(range(nb_labels + 1))), reverse=True)
        for _, label in ordered[2:]:
            mask[label_im == label] = 0
            artmsk[label_im == label] = 1

    mask = sim.binary_fill_holes(mask, struct).astype(
        np.uint8
    )  # pylint: disable=no-member

    out_file = out_file or str(generate_filename(in_file, suffix="gradmask").absolute())
    nb.Nifti1Image(mask, imnii.affine, hdr).to_filename(out_file)
    return out_file


def image_gradient(in_file, brainmask, sigma=4.0, out_file=None):
    """Computes the magnitude gradient of an image using numpy"""
    import nibabel as nb
    import numpy as np
    from scipy.ndimage import gaussian_gradient_magnitude as gradient

    from mriqc.workflows.utils import generate_filename

    imnii = nb.load(in_file)
    mask = np.bool_(nb.load(brainmask).dataobj)
    data = imnii.get_fdata(dtype=np.float32)
    datamax = np.percentile(data.reshape(-1), 99.5)
    data *= 100 / datamax
    data[mask] = 100

    zooms = np.array(imnii.header.get_zooms()[:3])
    sigma_xyz = 2 - zooms / min(zooms)
    grad = gradient(data, sigma * sigma_xyz)
    gradmax = np.percentile(grad.reshape(-1), 99.5)
    grad *= 100.0
    grad /= gradmax
    grad[mask] = 100

    out_file = out_file or str(generate_filename(in_file, suffix="grad").absolute())
    nb.Nifti1Image(grad, imnii.affine, imnii.header).to_filename(out_file)
    return out_file


def _binarize(in_file, threshold=0.5, out_file=None):
    import os.path as op

    import nibabel as nb
    import numpy as np

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, ext2 = op.splitext(fname)
            ext = ext2 + ext
        out_file = op.abspath(f"{fname}_bin{ext}")

    nii = nb.load(in_file)
    data = nii.get_fdata() > threshold

    hdr = nii.header.copy()
    hdr.set_data_dtype(np.uint8)
    nb.Nifti1Image(data.astype(np.uint8), nii.affine, hdr).to_filename(out_file)
    return out_file
