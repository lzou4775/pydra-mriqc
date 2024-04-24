import attrs
import logging
from pydra.engine import Workflow


logger = logging.getLogger(__name__)


def synthstrip_wf(in_files=attrs.NOTHING, name="synthstrip_wf", omp_nthreads=None):
    """Create a brain-extraction workflow using SynthStrip."""
    from pydra.tasks.ants.auto import N4BiasFieldCorrection
    from pydra.tasks.niworkflows.interfaces.nibabel import ApplyMask, IntensityClip
    from pydra.tasks.mriqc.interfaces.synthstrip import SynthStrip

    workflow = Workflow(name=name, input_spec=["in_files"], in_files=in_files)

    # truncate target intensity for N4 correction
    workflow.add(
        IntensityClip(
            p_min=10, p_max=99.9, in_files=workflow.lzin.in_files, name="pre_clip"
        )
    )
    workflow.add(
        N4BiasFieldCorrection(
            dimension=3,
            num_threads=omp_nthreads,
            rescale_intensities=True,
            copy_header=True,
            input_image=workflow.pre_clip.lzout.out_file,
            name="pre_n4",
        )
    )
    workflow.add(
        N4BiasFieldCorrection(
            dimension=3,
            save_bias=True,
            num_threads=omp_nthreads,
            n_iterations=[50] * 4,
            copy_header=True,
            input_image=workflow.pre_clip.lzout.out_file,
            name="post_n4",
        )
    )
    workflow.add(
        SynthStrip(
            num_threads=omp_nthreads,
            in_file=workflow.pre_n4.lzout.output_image,
            name="synthstrip",
        )
    )
    workflow.add(
        ApplyMask(
            in_mask=workflow.synthstrip.lzout.out_mask,
            in_file=workflow.post_n4.lzout.output_image,
            name="final_masked",
        )
    )
    # fmt: off
    workflow.post_n4.inputs.weight_image = workflow.synthstrip.lzout.out_mask
    workflow.set_output([('out_brain', workflow.final_masked.lzout.out_file)])
    workflow.set_output([('bias_image', workflow.post_n4.lzout.bias_image)])
    workflow.set_output([('out_mask', workflow.synthstrip.lzout.out_mask)])
    workflow.set_output([('out_corrected', workflow.post_n4.lzout.output_image)])
    # fmt: on
    return workflow
