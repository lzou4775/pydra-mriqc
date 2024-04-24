import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.diffusion.base import dmri_qc_workflow


logger = logging.getLogger(__name__)


def test_dmri_qc_workflow():
    workflow = dmri_qc_workflow()
    assert isinstance(workflow, Workflow)
