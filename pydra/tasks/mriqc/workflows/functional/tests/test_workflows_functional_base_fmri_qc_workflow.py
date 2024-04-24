import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import fmri_qc_workflow


logger = logging.getLogger(__name__)


def test_fmri_qc_workflow():
    workflow = fmri_qc_workflow()
    assert isinstance(workflow, Workflow)
