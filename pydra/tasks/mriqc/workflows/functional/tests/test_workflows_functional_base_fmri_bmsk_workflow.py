import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import fmri_bmsk_workflow


logger = logging.getLogger(__name__)


def test_fmri_bmsk_workflow():
    workflow = fmri_bmsk_workflow()
    assert isinstance(workflow, Workflow)
