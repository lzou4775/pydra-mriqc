import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import anat_qc_workflow


logger = logging.getLogger(__name__)


def test_anat_qc_workflow():
    workflow = anat_qc_workflow()
    assert isinstance(workflow, Workflow)
