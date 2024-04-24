import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import headmsk_wf


logger = logging.getLogger(__name__)


def test_headmsk_wf():
    workflow = headmsk_wf()
    assert isinstance(workflow, Workflow)
