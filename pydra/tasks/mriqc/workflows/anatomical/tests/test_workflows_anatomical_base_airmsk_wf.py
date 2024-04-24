import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import airmsk_wf


logger = logging.getLogger(__name__)


def test_airmsk_wf():
    workflow = airmsk_wf()
    assert isinstance(workflow, Workflow)
