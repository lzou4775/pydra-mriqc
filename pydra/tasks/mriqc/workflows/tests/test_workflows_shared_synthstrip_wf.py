import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.shared import synthstrip_wf


logger = logging.getLogger(__name__)


def test_synthstrip_wf():
    workflow = synthstrip_wf()
    assert isinstance(workflow, Workflow)
