import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import compute_iqms


logger = logging.getLogger(__name__)


def test_compute_iqms():
    workflow = compute_iqms()
    assert isinstance(workflow, Workflow)
