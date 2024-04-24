import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import hmc


logger = logging.getLogger(__name__)


def test_hmc():
    workflow = hmc()
    assert isinstance(workflow, Workflow)
