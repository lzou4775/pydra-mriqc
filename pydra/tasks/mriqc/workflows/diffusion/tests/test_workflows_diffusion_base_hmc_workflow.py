import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.diffusion.base import hmc_workflow


logger = logging.getLogger(__name__)


def test_hmc_workflow():
    workflow = hmc_workflow()
    assert isinstance(workflow, Workflow)
