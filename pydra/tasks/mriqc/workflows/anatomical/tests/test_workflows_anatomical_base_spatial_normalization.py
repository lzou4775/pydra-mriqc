import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import spatial_normalization


logger = logging.getLogger(__name__)


def test_spatial_normalization():
    workflow = spatial_normalization()
    assert isinstance(workflow, Workflow)
