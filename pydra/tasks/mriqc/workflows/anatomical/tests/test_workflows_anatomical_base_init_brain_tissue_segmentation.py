import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.base import init_brain_tissue_segmentation


logger = logging.getLogger(__name__)


def test_init_brain_tissue_segmentation():
    workflow = init_brain_tissue_segmentation()
    assert isinstance(workflow, Workflow)
