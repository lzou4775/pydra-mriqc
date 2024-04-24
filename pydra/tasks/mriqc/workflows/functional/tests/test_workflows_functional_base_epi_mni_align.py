import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.base import epi_mni_align


logger = logging.getLogger(__name__)


def test_epi_mni_align():
    workflow = epi_mni_align()
    assert isinstance(workflow, Workflow)
