import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.diffusion.output import init_dwi_report_wf


logger = logging.getLogger(__name__)


def test_init_dwi_report_wf():
    workflow = init_dwi_report_wf()
    assert isinstance(workflow, Workflow)
