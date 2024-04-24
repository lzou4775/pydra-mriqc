import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.anatomical.output import init_anat_report_wf


logger = logging.getLogger(__name__)


def test_init_anat_report_wf():
    workflow = init_anat_report_wf()
    assert isinstance(workflow, Workflow)
