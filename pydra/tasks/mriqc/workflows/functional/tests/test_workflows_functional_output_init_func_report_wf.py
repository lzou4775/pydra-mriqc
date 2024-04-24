import logging
from pydra.engine import Workflow
from pydra.tasks.mriqc.workflows.functional.output import init_func_report_wf


logger = logging.getLogger(__name__)


def test_init_func_report_wf():
    workflow = init_func_report_wf()
    assert isinstance(workflow, Workflow)
