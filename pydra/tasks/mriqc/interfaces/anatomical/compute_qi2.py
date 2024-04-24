from fileformats.generic import File
import logging
from pydra.tasks.mriqc.qc.anatomical import art_qi2
import nibabel as nb
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"qi2": float, "out_file": File}})
def ComputeQI2(in_file: File, air_msk: File) -> ty.Tuple[float, File]:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.anatomical.compute_qi2 import ComputeQI2

    """
    imdata = nb.load(in_file).get_fdata()
    airdata = nb.load(air_msk).get_fdata()
    qi2, out_file = art_qi2(imdata, airdata)
    qi2 = qi2
    out_file = out_file

    return qi2, out_file


# Nipype methods converted into functions
