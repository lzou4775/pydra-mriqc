import attrs
from fileformats.generic import File
import logging
from pydra.tasks.mriqc.utils.misc import BIDS_COMP
from pathlib import Path
import pydra.mark
import simplejson as json
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate({"return": {"out_file": File}})
def IQMFileSink(
    in_file: str,
    subject_id: str,
    modality: str,
    session_id: ty.Any,
    task_id: ty.Any,
    acq_id: ty.Any,
    rec_id: ty.Any,
    run_id: ty.Any,
    dataset: str,
    dismiss_entities: list,
    metadata: dict,
    provenance: dict,
    root: dict,
    out_dir: Path,
    _outputs: dict,
) -> File:
    """
    Examples
    -------

    >>> from fileformats.generic import File
    >>> from pydra.tasks.mriqc.interfaces.bids.iqm_file_sink import IQMFileSink

    """

    out_file, out_file = _gen_outfile(
        out_dir=out_dir, in_file=in_file, dismiss_entities=dismiss_entities
    )

    if root is not attrs.NOTHING:
        _out_dict = root

    root_adds = []
    for key, val in list(_outputs.items()):
        if (val is attrs.NOTHING) or key == "trait_added":
            continue

        if self.expr.match(key) is not None:
            root_adds.append(key)
            continue

        key, val = _process_name(key, val)
        self._out_dict[key] = val

    for root_key in root_adds:
        val = _outputs.get(root_key, None)
        if isinstance(val, dict):
            self._out_dict.update(val)
        else:
            logger.warning(
                'Output "%s" is not a dictionary (value="%s"), ' "discarding output.",
                root_key,
                str(val),
            )

    # Fill in the "bids_meta" key
    id_dict = {}
    for comp in BIDS_COMP:
        comp_val = getattr(self.inputs, comp, None)
        if (comp_val is not attrs.NOTHING) and comp_val is not None:
            id_dict[comp] = comp_val
    id_dict["modality"] = modality

    if (metadata is not attrs.NOTHING) and metadata:
        id_dict.update(metadata)

    if self._out_dict.get("bids_meta") is None:
        self._out_dict["bids_meta"] = {}
    self._out_dict["bids_meta"].update(id_dict)

    if dataset is not attrs.NOTHING:
        self._out_dict["bids_meta"]["dataset"] = dataset

    # Fill in the "provenance" key
    # Predict QA from IQMs and add to metadata
    prov_dict = {}
    if (provenance is not attrs.NOTHING) and provenance:
        prov_dict.update(provenance)

    if self._out_dict.get("provenance") is None:
        self._out_dict["provenance"] = {}
    self._out_dict["provenance"].update(prov_dict)

    with open(out_file, "w") as f:
        f.write(
            json.dumps(
                self._out_dict,
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
            )
        )

    return out_file


# Nipype methods converted into functions


def _gen_outfile(out_dir=None, in_file=None, dismiss_entities=None):
    out_file = attrs.NOTHING
    out_dir = Path()
    if out_dir is not attrs.NOTHING:
        out_dir = Path(out_dir)

    # Crawl back to the BIDS root
    path = Path(in_file)
    for i in range(1, 4):
        if str(path.parents[i].name).startswith("sub-"):
            bids_root = path.parents[i + 1]
            break
    in_file = str(path.relative_to(bids_root))

    if (dismiss_entities is not attrs.NOTHING) and (dismiss := dismiss_entities):
        for entity in dismiss:
            bids_chunks = [
                chunk
                for chunk in path.name.split("_")
                if not chunk.startswith(f"{entity}-")
            ]
            path = path.parent / "_".join(bids_chunks)

    # Build path and ensure directory exists
    bids_path = out_dir / in_file.replace("".join(Path(in_file).suffixes), ".json")
    bids_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = str(bids_path)
    return out_file, out_file


def _process_name(name, val):
    if "." in name:
        newkeys = name.split(".")
        name = newkeys.pop(0)
        nested_dict = {newkeys.pop(): val}

        for nk in reversed(newkeys):
            nested_dict = {nk: nested_dict}
        val = nested_dict

    return name, val
