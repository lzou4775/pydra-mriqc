import attrs
from fileformats.generic import Directory, File
import logging
import os
from pathlib import Path
from pydra.engine.specs import MultiInputObj, MultiOutputType
import pydra.mark
import typing as ty


logger = logging.getLogger(__name__)


@pydra.mark.task
@pydra.mark.annotate(
    {
        "return": {
            "out_file": ty.List[File],
            "out_meta": ty.List[File],
            "compression": ty.Union[list, object, MultiOutputType],
            "fixed_hdr": list,
        }
    }
)
def DerivativesDataSink(
    base_directory: Directory,
    check_hdr: bool,
    compress: MultiInputObj,
    data_dtype: str,
    dismiss_entities: MultiInputObj,
    in_file: ty.List[File],
    meta_dict: dict,
    source_file: ty.List[File],
) -> ty.Tuple[
    ty.List[File], ty.List[File], ty.Union[list, object, MultiOutputType], list
]:
    """
    Examples
    -------

    >>> from fileformats.generic import Directory, File
    >>> from pydra.engine.specs import MultiOutputType, MultiInputObj
    >>> from pydra.tasks.mriqc.interfaces.derivatives_data_sink import DerivativesDataSink

    """
    from bids.layout import parse_file_entities, Config
    from bids.layout.writing import build_path
    from bids.utils import listify

    # Ready the output folder
    base_directory = os.getcwd()
    if base_directory is not attrs.NOTHING:
        base_directory = base_directory
    base_directory = Path(base_directory).absolute()
    out_path = base_directory / self.out_path_base
    out_path.mkdir(exist_ok=True, parents=True)

    # Ensure we have a list
    in_file = listify(in_file)

    # Read in the dictionary of metadata
    if meta_dict is not attrs.NOTHING:
        meta = meta_dict
        # inputs passed in construction take priority
        meta.update(self._metadata)
        _metadata = meta

    # Initialize entities with those from the source file.
    custom_config = Config(
        name="custom",
        entities=self._config_entities_dict,
        default_path_patterns=self._file_patterns,
    )
    in_entities = [
        parse_file_entities(
            str(relative_to_root(source_file)),
            config=["bids", "derivatives", custom_config],
        )
        for source_file in source_file
    ]
    out_entities = {
        k: v
        for k, v in in_entities[0].items()
        if all(ent.get(k) == v for ent in in_entities[1:])
    }
    for drop_entity in listify(dismiss_entities or []):
        out_entities.pop(drop_entity, None)

    # Override extension with that of the input file(s)
    out_entities["extension"] = [
        # _splitext does not accept .surf.gii (for instance)
        "".join(Path(orig_file).suffixes).lstrip(".")
        for orig_file in in_file
    ]

    compress = listify(compress) or [None]
    if len(compress) == 1:
        compress = compress * len(in_file)
    for i, ext in enumerate(out_entities["extension"]):
        if compress[i] is not None:
            ext = regz.sub("", ext)
            out_entities["extension"][i] = f"{ext}.gz" if compress[i] else ext

    # Override entities with those set as inputs
    for key in self._allowed_entities:
        value = getattr(self.inputs, key)
        if value is not None and (value is not attrs.NOTHING):
            out_entities[key] = value

    # Clean up native resolution with space
    if out_entities.get("resolution") == "native" and out_entities.get("space"):
        out_entities.pop("resolution", None)

    # Expand templateflow resolutions
    resolution = out_entities.get("resolution")
    space = out_entities.get("space")
    if resolution:
        # Standard spaces
        if space in self._standard_spaces:
            res = _get_tf_resolution(space, resolution)
        else:  # TODO: Nonstandard?
            res = "Unknown"
        self._metadata["Resolution"] = res

    if len(set(out_entities["extension"])) == 1:
        out_entities["extension"] = out_entities["extension"][0]

    # Insert custom (non-BIDS) entities from allowed_entities.
    custom_entities = set(out_entities) - set(self._config_entities)
    patterns = self._file_patterns
    if custom_entities:
        # Example: f"{key}-{{{key}}}" -> "task-{task}"
        custom_pat = "_".join(f"{key}-{{{key}}}" for key in sorted(custom_entities))
        patterns = [
            pat.replace("_{suffix", "_".join(("", custom_pat, "{suffix")))
            for pat in patterns
        ]

    # Prepare SimpleInterface outputs object
    out_file = []
    compression = []
    fixed_hdr = [False] * len(in_file)

    dest_files = build_path(out_entities, path_patterns=patterns)
    if not dest_files:
        raise ValueError(f"Could not build path with entities {out_entities}.")

    # Make sure the interpolated values is embedded in a list, and check
    dest_files = listify(dest_files)
    if len(in_file) != len(dest_files):
        raise ValueError(
            f"Input files ({len(in_file)}) not matched "
            f"by interpolated patterns ({len(dest_files)})."
        )

    for i, (orig_file, dest_file) in enumerate(zip(in_file, dest_files)):
        out_file = out_path / dest_file
        out_file.parent.mkdir(exist_ok=True, parents=True)
        out_file.append(str(out_file))
        compression.append(str(dest_file).endswith(".gz"))

        # An odd but possible case is that an input file is in the location of
        # the output and we have made no changes to it.
        # The primary use case is pre-computed derivatives where the output
        # directory will be filled in.
        # From a provenance perspective, I would rather inputs and outputs be
        # cleanly separated, but that is better handled by warnings at the CLI
        # level than a crash in a datasink.
        try:
            if os.path.samefile(orig_file, out_file):
                continue
        except FileNotFoundError:
            pass

        # Set data and header iff changes need to be made. If these are
        # still None when it's time to write, just copy.
        new_data, new_header = None, None

        is_nifti = False
        with suppress(nb.filebasedimages.ImageFileError):
            is_nifti = isinstance(nb.load(orig_file), nb.Nifti1Image)

        data_dtype = data_dtype or self._default_dtypes[suffix]
        if is_nifti and any((check_hdr, data_dtype)):
            nii = nb.load(orig_file)

            if check_hdr:
                hdr = nii.header
                curr_units = tuple(
                    [None if u == "unknown" else u for u in hdr.get_xyzt_units()]
                )
                curr_codes = (int(hdr["qform_code"]), int(hdr["sform_code"]))

                # Default to mm, use sec if data type is bold
                units = (
                    curr_units[0] or "mm",
                    "sec" if out_entities["suffix"] == "bold" else None,
                )
                xcodes = (1, 1)  # Derivative in its original scanner space
                if space:
                    xcodes = (4, 4) if space in self._standard_spaces else (2, 2)

                curr_zooms = zooms = hdr.get_zooms()
                if "RepetitionTime" in self.inputs.get():
                    zooms = curr_zooms[:3] + (RepetitionTime,)

                if (curr_codes, curr_units, curr_zooms) != (xcodes, units, zooms):
                    fixed_hdr[i] = True
                    new_header = hdr.copy()
                    new_header.set_qform(nii.affine, xcodes[0])
                    new_header.set_sform(nii.affine, xcodes[1])
                    new_header.set_xyzt_units(*units)
                    new_header.set_zooms(zooms)

            if data_dtype == "source":  # match source dtype
                try:
                    data_dtype = nb.load(source_file[0]).get_data_dtype()
                except Exception:
                    LOGGER.warning(f"Could not get data type of file {source_file[0]}")
                    data_dtype = None

            if data_dtype:
                data_dtype = np.dtype(data_dtype)
                orig_dtype = nii.get_data_dtype()
                if orig_dtype != data_dtype:
                    LOGGER.warning(
                        f"Changing {out_file} dtype from {orig_dtype} to {data_dtype}"
                    )
                    # coerce dataobj to new data dtype
                    if np.issubdtype(data_dtype, np.integer):
                        new_data = np.rint(nii.dataobj).astype(data_dtype)
                    else:
                        new_data = np.asanyarray(nii.dataobj, dtype=data_dtype)
                    # and set header to match
                    if new_header is None:
                        new_header = nii.header.copy()
                    new_header.set_data_dtype(data_dtype)
            del nii

        unlink(out_file, missing_ok=True)
        if new_data is new_header is None:
            _copy_any(orig_file, str(out_file))
        else:
            orig_img = nb.load(orig_file)
            if new_data is None:
                set_consumables(new_header, orig_img.dataobj)
                new_data = orig_img.dataobj.get_unscaled()
            else:
                # Without this, we would be writing nans
                # This is our punishment for hacking around nibabel defaults
                new_header.set_slope_inter(slope=1.0, inter=0.0)
            unsafe_write_nifti_header_and_data(
                fname=out_file, header=new_header, data=new_data
            )
            del orig_img

    if len(out_file) == 1:
        meta_fields = self.inputs.copyable_trait_names()
        self._metadata.update(
            {
                k: getattr(self.inputs, k)
                for k in meta_fields
                if k not in self._static_traits
            }
        )
        if self._metadata:
            sidecar = out_file.parent / f"{out_file.name.split('.', 1)[0]}.json"
            unlink(sidecar, missing_ok=True)
            sidecar.write_text(dumps(self._metadata, sort_keys=True, indent=2))
            out_meta = str(sidecar)

    return out_file, out_meta, compression, fixed_hdr


# Nipype methods converted into functions
