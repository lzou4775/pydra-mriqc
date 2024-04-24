import hashlib
import logging
import os.path as op


logger = logging.getLogger(__name__)


def fname_presuffix(fname, prefix="", suffix="", newpath=None, use_ext=True):
    """Manipulates path and name of input filename

    Parameters
    ----------
    fname : string
        A filename (may or may not include path)
    prefix : string
        Characters to prepend to the filename
    suffix : string
        Characters to append to the filename
    newpath : string
        Path to replace the path of the input fname
    use_ext : boolean
        If True (default), appends the extension of the original file
        to the output name.

    Returns
    -------
    Absolute path of the modified filename

    >>> from nipype.utils.filemanip import fname_presuffix
    >>> fname = 'foo.nii.gz'
    >>> fname_presuffix(fname,'pre','post','/tmp')
    '/tmp/prefoopost.nii.gz'

    >>> from nipype.interfaces.base import type(attrs.NOTHING)
    >>> fname_presuffix(fname, 'pre', 'post', type(attrs.NOTHING)) == \
            fname_presuffix(fname, 'pre', 'post')
    True

    """
    pth, fname, ext = split_filename(fname)
    if not use_ext:
        ext = ""

    # No need for : bool(type(attrs.NOTHING) is not attrs.NOTHING) evaluates to False
    if newpath:
        pth = op.abspath(newpath)
    return op.join(pth, prefix + fname + suffix + ext)


def hash_infile(afile, chunk_len=8192, crypto=hashlib.md5, raise_notfound=False):
    """
    Computes hash of a file using 'crypto' module

    >>> hash_infile('smri_ants_registration_settings.json')
    'f225785dfb0db9032aa5a0e4f2c730ad'

    >>> hash_infile('surf01.vtk')
    'fdf1cf359b4e346034372cdeb58f9a88'

    >>> hash_infile('spminfo')
    '0dc55e3888c98a182dab179b976dfffc'

    >>> hash_infile('fsl_motion_outliers_fd.txt')
    'defd1812c22405b1ee4431aac5bbdd73'


    """
    if not op.isfile(afile):
        if raise_notfound:
            raise RuntimeError('File "%s" not found.' % afile)
        return None

    crypto_obj = crypto()
    with open(afile, "rb") as fp:
        while True:
            data = fp.read(chunk_len)
            if not data:
                break
            crypto_obj.update(data)
    return crypto_obj.hexdigest()


def split_filename(fname):
    """Split a filename into parts: path, base filename and extension.

    Parameters
    ----------
    fname : str
        file or path name

    Returns
    -------
    pth : str
        base path from fname
    fname : str
        filename from fname, without extension
    ext : str
        file extension from fname

    Examples
    --------
    >>> from nipype.utils.filemanip import split_filename
    >>> pth, fname, ext = split_filename('/home/data/subject.nii.gz')
    >>> pth
    '/home/data'

    >>> fname
    'subject'

    >>> ext
    '.nii.gz'

    """

    special_extensions = [".nii.gz", ".tar.gz", ".niml.dset"]

    pth = op.dirname(fname)
    fname = op.basename(fname)

    ext = None
    for special_ext in special_extensions:
        ext_len = len(special_ext)
        if (len(fname) > ext_len) and (fname[-ext_len:].lower() == special_ext.lower()):
            ext = fname[-ext_len:]
            fname = fname[:-ext_len]
            break
    if not ext:
        fname, ext = op.splitext(fname)

    return pth, fname, ext
