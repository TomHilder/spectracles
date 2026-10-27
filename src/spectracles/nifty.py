"""nifty.py - NIFTy.re adapter (optional: pip install spectracles[nifty]).

Importing this module never imports NIFTy; only the functions that need it do.
"""


def _import_nifty():
    """Import nifty.re, with a helpful error if it isn't installed."""
    try:
        import nifty.re as jft
    except ImportError as e:
        raise ImportError(
            "This needs NIFTy.re, which is an optional dependency of spectracles. "
            "Install it with: pip install 'spectracles[nifty]'"
        ) from e
    return jft
