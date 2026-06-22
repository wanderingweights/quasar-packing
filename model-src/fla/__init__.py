# Lightweight package initializer for the vendored FLA subset used by Quasar.
#
# The upstream file eagerly imports every layer and model, which is slow and can
# hang on fresh training containers while optional kernels are being resolved.
# Import concrete modules directly, e.g. `from fla.layers.quasar import ...`.

__version__ = "0.1.0"
__all__ = []
