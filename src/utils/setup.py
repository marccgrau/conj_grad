from keras import backend as K

# noinspection PyTypeChecker
DTYPE: str = None


def set_dtype(_dtype: str):
    global DTYPE

    if _dtype in {"16", "32", "64"}:
        DTYPE = f"float{_dtype}"
    elif _dtype in {"float16", "float32", "float64"}:
        DTYPE = _dtype
    else:
        raise TypeError(f"This dtype is not understood: {_dtype!r}")

    K.set_floatx(DTYPE)
    return DTYPE
