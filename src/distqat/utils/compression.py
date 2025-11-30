# Code taken from https://github.com/PrimeIntellect-ai/OpenDiloco/blob/main/open_diloco/utils/compression.py

def get_compression_kwargs(hivemind_compression: str | None) -> dict:
    """Return the compression kwargs for hivemind optimizer based on the hivemind_compression argument."""
    ret_kwargs = {}

    if hivemind_compression is None:
        from hivemind.compression import NoCompression

        ret_kwargs["grad_compression"] = NoCompression()
        ret_kwargs["state_averaging_compression"] = NoCompression()

    elif hivemind_compression == "fp16":
        from hivemind.compression import Float16Compression

        ret_kwargs["grad_compression"] = Float16Compression()
        ret_kwargs["state_averaging_compression"] = Float16Compression()
    elif hivemind_compression == "scaled-fp16":
        from hivemind.compression import ScaledFloat16Compression

        ret_kwargs["grad_compression"] = ScaledFloat16Compression()
        ret_kwargs["state_averaging_compression"] = ScaledFloat16Compression()
    elif hivemind_compression == "uniform8bit":
        from hivemind.compression import Uniform8BitQuantization

        ret_kwargs["grad_compression"] = Uniform8BitQuantization()
        ret_kwargs["state_averaging_compression"] = Uniform8BitQuantization()
    elif hivemind_compression == "quantile8bit":
        from hivemind.compression import Quantile8BitQuantization

        ret_kwargs["grad_compression"] = Quantile8BitQuantization()
        ret_kwargs["state_averaging_compression"] = Quantile8BitQuantization()

    elif hivemind_compression == "blockwise8bit":
        from hivemind.compression import BlockwiseQuantization

        ret_kwargs["grad_compression"] = BlockwiseQuantization()
        ret_kwargs["state_averaging_compression"] = BlockwiseQuantization()
    else:
        raise ValueError(f"Invalid hivemind_compression: {hivemind_compression}")
    return ret_kwargs
