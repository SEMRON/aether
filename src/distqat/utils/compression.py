from hivemind.proto.runtime_pb2 import CompressionType
from torch import Block

# # Code adapted from https://github.com/PrimeIntellect-ai/OpenDiloco/blob/main/open_diloco/utils/compression.py
def get_compression_kwargs(hivemind_compression: str | None) -> dict:
    """Return the compression kwargs for hivemind optimizer based on the hivemind_compression argument."""
    ret_kwargs = {}

    if hivemind_compression is None:
        from hivemind.compression import NoCompression

        ret_kwargs["grad_compression"] = CompressionType.NONE
        ret_kwargs["state_averaging_compression"] = NoCompression()
        ret_kwargs["state_compression"] = NoCompression()
    elif hivemind_compression == "fp16":
        from hivemind.compression import Float16Compression

        ret_kwargs["grad_compression"] = CompressionType.FLOAT16
        ret_kwargs["state_averaging_compression"] = Float16Compression()
        ret_kwargs["state_compression"] = Float16Compression()
    elif hivemind_compression == "scaled-fp16":
        from hivemind.compression import ScaledFloat16Compression

        ret_kwargs["grad_compression"] = CompressionType.MEANSTD_16BIT
        ret_kwargs["state_averaging_compression"] = ScaledFloat16Compression()
        ret_kwargs["state_compression"] = ScaledFloat16Compression()
    elif hivemind_compression == "uniform8bit":
        from hivemind.compression import Uniform8BitQuantization

        ret_kwargs["grad_compression"] = CompressionType.UNIFORM_8BIT
        ret_kwargs["state_averaging_compression"] = Uniform8BitQuantization()
        ret_kwargs["state_compression"] = Uniform8BitQuantization()
    elif hivemind_compression == "quantile8bit":
        from hivemind.compression import Quantile8BitQuantization

        ret_kwargs["grad_compression"] = CompressionType.QUANTILE_8BIT
        ret_kwargs["state_averaging_compression"] = Quantile8BitQuantization()
        ret_kwargs["state_compression"] = Quantile8BitQuantization()
    elif hivemind_compression == "blockwise8bit":
        from hivemind.compression import BlockwiseQuantization

        ret_kwargs["grad_compression"] = CompressionType.BLOCKWISE_8BIT
        ret_kwargs["state_averaging_compression"] = BlockwiseQuantization()
        ret_kwargs["state_compression"] = BlockwiseQuantization()
    elif hivemind_compression == "size-adaptive":
        from hivemind.compression import SizeAdaptiveCompression

        ret_kwargs["grad_compression"] = CompressionType.FLOAT16
        ret_kwargs["state_averaging_compression"] = SizeAdaptiveCompression(threshold=2 ** 16 + 1, less=Float16Compression(),
                    greater_equal=Uniform8BitQuantization())
        ret_kwargs["state_compression"] = SizeAdaptiveCompression(threshold=2 ** 16 + 1, less=Float16Compression(),
                    greater_equal=Uniform8BitQuantization())
    elif hivemind_compression == "fp16and8bit":
        from hivemind.compression import Float16Compression, Uniform8BitQuantization

        ret_kwargs["grad_compression"] = CompressionType.FLOAT16
        ret_kwargs["state_averaging_compression"] = Uniform8BitQuantization()
        ret_kwargs["state_compression"] = Uniform8BitQuantization()
    else:
        raise ValueError(f"Invalid hivemind_compression: {hivemind_compression}")
    return ret_kwargs


# def parse_runtime_compression(hivemind_compression: str | None) -> int:
#     """
#     Map distqat config strings to hivemind.proto.runtime_pb2.CompressionType.

#     Note: runtime_pb2.CompressionType is a protobuf EnumTypeWrapper; you cannot call it like CompressionType(x).
#     """
#     if hivemind_compression is None or hivemind_compression == "none":
#         return CompressionType.NONE

#     mapping = {
#         "fp16": CompressionType.FLOAT16,
#         "scaled-fp16": CompressionType.MEANSTD_16BIT,
#         "quantile8bit": CompressionType.QUANTILE_8BIT,
#         "uniform8bit": CompressionType.UNIFORM_8BIT,
#         "blockwise8bit": CompressionType.BLOCKWISE_8BIT,
#     }
#     try:
#         return mapping[hivemind_compression]
#     except KeyError:
#         raise ValueError(f"Invalid hivemind_compression for runtime tensor compression: {hivemind_compression!r}")