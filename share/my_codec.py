import ctypes
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Self

import numpy as np
from rmn._sharedlib import librmn
from rmn.fstrecord import FstDataType, fst_record, numpy_type_to_fst_type
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

logging.basicConfig(level=logging.DEBUG)
_log = logging.getLogger(__name__)


_fst24_decode_data_rsf = librmn.fst24_decode_data_rsf
_fst24_decode_data_rsf.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
_fst24_decode_data_rsf.restype = fst_record

_fst24_decode_data_xdf = librmn.fst24_decode_data_xdf
_fst24_decode_data_xdf.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
_fst24_decode_data_xdf.restype = fst_record


@dataclass(frozen=True)
class FSTCodec(ArrayBytesCodec):
    async def _decode_single(
        self, input_buffer: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        raw_data = input_buffer.to_bytes()
        
        record = _fst24_decode_data_xdf(raw_data, None)
        
        #if not record._data:
        #    raise RuntimeError("FST decoding failed: returned null pointer")

        count = record.ni * record.nj * record.nk

        if record.data_bits > 32:
            ptr_type = ctypes.POINTER(ctypes.c_double)
        else:
            ptr_type = ctypes.POINTER(ctypes.c_float)

        data_ptr = ctypes.cast(record._data, ptr_type)
        
        array_3d = np.ctypeslib.as_array(data_ptr, shape=(count,)).reshape(
            chunk_spec.shape, order="F"
        )

        return chunk_spec.prototype.nd_buffer.from_numpy_array(array_3d)
        
    # ==========================================================================
    # BATCH PROCESSING
    # ==========================================================================
    async def encode(
        self,
        chunks_and_specs: Iterable[tuple[Optional[NDBuffer], ArraySpec]],
    ) -> Iterable[Optional[Buffer]]:
        chunks_and_specs = list(chunks_and_specs)
        print(f"[encode] {len(chunks_and_specs)} chunk(s)")

        for i, (_, spec) in enumerate(chunks_and_specs):
            print(f"  chunk[{i}] shape={spec.shape}, dtype={spec.dtype}")

        return await super().encode(chunks_and_specs)

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Optional[Buffer], ArraySpec]],
    ) -> Iterable[Optional[NDBuffer]]:
        chunks_and_specs = list(chunks_and_specs)
        print(f"[decode] {len(chunks_and_specs)} chunk(s)")

        for i, (_, spec) in enumerate(chunks_and_specs):
            print(f"  chunk[{i}] shape={spec.shape}, dtype={spec.dtype}")

        return await super().decode(chunks_and_specs)

    # ================================================================================
    # REQUIRED METADATA METHODS
    # ================================================================================
    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        """
        Calculates the maximum expected size (in bytes) of the chunk after encoding.
        If the size is unpredictable (like with most compression algorithms),
        raise a NotImplementedError to force dynamic allocation.
        """
        print(
            f"[compute_encoded_size] input_byte_length={input_byte_length}, shape={chunk_spec.shape}"
        )
        raise NotImplementedError("FST encoded size is data-dependent.")

    # ================================================================================
    # OPTIONAL METADATA METHODS
    # ================================================================================
    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        """
        (Optional) Used to check that the codec metadata is compatible with the
        array metadata. It should raise errors if not.
        """
        print(f"[validate] shape={shape}, dtype={dtype}, chunk_grid={chunk_grid}")

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """
        (Optional) Important for codecs that change the shape, dtype or
        fill value of a chunk.
        """
        print(f"[resolve_metadata] shape={chunk_spec.shape}, dtype={chunk_spec.dtype}")
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """
        (Optional) Useful for automatically filling the codec configuration metadata
        from the array metadata.
        """
        print(
            f"[evolve_from_array_spec] shape={array_spec.shape}, dtype={array_spec.dtype}"
        )
        return self

    @classmethod
    def from_dict(cls, data):
        config = data.get("configuration", {})
        return cls(**config)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import stat

_FD_PATH_PREFIX = "/dev/fd/"


# def _write_record_to_bytes(rec: fst_record) -> bytes:
#     # 1. Get a unique temporary path
#     fd, tmp_path = tempfile.mkstemp(prefix="fst_chunk_write_", suffix=".fst")
#     os.close(fd)
#
#     # 2. DELETE the 0-byte file so librmn can create it from scratch with proper headers
#     os.remove(tmp_path)
#
#     try:
#         # Let the C library create the file and write the record
#         with fst24_file(tmp_path, "w") as f:
#             f.write(rec)
#
#         # Read the resulting encoded bytes back into Python
#         with open(tmp_path, "rb") as f_raw:
#             return f_raw.read()
#     finally:
#         if os.path.exists(tmp_path):
#             try:
#                 os.remove(tmp_path)
#             except OSError:
#                 pass


def _read_record_from_bytes(payload: bytes) -> fst_record:
    # 1. Create a C-compatible byte array directly from the Python bytes
    c_buffer = (ctypes.c_char * len(payload)).from_buffer_copy(payload)

    # 2. Get a void pointer to the memory buffer
    buffer_ptr = ctypes.cast(c_buffer, ctypes.c_void_p)

    # 3. Decode directly in memory using the function you already imported!
    # (Note: The second argument is usually a context pointer, passing None/0 typically works
    # for default extraction, but adjust if librmn expects a specific struct pointer here).
    rec = _fst24_decode_data_xdf(buffer_ptr, None)

    # 4. Fallback check if the C library failed to return a valid record
    # Depending on how the ctypes wrapper is structured, you might need to check if rec.data is null
    if not rec or not rec.data:
        raise RuntimeError("Failed to decode in-memory FST payload.")

    return rec


def _shape_to_nijk(shape: tuple[int, ...]) -> tuple[int, int, int]:
    ndim = len(shape)
    if ndim == 1:
        return shape[0], 1, 1
    if ndim == 2:
        return shape[0], shape[1], 1
    if ndim == 3:
        return shape[0], shape[1], shape[2]
    ni = 1
    for s in shape[:-2]:
        ni *= s
    return ni, shape[-2], shape[-1]


def _fst_dtype_and_bits(
    codec_type: FstDataType, dtype: np.dtype
) -> tuple[FstDataType, int]:
    possible_types, data_bits = numpy_type_to_fst_type(dtype.name)
    if codec_type in possible_types:
        return codec_type, data_bits
    _log.debug(
        "data_type=%s incompatible with dtype=%s, falling back to %s",
        codec_type.name,
        dtype,
        possible_types[0].name,
    )
    return possible_types[0], data_bits
