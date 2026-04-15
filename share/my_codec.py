import ctypes
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Self

import numpy as np
from rmn._sharedlib import librmn
from rmn.fstrecord import fst_record
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
    fst_format: str = "XDF"
    
    async def _decode_single(
        self, input_buffer: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        print(f"[_decode_single] self={self}, input_buffer={input_buffer}, chunk_spec={chunk_spec}")
              
        raw_bytes = input_buffer.to_bytes()
        
        if self.fst_format.upper() == "RSF":
            record = _fst24_decode_data_rsf(raw_bytes, None)
        else:
            record = _fst24_decode_data_xdf(raw_bytes, None)
        
        #if not record.data:
        #    raise RuntimeError("FST decoding failed: returned null pointer")

        c_type = ctypes.c_double if record.data_bits > 32 else ctypes.c_float
        data_ptr = ctypes.cast(record._data, ctypes.POINTER(c_type))
        
        count = record.ni * record.nj * record.nk
        array_3d = np.ctypeslib.as_array(data_ptr, shape=(count,)).reshape(
            chunk_spec.shape, order="F"
        )

        return chunk_spec.prototype.nd_buffer.from_numpy_array(array_3d)

    async def decode(
        self,
        chunks_and_specs: Iterable[tuple[Optional[Buffer], ArraySpec]],
    ) -> Iterable[Optional[NDBuffer]]:
        chunks_and_specs = list(chunks_and_specs)
        print(f"[decode] {len(chunks_and_specs)} chunk(s)")

        for i, (_, spec) in enumerate(chunks_and_specs):
            print(f"  chunk[{i}] shape={spec.shape}, dtype={spec.dtype}")

        return await super().decode(chunks_and_specs)

    # ==========================================================================
    # REQUIRED METADATA METHODS
    # ==========================================================================
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

    # ==========================================================================
    # OPTIONAL METADATA METHODS
    # ==========================================================================
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

    # ==========================================================================
    # ===== HELPER METHODS =====================================================
    # ==========================================================================
    @staticmethod
    def detect_format(file_path):
        with open(file_path, 'rb') as f:
            f.seek(8)
            signature = f.read(4)

        if signature == b'XDF0':
            return "XDF"
        elif signature == b'RSF0':
            return "RSF"
        else:
            return "Unknown"
