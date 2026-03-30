from dataclasses import dataclass
import numpy as np

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import NDBuffer, Buffer
from zarr.core.array_spec import ArraySpec
from zarr.registry import register_codec

@dataclass(frozen=True)
class FSTCodec(ArrayBytesCodec):
    """
    A custom Zarr v3 codec for FSTD compression.
    """
    level: int = 1

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        """
        Calculates the maximum possible size of the encoded bytes.
        """
        raise NotImplementedError("FST compression ratio is data-dependent.")

    async def _encode_single(self, input_buffer: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        """
        Write (Array -> Bytes)
        
        Takes an N-dimensional buffer (the raw data) and converts it into a flat sequence of bytes saved to the disk.
        
        Args:
            input_buffer: The multidimensional data  to compress.
            chunk_spec: Metadata about the chunk (shape, dtype, etc.).
        """
        data = input_buffer.as_numpy_array()
        return chunk_spec.prototype.buffer.from_bytes(data.tobytes())

    async def _decode_single(self, input_buffer: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        """
        Read (Bytes -> Array)
        
        Takes a flat sequence of bytes read from disk and reconstructs the N-dimensional array.
        
        Args:
            input_buffer: The raw compressed bytes read from storage.
            chunk_spec: Metadata telling us the shape and dtype we need to restore.
        """
        raw_bytes = input_buffer.to_bytes()
        
        if hasattr(chunk_spec.dtype, 'to_native_dtype'):
            numpy_dtype = chunk_spec.dtype.to_native_dtype()
        elif hasattr(chunk_spec.dtype, 'to_numpy_dtype'):
            numpy_dtype = chunk_spec.dtype.to_numpy_dtype()
        else:
            numpy_dtype = chunk_spec.dtype
        
        restored_data = np.frombuffer(raw_bytes, dtype=numpy_dtype).reshape(chunk_spec.shape)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(restored_data)

register_codec("eccc-rpn-fst", FSTCodec)
