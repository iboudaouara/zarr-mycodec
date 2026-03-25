from dataclasses import dataclass
import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import NDBuffer, Buffer
from zarr.core.array_spec import ArraySpec
from zarr.registry import register_codec

@dataclass(frozen=True)
class FSTCodec(ArrayBytesCodec):
    level: int = 1

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError("FST compression ratio is data-dependent.")

    async def _encode_single(self, input_buffer: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        data = input_buffer.as_numpy_array()
        return chunk_spec.prototype.buffer.from_bytes(data.tobytes())

    async def _decode_single(self, input_buffer: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raw_bytes = input_buffer.to_bytes()
        
        # UNIVERSAL ADAPTER FOR ZARR V3
        # 1. Try the newest 'to_native_dtype' (suggested by your error)
        # 2. Try the older 'to_numpy_dtype'
        # 3. Fallback to the object itself (for standard NumPy dtypes)
        if hasattr(chunk_spec.dtype, 'to_native_dtype'):
            numpy_dtype = chunk_spec.dtype.to_native_dtype()
        elif hasattr(chunk_spec.dtype, 'to_numpy_dtype'):
            numpy_dtype = chunk_spec.dtype.to_numpy_dtype()
        else:
            numpy_dtype = chunk_spec.dtype
        
        restored_data = np.frombuffer(raw_bytes, dtype=numpy_dtype).reshape(chunk_spec.shape)
        return chunk_spec.prototype.nd_buffer.from_numpy_array(restored_data)

register_codec("rpn-fst-eccc", FSTCodec)
