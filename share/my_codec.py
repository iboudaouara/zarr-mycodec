from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Any, Self

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import NDBuffer, Buffer
from zarr.core.array_spec import ArraySpec


@dataclass(frozen=True)
class FSTCodec(ArrayBytesCodec):
    """
    Zarr V3 Template for a Custom Codec (Array-to-Bytes)

    This codec serves as a starting point for integrating custom formats (e.g., RPN-FST). Registration with Zarr is handled automatically via 'entry_points' in the pyproject.toml / pixi.toml file.
    """

    @classmethod
    def from_dict(cls, data):
        config = data.get("configuration", {})
        return cls(**config)

    # Codec attributes (e.g., compression level, specific options)
    level: int = 1

    # ================================================================================
    # 1. SINGLE CHUNK PROCESSING (Recommended by default)
    # Most custom codecs should implement these. They operate on single chunks.
    # ================================================================================

    async def _encode_single(
        self, input_buffer: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer:
        """
        Encodes a single chunk of data.
        Takes an N-dimensional buffer and converts it into a flat sequence of bytes.

            Write (Array -> Bytes)

            Takes an N-dimensional buffer (the raw data) and converts it into a flat sequence of bytes saved to the disk.

            Args:
                input_buffer: The multidimensional data  to compress.
                chunk_spec: Metadata about the chunk (shape, dtype, etc.).
        """
        # data = input_buffer.as_numpy_array()
        # TODO: Encoding logic here
        # return chunk_spec.prototype.buffer.from_bytes(data.tobytes())
        raise NotImplementedError("Implement the single chunk encoding logic.")

    async def _decode_single(
        self, input_buffer: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        """Decodes a single chunk of data."""
        """
        Read (Bytes -> Array)
        Takes a flat sequence of bytes read from disk and reconstructs the N-dimensional array.
        Args:
            input_buffer: The raw compressed bytes read from storage.
            chunk_spec: Metadata telling us the shape and dtype we need to restore.
        """
        raw_bytes = input_buffer.to_bytes()
        if hasattr(chunk_spec.dtype, "to_native_dtype"):
            numpy_dtype = chunk_spec.dtype.to_native_dtype()
        elif hasattr(chunk_spec.dtype, "to_numpy_dtype"):
            numpy_dtype = chunk_spec.dtype.to_numpy_dtype()
        else:
            numpy_dtype = chunk_spec.dtype
        restored_data = np.frombuffer(raw_bytes, dtype=numpy_dtype).reshape(
            chunk_spec.shape
        )
        return chunk_spec.prototype.nd_buffer.from_numpy_array(restored_data)

    #    raise NotImplementedError("Implement the single chunk decoding logic.")

    # ================================================================================
    # 2. BATCH PROCESSING (Alternative)
    # Implement these if the codec is intended to implement its own batch
    # processing (e.g., for C/Fortran vectorization).
    # If uncommented, these will override the _encode_single calls.
    # ================================================================================

    # async def encode(
    #     self,
    #     chunks_and_specs: Iterable[Tuple[Optional[NDBuffer], ArraySpec]],
    # ) -> Iterable[Optional[Buffer]]:
    #     """Operates on batches of chunks for encoding."""
    #     pass

    # async def decode(
    #     self,
    #     chunks_and_specs: Iterable[Tuple[Optional[Buffer], ArraySpec]],
    # ) -> Iterable[Optional[NDBuffer]]:
    #     """Operates on batches of chunks for decoding."""
    #     pass

    # ================================================================================
    # REQUIRED  BASE CLASS METHODS
    # ================================================================================

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        """
        Calculates the maximum expected size (in bytes) of the chunk after encoding.
        If the size is unpredictable (like with most compression algorithms),
        raise a NotImplementedError to force dynamic allocation.
        """
        raise NotImplementedError("FST encoded size is data-dependent.")

    # ================================================================================
    # OPTIONAL  BASE CLASS METHODS
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
        pass

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """
        (Optional) Important for codecs that change the shape, dtype or
        fill value of a chunk.
        """
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """
        (Optional) Useful for automatically filling the codec configuration metadata
        from the array metadata.
        """
        return self
