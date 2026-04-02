from dataclasses import dataclass
from typing import Iterable, Optional, Self

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.chunk_grids import ChunkGrid
from zarr.core.dtype.wrapper import TBaseDType, TBaseScalar, ZDType

import logging

logging.basicConfig(level=logging.DEBUG)
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FSTCodec(ArrayBytesCodec):

    # ================================================================================
    # SINGLE CHUNK PROCESSING
    # ================================================================================
    async def _encode_single(
        self, input_buffer: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer:
        _ = input_buffer
        _log.debug("shape=%s, dtype=%s", chunk_spec.shape, chunk_spec.dtype)
        # print(f"[_encode_single] shape={chunk_spec.shape}, dtype={chunk_spec.dtype}")
        raise NotImplementedError("Implement the single chunk encoding logic.")

    async def _decode_single(
        self, input_buffer: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        _ = input_buffer
        print(f"[_decode_single] shape={chunk_spec.shape}, dtype={chunk_spec.dtype}")
        raise NotImplementedError("Implement the single chunk decoding logic.")

    # ================================================================================
    # BATCH PROCESSING
    # ================================================================================
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
