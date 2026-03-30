from dataclasses import dataclass

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.array_spec import ArraySpec


@dataclass(frozen=True)
class FSTCodec(ArrayBytesCodec):

    level: int = 1

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        """
        Calculates the maximum possible size of the encoded bytes.
        """
        raise NotImplementedError("FST compression ratio is data-dependent.")

    def validate(
        self,
        *,
        shape: tuple[int, ...],
        dtype: ZDType[TBaseDType, TBaseScalar],
        chunk_grid: ChunkGrid,
    ) -> None:
        return self

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        return chunk_spec

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return self
