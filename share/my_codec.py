from __future__ import annotations

import ctypes
import io
import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Iterable, Optional, Self

import numpy as np
import rmn
from rmn._sharedlib import librmn
from rmn.fst24file import fst24_file
from rmn.fstrecord import (
    FstDataType,
    fst_record,
    fst_type_to_numpy_type,
    numpy_type_to_fst_type,
)
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

_fst24_read_record = librmn.fst24_read_record
_fst24_read_record.argtypes = (ctypes.POINTER(fst_record),)
_fst24_read_record.restype = ctypes.c_int

_get_default_fst_record = librmn.get_default_fst_record
_get_default_fst_record.argtypes = ()
_get_default_fst_record.restype = fst_record


@dataclass(frozen=True)
class FSTCodec(ArrayBytesCodec):

    data_type: FstDataType = FstDataType.FST_TYPE_REAL
    pack_bits: int = 16
    nomvar: str = "    "
    typvar: str = "P"
    grtyp: str = "X"
    etiket: str = ""
    ip1: int = 0
    ip2: int = 0
    ip3: int = 0
    ig1: int = 0
    ig2: int = 0
    ig3: int = 0
    ig4: int = 0
    deet: int = 0
    npas: int = 0
    dateo: int = 0

    # ================================================================================
    # SINGLE CHUNK PROCESSING
    # ================================================================================
    async def _encode_single(
        self, input_buffer: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer:
        array = input_buffer.as_numpy_array()
        ni, nj, nk = _shape_to_nijk(chunk_spec.shape)
        fst_dtype, data_bits = _fst_dtype_and_bits(self.data_type, array.dtype)

        _log.debug(
            "encode shape=%s ni=%d nj=%d nk=%d dtype=%s pack_bits=%d",
            chunk_spec.shape,
            ni,
            nj,
            nk,
            chunk_spec.dtype,
            self.pack_bits,
        )

        rec = fst_record(
            ni=ni,
            nj=nj,
            nk=nk,
            data_type=fst_dtype,
            data_bits=data_bits,
            pack_bits=self.pack_bits,
            nomvar=self.nomvar,
            typvar=self.typvar,
            grtyp=self.grtyp,
            etiket=self.etiket,
            ip1=self.ip1,
            ip2=self.ip2,
            ip3=self.ip3,
            ig1=self.ig1,
            ig2=self.ig2,
            ig3=self.ig3,
            ig4=self.ig4,
            deet=self.deet,
            npas=self.npas,
            dateo=self.dateo,
        )
        rec.data = np.asfortranarray(array.reshape(ni, nj, nk))

        payload = _write_record_to_bytes(rec)
        _log.debug(
            "encoded %d B -> %d B (ratio %.2f)",
            array.nbytes,
            len(payload),
            array.nbytes / len(payload),
        )

        return Buffer.create_zero_length().__class__.from_bytes(payload)

    async def _decode_single(self, input_buffer: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        raw_bytes = input_buffer.to_bytes()

        _log.debug("[_decode_single] shape=%s, dtype=%s, input_bytes=%d", chunk_spec.shape, chunk_spec.dtype, len(raw_bytes))

        c_byte_array = ctypes.c_char_p(raw_bytes)
        ptr_input = ctypes.cast(c_byte_array, ctypes.c_void_p)

        rec = _fst24_decode_data_rsf(ptr_input, None)

        if chunk_spec.dtype == np.float32:
            c_type_ptr = ctypes.POINTER(ctypes.c_float)
        elif chunk_spec.dtype == np.float64:
            c_type_ptr = ctypes.POINTER(ctypes.c_double)
        elif chunk_spec.dtype == np.int32:
            c_type_ptr = ctypes.POINTER(ctypes.c_int32)
        else:
            raise ValueError(f"D")

        data_ptr = ctypes.cast(rec.data, c_type_ptr)

        total_elements = np.prod(chunk_spec.shape)

        flat_array = np.ctypeslib.as_array(data_ptr, shape=(total_elements,))

        array_3d = flat_array.reshape(chunk_spec.shape, order='F')

        array_3d = array_3d.astype(chunk_spec.dtype, copy=False)

        raise NDBuffer.create(array_3d)

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
import tempfile
import os
import uuid


import tempfile
import os


def _write_record_to_bytes(rec: fst_record) -> bytes:
    # 1. On crée un vrai fichier temporaire pour obtenir un descripteur (fd) valide
    fd, actual_path = tempfile.mkstemp(prefix="fst_chunk_write_")
    try:
        # 2. LE TRUC MAGIQUE : On passe par /proc/self/fd/ pour forcer librmn
        # à agir comme un flux, contournant ainsi le crash de c_fnom
        path_fd = f"/proc/self/fd/{fd}"

        with fst24_file(path_fd, "w") as f:
            f.write(rec)

        # 3. On revient au début du fichier et on lit les octets
        os.lseek(fd, 0, os.SEEK_SET)
        return os.read(fd, os.fstat(fd).st_size)
    finally:
        # 4. On ferme proprement et on supprime le fichier physique
        os.close(fd)
        if os.path.exists(actual_path):
            os.remove(actual_path)


def _read_record_from_bytes(payload: bytes) -> fst_record:
    # On applique la même astuce pour la lecture
    fd, actual_path = tempfile.mkstemp(prefix="fst_chunk_read_")
    try:
        # On écrit les octets dans le descripteur
        os.write(fd, payload)
        os.lseek(fd, 0, os.SEEK_SET)

        # On trompe librmn avec /proc/self/fd/
        path_fd = f"/proc/self/fd/{fd}"
        with fst24_file(path_fd, "r") as f:
            records = list(f)

        if not records:
            raise RuntimeError("Aucun enregistrement trouvé dans le payload FST")
        return records[0]
    finally:
        os.close(fd)
        if os.path.exists(actual_path):
            os.remove(actual_path)


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
    possible_types, data_bits = numpy_type_to_fst_type(dtype)
    if codec_type in possible_types:
        return codec_type, data_bits
    _log.debug(
        "data_type=%s incompatible with dtype=%s, falling back to %s",
        codec_type.name,
        dtype,
        possible_types[0].name,
    )
    return possible_types[0], data_bits
