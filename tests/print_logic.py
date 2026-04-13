import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock

from zarr.core.array_spec import ArraySpec, ArrayConfig
from zarr.core.dtype.wrapper import ZDType
from zarr.core.buffer import NDBuffer
from zarr.core.chunk_grids import RegularChunkGrid
from zarr.core.buffer.cpu import buffer_prototype

from share.my_codec import FSTCodec


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_chunk_spec(shape=(4, 4), dtype=np.float32):
    spec = MagicMock()
    spec.shape = shape
    spec.dtype.to_native_dtype.return_value = np.dtype(dtype)
    spec.dtype.to_numpy_dtype.return_value = np.dtype(dtype)
    spec.prototype.nd_buffer.from_numpy_array.side_effect = lambda arr: arr
    spec.prototype.buffer.from_bytes.side_effect = lambda b: MagicMock(
        to_bytes=lambda: b
    )
    return spec

def make_real_chunk_spec(shape=(4, 4), dtype="float32"):
    # On crée un vrai objet ZDType
    z_dtype = ZDType.from_native_dtype(np.dtype(dtype))
    
    # On crée une configuration minimale
    config = ArrayConfig(
        shape=shape,
        dtype=z_dtype,
        chunk_grid=RegularChunkGrid(chunk_shape=shape),
        fill_value=0.0
    )
    
    # On instancie le vrai ArraySpec
    return ArraySpec(
        shape=shape,
        dtype=z_dtype,
        fill_value=0.0,
        config=config,
        prototype=buffer_prototype()
    )
    
def make_nd_buffer(shape=(4, 4), dtype=np.float32):
    arr = np.random.rand(*shape).astype(dtype)
    buf = MagicMock()
    buf.as_numpy_array.return_value = arr
    return buf, arr


def make_raw_buffer(arr: np.ndarray):
    raw = arr.tobytes()
    buf = MagicMock()
    buf.to_bytes.return_value = raw
    return buf


def run(coro):
    return asyncio.run(coro)


# ── Codec fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def codec():
    return FSTCodec()


# ── Tests — prints appear directly in terminal with: pytest -v -s ─────────────


class TestEncodeSingle:
    def test_called(self, codec):
        nd_buf, _ = make_nd_buffer()
        spec = make_real_chunk_spec()
        with pytest.raises(NotImplementedError):
            run(codec._encode_single(nd_buf, spec))


class TestDecodeSingle:
    def test_called(self, codec):
        arr = np.random.rand(4, 4).astype(np.float32)
        spec = make_chunk_spec()
        with pytest.raises(NotImplementedError):
            run(codec._decode_single(make_raw_buffer(arr), spec))


class TestEncodeBatch:
    def test_called(self, codec):
        nd_buf, _ = make_nd_buffer()
        spec = make_chunk_spec()
        with pytest.raises(NotImplementedError):
            run(codec.encode([(nd_buf, spec), (nd_buf, spec)]))


class TestDecodeBatch:
    def test_called(self, codec):
        arr = np.random.rand(4, 4).astype(np.float32)
        spec = make_chunk_spec()
        with pytest.raises(NotImplementedError):
            run(
                codec.decode(
                    [(make_raw_buffer(arr), spec), (make_raw_buffer(arr), spec)]
                )
            )


class TestComputeEncodedSize:
    def test_called(self, codec):
        spec = make_chunk_spec()
        with pytest.raises(NotImplementedError):
            codec.compute_encoded_size(128, spec)


class TestValidate:
    def test_called(self, codec):
        codec.validate(shape=(4, 4), dtype=np.dtype("float32"), chunk_grid=MagicMock())


class TestResolveMetadata:
    def test_called(self, codec):
        spec = make_chunk_spec()
        result = codec.resolve_metadata(spec)
        assert result is spec


class TestEvolveFromArraySpec:
    def test_called(self, codec):
        spec = make_chunk_spec()
        result = codec.evolve_from_array_spec(spec)
        assert result is codec
