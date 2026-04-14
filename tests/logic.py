import numpy as np
import pytest
from rmn.fst24file import fst24_file
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, default_buffer_prototype

from share.my_codec import FSTCodec


@pytest.mark.asyncio
async def test_fst_decode_single():
    fst_file = "eccc-data/2026010700_000"
    codec = FSTCodec()

    with fst24_file(fst_file, "R") as f:
        expected_rec = next(r for r in f if r.nomvar.strip() != ">>")
        official_data = np.array(expected_rec.data, copy=True)
        offset = expected_rec.file_offset

    with open(fst_file, "rb") as raw_f:
        raw_f.seek(offset)
        header_word = np.frombuffer(raw_f.read(4), dtype=">u4")[0]
        rec_lng = header_word & 0x00FFFFFF

        raw_f.seek(offset)
        raw_bytes = raw_f.read(rec_lng * 8)

    spec = ArraySpec(
        shape=official_data.shape,
        dtype=str(official_data.dtype),
        fill_value=0,
        config={},
        prototype=default_buffer_prototype(),
    )

    input_buffer = default_buffer_prototype().buffer.from_bytes(raw_bytes)
    result_buffer = await codec._decode_single(input_buffer, spec)

    assert np.allclose(result_buffer.as_numpy_array(), official_data)
