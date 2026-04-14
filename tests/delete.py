import pytest
import numpy as np
import struct
from rmn.fst24file import fst24_file
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype
from share.my_codec import FSTCodec

@pytest.mark.asyncio
async def test_decode_with_real_fst_file():
    fst_file = "eccc-data/2026010700_000"

    with fst24_file(fst_file, "R") as f:
        record = next(iter(f))
        official_data = np.array(record.data, copy=True)
        offset, length = record.file_offset, record.total_stored_bytes

    with open(fst_file, "rb") as raw_f:
        raw_f.seek(offset)
        chunk_bytes = raw_f.read(length)

    # Verify Header
    h1, h2 = struct.unpack(">II", chunk_bytes[:8])
    print(f"Header Words: {hex(h1)}, {hex(h2)} | Format: {'XDF' if chunk_bytes[0] == 1 else 'RSF'}")

    input_buffer = default_buffer_prototype().buffer.from_bytes(chunk_bytes)
    chunk_spec = ArraySpec(
        shape=official_data.shape,
        dtype=official_data.dtype,
        fill_value=0,
        config={},
        prototype=default_buffer_prototype
    )

    decoded_array = await FSTCodec()._decode_single(input_buffer, chunk_spec)
    np.testing.assert_array_equal(decoded_array, official_data)
