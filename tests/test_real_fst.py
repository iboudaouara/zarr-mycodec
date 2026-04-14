import numpy as np
import pytest
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
        
        print(f"{official_data}")
        
        offset, length = record.file_offset, record.total_stored_bytes
        print(f"DEBUG: record.ni, nj, nk: {record.ni}, {record.nj}, {record.nk}")
        print(f"DEBUG: record.datyp: {record.data}")
        print(f"DEBUG: File Offset: {offset}, Length: {length}")
    
    print(f"Data pointer address: {record.data}")
    with open(fst_file, "rb") as raw_f:
        raw_f.seek(offset)
        chunk_bytes = raw_f.read(length)

    print(f"DEBUG: First 16 bytes of chunk: {chunk_bytes[:16].hex(' ')}")

    if record.data is None or (
        isinstance(record.data, np.ndarray) and record.data.size == 0
    ):
        raise RuntimeError("Record data is empty or NULL")

    with open(fst_file, "rb") as raw_f:
        raw_f.seek(offset)
        chunk_bytes = raw_f.read(length)
        hexdump_c(chunk_bytes, length=128)

    input_buffer = default_buffer_prototype().buffer.from_bytes(chunk_bytes)
    chunk_spec = ArraySpec(
        shape=official_data.shape,
        dtype=official_data.dtype,
        fill_value=0,
        config={},
        prototype=default_buffer_prototype,
    )
    import struct

    header_ints = struct.unpack("II", chunk_bytes[:8])
    print(f"DEBUG: First two 32-bit words of chunk: {header_ints}")

    decoded_array = await FSTCodec()._decode_single(input_buffer, chunk_spec)

    np.testing.assert_array_equal(decoded_array, official_data)


def hexdump_c(data, length=64):
    """Simple Python implementation of hexdump -C"""
    print("-" * 70)
    for i in range(0, min(len(data), length), 16):
        chunk = data[i : i + 16]
        hex_vals = " ".join(f"{b:02x}" for b in chunk)
        # Handle the gap in the middle of hexdump -C
        if len(chunk) > 8:
            hex_vals = f"{hex_vals[:23]}  {hex_vals[24:]}"

        ascii_vals = "".join(chr(b) if 32 <= b <= 126 else "." for b in chunk)
        print(f"{i:08x}  {hex_vals:<48}  |{ascii_vals}|")
    print("-" * 70)
