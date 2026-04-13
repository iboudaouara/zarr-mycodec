import numpy as np
import pytest
import rmn
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype

from share.my_codec import FSTCodec


@pytest.mark.asyncio
async def test_decode_single_real_data():
    file_path = "eccc-data/2026010700_000"

    with rmn.fst24_file(file_path, "R") as f:
        q = f.new_query(nomvar="TT")
        
        record = next(iter(q))
        print(record)
        
        if record.data_type == 1:
            data_type = np.float32 if record.data_bits <= 32 else np.float64
        elif record.data_type == 2:
            data_type = np.int32
        else:
            data_type = np.float32
        
        shape = (record.ni, record.nj, record.nk)
        input_buffer = record.data

    chunk_spec = ArraySpec(
        shape=shape,
        dtype=data_type,
        fill_value=np.nan,
        config={},
        prototype=default_buffer_prototype(),
    )

    # 4. Exécution du codec
    codec = FSTCodec(format="xdf")
    result = await codec._decode_single(input_buffer, chunk_spec)

    # 5. Assertions
    assert result.shape == shape
    #np.testing.assert_array_almost_equal(result, expected_data)
    
