import pytest
import numpy as np
from rmn.fst24file import fst24_file
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import default_buffer_prototype

from share.my_codec import FSTCodec


@pytest.mark.asyncio
async def test_decode_with_real_fst_file():
    fst_file = "eccc-data/2026010700_000"
    print(f"1. {fst_file}")

    with fst24_file(fst_file, "R") as f:
        records = list(f)
        assert len(records) > 0, "The test FST file is empty"
        record = records[0]
        official_data = np.array(record.data, copy=True)
        print(f"2. {official_data}")

    #expected_shape = (record.ni, record.nj, record.nk)
    #expected_dtype = official_data.dtype
#
    #chunk_spec = ArraySpec(
    #    shape=expected_shape,
    #    dtype=expected_dtype,
    #    fill_value=np.nan,
    #    config={},
    #    prototype=default_buffer_prototype(),
    #)
#
    #codec = FSTCodec(
    #    nomvar=record.nomvar,
    #    typvar=record.typvar,
    #    grtyp=record.grtyp,
    #    pack_bits=record.pack_bits,
    #    data_type=record.data_type,
    #    ip1=record.ip1,
    #    ip2=record.ip2,
    #    ip3=record.ip3,
    #    ig1=record.ig1,
    #    ig2=record.ig2,
    #    ig3=record.ig3,
    #    ig4=record.ig4,
    #    deet=record.deet,
    #    npas=record.npas,
    #    dateo=record.dateo,
    #)
#
    ## --- THE FIX IS HERE ---
    ## Instead of writing a new file, we read the exact bytes of this record
    ## directly from the original FST file using its byte offset and size.
    #with open(fst_file, "rb") as f_raw:
    #    f_raw.seek(record.file_offset)
    #    raw_bytes = f_raw.read(record.total_stored_bytes)
#
    #prototype = default_buffer_prototype()
    #encoded_buffer = prototype.buffer.from_bytes(raw_bytes)
#
    ## 3. Decode the raw disk buffer back to an ndbuffer
    #decoded_ndbuffer = await codec._decode_single(encoded_buffer, chunk_spec)
    #donnees_decodees = decoded_ndbuffer.as_numpy_array()
#
    ## 4. Assertions against the official rmn output
    #assert donnees_decodees.shape == official_data.shape, "Shape mismatch"
    #assert donnees_decodees.dtype == official_data.dtype, "Dtype mismatch"
    #np.testing.assert_array_almost_equal(
    #    official_data,
    #    donnees_decodees,
    #    err_msg="Decoded data differs from the official library output!",
    #)
#
