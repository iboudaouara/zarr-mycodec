import pytest
import numpy as np
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import (Buffer, default_buffer_prototype)
from rmn.fst24file import fst24_file

from share.my_codec import FSTCodec, _write_record_to_bytes


@pytest.mark.asyncio
async def test_decode_with_real_fst_file():
    fst_file = "eccc-data/2026010700_000"

    with fst24_file(fst_file, "r") as f:
        records = list(f)
        assert len(records) > 0, "The test FST file is empy"
        record = records[0]
        official_data = np.array(record.data, copy=True)

    expected_shape = (record.ni, record.nj, record.nk)
    expected_dtype = official_data.dtype
    
    
    chunk_spec = ArraySpec(shape=expected_shape, dtype=expected_dtype, fill_value=np.nan, config={}, prototype=default_buffer_prototype)

    codec = FSTCodec(
        nomvar=record.nomvar,
        typvar=record.typvar,
        grtyp=record.grtyp,
        pack_bits=record.pack_bits,
        data_type=record.data_type,
        ip1=record.ip1,
        ip2=record.ip2,
        ip3=record.ip3,
        ig1=record.ig1,
        ig2=record.ig2,
        ig3=record.ig3,
        ig4=record.ig4,
        deet=record.deet,
        npas=record.npas,
        dateo=record.dateo,
    )

    raw_bytes = _write_record_to_bytes(record)
    input_buffer = Buffer.from_bytes(raw_bytes)

    decoded_ndbuffer = await codec._decode_single(input_buffer, chunk_spec)
    donnees_decodees = decoded_ndbuffer.as_numpy_array()

    # ── ÉTAPE 4 : assertions ─────────────────────────────────────────────────
    assert donnees_decodees.shape == official_data.shape, "Shape mismatch"
    assert donnees_decodees.dtype == official_data.dtype, "Dtype mismatch"
    np.testing.assert_array_almost_equal(
        official_data,
        donnees_decodees,
        err_msg="Decoded data differs from the official library output!",
    )
