import asyncio
import numpy as np
import zarr
from zarr.core.array_spec import ArraySpec, ArrayConfig
from zarr.core.buffer import NDBuffer, Buffer, default_buffer_prototype
from share.my_codec import FSTCodec

async def test_codec_logic():
    print("--- Running Unit Test: Codec Symmetry ---")
    
    codec = FSTCodec(level=1)
    data = np.arange(100, dtype='float32').reshape(10, 10)
    
    spec = ArraySpec(
        shape=(10, 10), 
        dtype=np.dtype('float32'), 
        fill_value=0.0,
        config=ArrayConfig(order='C', write_empty_chunks=False),
        prototype=default_buffer_prototype()
    )
    
    input_buffer = spec.prototype.nd_buffer.from_numpy_array(data)

    # Test Encoding
    encoded_buffer = await codec._encode_single(input_buffer, spec)
    # FIX: Changed as_bytes() to to_bytes()
    print(f"   - Encoded {data.nbytes} bytes into {len(encoded_buffer.to_bytes())} bytes.")

    # Test Decoding
    decoded_buffer = await codec._decode_single(encoded_buffer, spec)
    decoded_data = decoded_buffer.as_numpy_array()

    np.testing.assert_array_equal(data, decoded_data)
    print("✅ Success: Decoded data matches original data!\n")

def test_zarr_integration():
    print("--- Running Integration Test: Zarr Array ---")
    store = zarr.storage.MemoryStore()
    my_codec = FSTCodec(level=5)
    
    z_array = zarr.create(
        shape=(20, 20),
        chunks=(10, 10),
        dtype='float32',
        store=store,
        codecs=[my_codec]
    )

    input_data = np.random.rand(20, 20).astype('float32')
    z_array[:] = input_data
    output_data = z_array[:]

    np.testing.assert_allclose(input_data, output_data)
    print("✅ Success: Zarr integration works perfectly!")

if __name__ == "__main__":
    try:
        asyncio.run(test_codec_logic())
        test_zarr_integration()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Test Failed with error: {e}")
