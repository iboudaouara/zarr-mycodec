import zarr
import numpy as np
# If this is in a separate file, make sure to import your codec to register it!
import share.my_codec 

def test_custom_codec():
    print("1. Creating dummy data...")
    data = np.arange(100, dtype='i4')
    
    # We instantiate your codec
    my_compressor = share.my_codec.FSTCodec(level=1)
    
    print(f"2. Asking Zarr to use codec: {my_compressor.codec_id}")
    
    try:
        # This will attempt to ENCODE the data using your codec
        zarr.array(data, codecs=[my_compressor])
    except NotImplementedError as e:
        print(f"3. SUCCESS! Zarr tried to use the codec and hit your error: {e}")

if __name__ == "__main__":
    test_custom_codec()
