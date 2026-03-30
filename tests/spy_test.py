import json
import fsspec
import zarr
import numpy as np
from dataclasses import dataclass

from zarr.abc.codec import ArrayBytesCodec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.array_spec import ArraySpec
from zarr.registry import register_codec

# =========================================================================
# 1. THE DEMONSTRATOR CODEC
# This codec doesn't decompress anything. Its sole purpose is to use 
# 'assert' statements to prove that what it receives exactly matches the manifest.
# =========================================================================
@dataclass(frozen=True)
class RelationDemonstratorCodec(ArrayBytesCodec):
    # We pass the expected values from the manifest via the codec configuration
    expected_shape: list
    expected_dtype: str
    expected_payload_str: str

    @classmethod
    def from_dict(cls, data: dict):
        """Extracts the 'configuration' block from the Zarr JSON."""
        config = data.get("configuration", {})
        return cls(**config)

    def compute_encoded_size(self, input_byte_length: int, chunk_spec: ArraySpec) -> int:
        raise NotImplementedError()

    async def _encode_single(self, input_buffer: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        pass

    async def _decode_single(self, input_buffer: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        # PROOF 1: The physical bytes relationship
        # Asserts that the Kerchunk [offset, size] strictly defines the input_buffer
        actual_bytes = input_buffer.to_bytes()
        expected_bytes = self.expected_payload_str.encode('utf-8')
        
        assert actual_bytes == expected_bytes, \
            f"Buffer mismatch! Expected {expected_bytes}, got {actual_bytes}"
        
        # PROOF 2: The logical metadata relationship
        # Asserts that the manifest's zarr.json strictly defines the chunk_spec
        assert list(chunk_spec.shape) == self.expected_shape, \
            f"Shape mismatch! Expected {self.expected_shape}, got {chunk_spec.shape}"
        
	# Convert Zarr's internal Float32 object back to a standard string
        actual_dtype = chunk_spec.dtype.to_native_dtype().name if hasattr(chunk_spec.dtype, 'to_native_dtype') else str(chunk_spec.dtype)
        
        assert actual_dtype == self.expected_dtype, \
            f"Dtype mismatch! Expected {self.expected_dtype}, got {actual_dtype}"

        print("\n" + "="*65)
        print("✅ PROOF SUCCESSFUL: DECOMPRESSION RELATIONSHIPS VERIFIED")
        print("="*65)
        print(f"1. Manifest [offset, size]   -> input_buffer   (Exactly {len(actual_bytes)} bytes)")
        print(f"2. Manifest 'shape' metadata -> chunk_spec.shape {chunk_spec.shape}")
        print(f"3. Manifest 'dtype' metadata -> chunk_spec.dtype {chunk_spec.dtype}")
        print("="*65 + "\n")
        
        # Return dummy data to fulfill the Zarr pipeline requirement
        fake_data = np.zeros(chunk_spec.shape, dtype=chunk_spec.dtype.to_native_dtype())
        return chunk_spec.prototype.nd_buffer.from_numpy_array(fake_data)

# Register the codec so Zarr can instantiate it from the manifest
register_codec("relation-demonstrator", RelationDemonstratorCodec)


# =========================================================================
# 2. THE TEST EXECUTION
# =========================================================================
def run_proof():
    # A. Write a dummy binary file to simulate an archive (e.g., FST)
    dummy_payload = b"COMPRESSED_DATA_PAYLOAD"
    
    with open("dummy_archive.bin", "wb") as f:
        f.write(b"IGNORE_THIS_HEADER_DATA") # Write some garbage at the beginning
        offset = f.tell()                   # Save the exact byte location
        f.write(dummy_payload)              # Write the payload
        size = len(dummy_payload)           # Save the exact byte size
        
    # B. Create the Kerchunk Manifest
    shape = [15, 25, 35]
    dtype = "float32"
    
    manifest = {
        "version": 1,
        "refs": {
            "zarr.json": json.dumps({"zarr_format": 3, "node_type": "group"}),
            
            # The metadata that becomes the 'chunk_spec'
            "test_array/zarr.json": json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": shape,
                "data_type": dtype,
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": shape}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "codecs": [{
                    "name": "relation-demonstrator", 
                    "configuration": {
                        "expected_shape": shape,
                        "expected_dtype": dtype,
                        "expected_payload_str": dummy_payload.decode('utf-8')
                    }
                }],
                "fill_value": 0.0
            }),
            
            # The pointer that populates the 'input_buffer'
            "test_array/c/0/0/0": ["dummy_archive.bin", offset, size]
        }
    }

    # C. Mount the manifest and trigger decompression
    fs = fsspec.filesystem("reference", fo=manifest, remote_protocol="file")
    store = fs.get_mapper("")
    z = zarr.open(store=store, mode="r")
    
    # This line triggers the read, which calls _decode_single
    _ = z["test_array"][:]

if __name__ == "__main__":
    run_proof()
