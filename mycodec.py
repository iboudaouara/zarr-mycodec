import numpy as np
import numcodecs
from numcodecs.abc import Codec
from numcodecs.compat import ensure_contiguous

class XOROffsetCodec(Codec):
    """
    A custom Zarr 3 compatible codec that applies a bitwise XOR 
    and an integer offset to the data.
    """
    codec_id = 'xor_offset'

    def __init__(self, key=0, offset=0):
        self.key = int(key)
        self.offset = int(offset)

    def encode(self, buf):
        # Convert input buffer to a numpy array for vector operations
        data = np.frombuffer(ensure_contiguous(buf), dtype='uint8').copy()
        # Apply transformation: (x + offset) ^ key
        encoded = (data + self.offset) ^ self.key
        return encoded.tobytes()

    def decode(self, buf, out=None):
        # Reverse transformation: (x ^ key) - offset
        data = np.frombuffer(ensure_contiguous(buf), dtype='uint8').copy()
        decoded = (data ^ self.key) - self.offset
        return ensure_contiguous(decoded.tobytes(), out)

    def get_config(self):
        # Required for metadata serialization in Zarr 3 (zarr.json)
        return {'id': self.codec_id, 'key': self.key, 'offset': self.offset}

    @classmethod
    def from_config(cls, config):
        return cls(key=config.get('key', 0), offset=config.get('offset', 0))

# IMPORTANT: Register the codec globally
numcodecs.registry.register_codec(XOROffsetCodec)
