import asyncio

from share.my_codec import FSTCodec
from unittest.mock import MagicMock

faux_spec = MagicMock()
faux_spec.shape = (4, 4)
faux_spec.dtype = "float32"

codec = FSTCodec()
asyncio.run(codec._encode_single(None, faux_spec))
