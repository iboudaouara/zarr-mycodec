# zarr-mycodec/test_run.py
import logging

logging.basicConfig(level=logging.DEBUG)

from share.my_codec import FSTCodec
from rmn.fstrecord import FstDataType

codec = FSTCodec(data_type=FstDataType.FST_TYPE_REAL_IEEE, pack_bits=32)
print(codec)
