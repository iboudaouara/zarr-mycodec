import os

import numpy as np
import rmn
from rmn.fst24file import fst24_file

target_file = "eccc-data/sample_safe.fst"
if os.path.exists(target_file):
    os.remove(target_file)

rng = np.random.default_rng(42)
data_to_write = rng.standard_normal((8, 8, 1)).astype("float32")

data_to_write = np.asfortranarray(data_to_write)

rec = rmn.fst_record(
    nomvar = "TT",
    etiket = "TESTING",
    typvar = "E",
    datev = 202601070,
    dateo = 0,
    deet = 0,
    npas = 0,
    ip1 = 0, ip2 = 0, ip3 = 0,
    ig1 = 0, ig2 = 0, ig3 = 0, ig4 = 0,
    data_type = 5,
    data_bits = 32,
    pack_bits = 32,
    ni = 8, nj = 8, nk = 1,
    data = data_to_write
)


with fst24_file(target_file, "R/W") as fout:
    fout.write(rec, rewrite=False)

print(f"Successfully created safe sample: {target_file}")
