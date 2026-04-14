import numpy as np
import rmn
from rmn.fst24file import fst24_file
import os

rec = rmn.fst_record()

rec.nomvar = "TEST"
rec.etiket = "TESTING"
rec.typvar = "P"
rec.datev = 202601070
rec.dateo = 0
rec.deet = 0
rec.npas = 0
rec.ip1, rec.ip2, rec.ip3 = 0, 0, 0
rec.ig1, rec.ig2, rec.ig3, rec.ig4 = 0, 0, 0, 0

rec.data_type = 1
rec.data_bits = 32
rec.ni, rec.nj, rec.nk = 8, 8, 1

rec.data = np.zeros((8, 8), dtype="float32", order="F")

target_file = "eccc-data/sample_safe.fst"
if os.path.exists(target_file):
    os.remove(target_file)

with fst24_file(target_file, "R/W") as fout:
    fout.write(rec, rewrite=False)

print(f"Successfully created safe sample: {target_file}")
