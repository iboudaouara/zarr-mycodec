import numpy as np

data = np.random.rand(10, 10).astype("float32")

encoded_bytes = data.tobytes()

with open("data/source_data.fst", "wb") as f:
    f.write(b"HEADER_FST_")
    start_offset = f.tell()
    f.write(encoded_bytes)
    end_offset = f.tell()
    size = end_offset - start_offset

import json

manifest = {
    "version": 1,
    "refs": {
        "zarr.json": json.dumps({"zarr_format": 3, "node_type": "group"}),
        "temperature/zarr.json": json.dumps(
            {
                "zarr_format": 3,
                "node_type": "array",
                "shape": [10, 10],
                "data_type": "float32",
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [10, 10]},
                },
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "/"},
                },
                "codecs": [{"name": "eccc-rpn-fst", "configuration": {"level": 1}}],
                "fill_value": 0.0,
            }
        ),
        "temperature/c/0/0": ["data/source_data.fst", start_offset, size],
    },
}

with open("manifest.json", "w") as f:
    json.dump(manifest, f)

import zarr
import fsspec

fs = fsspec.filesystem("reference", fo="manifest.json", remote_protocol="file")
store = fs.get_mapper("")

z = zarr.open(store=store, mode="r")
output = z["temperature"][:]


print(output.shape)
print(np.allclose(output, data))

print("original:", data[0, 0:3])
print("decoded :", output[0, 0:3])
