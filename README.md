# Custom Codec for Zarr v3

Exploring ways to implement a custom codec using the Zarr v3 codec interface (`zarr.abc.codec`).

## Overview

The goal is to build a custom `ArrayBytesCodec` that can encode and decode data using a specialized format (e.g., FSTD), and later integrate with VirtualiZarr for data access.

## Setup

This project uses Pixi for environment management.

### Install Dependencies

```bash
pixi install
```

### Run Code

```bash
pixi run python tests/test_codec_logic.py
```

## Development Notes: Zarr v3 Codec Contract

This codec implements the `zarr.abc.codec.ArrayBytesCodec` interface.
To function correctly with Zarr v3, the following methods are implemented:

* **Required Methods:**
  * `_encode_single`: Converts an N-dimensional buffer to flat bytes.
  * `_decode_single`: Converts flat bytes back to an N-dimensional buffer.
  * `compute_encoded_size`: Estimates buffer size for memory pre-allocation.
  
* **Optional Methods:**
  * `get_config()` / `from_config()`: For writing metadata to zarr.json.

## Methods to implement (copy-pasted from [Zarr Docs – Custom Codecs](https://zarr.readthedocs.io/en/latest/user-guide/extending/#custom-codecs))

> Custom codecs should also implement the following methods:
> 
> compute_encoded_size, which returns the byte size of the encoded data given the byte size of the original data. It should raise NotImplementedError for codecs with variable-sized outputs, such as compression codecs.
> validate (optional), which can be used to check that the codec metadata is compatible with the array metadata. It should raise errors if not.
> resolve_metadata (optional), which is important for codecs that change the shape, dtype or fill value of a chunk.
> evolve_from_array_spec (optional), which can be useful for automatically filling in codec configuration metadata from the array metadata.
> 
> Source: [Zarr Docs – Custom Codecs](https://zarr.readthedocs.io/en/latest/user-guide/extending/#custom-codecs)

