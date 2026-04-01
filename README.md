# Custom Codec for Zarr V3

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
