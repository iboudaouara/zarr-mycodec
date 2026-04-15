import zarr.registry
import numcodecs.registry

print("--- Zarr v3 Native Codecs ---")
# Zarr v3 keeps its codecs in a protected dictionary. We can access its keys.
if hasattr(zarr.registry, '__codec_registries'):
    print(list(zarr.registry.__codec_registries.keys()))
else:
    print("Could not find v3 registry (API may have changed).")

print("\n--- Numcodecs (Zarr v2) ---")
# This lists all the classic codecs like 'zlib', 'blosc', etc.
print(list(numcodecs.registry.codec_registry.keys()))
