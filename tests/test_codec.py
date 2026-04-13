import pytest
import numpy as np
from types import SimpleNamespace

from rmn.fst24file import fst24_file
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import (
    Buffer,
)  # Ajustez selon l'import exact de Buffer dans Zarr v3

# Remplacez par le nom de votre module
from share.my_codec import FSTCodec, _write_record_to_bytes


@pytest.mark.asyncio
async def test_decode_with_real_fst_file():
    # Remplacez ceci par le chemin vers un petit fichier FST de test que vous avez en local
    fichier_fst_test = "eccc-data/2026010700_000"

    # ---------------------------------------------------------
    # ÉTAPE 1 : Lire la "Vérité Absolue" avec la librairie officielle
    # ---------------------------------------------------------
    with fst24_file(fichier_fst_test, "r") as f:
        records = list(f)
        assert len(records) > 0, "Le fichier FST de test est vide"

        # On prend le premier enregistrement pour le test
        record = records[0]

        # On copie les données décodées officiellement dans un tableau NumPy
        # (J'assume ici que record.data renvoie un numpy array ou un buffer compatible)
        donnees_officielles = np.array(record.data, copy=True)

        # On définit les spécifications attendues par Zarr
        shape_attendu = (record.ni, record.nj, record.nk)
        dtype_attendu = donnees_officielles.dtype
        chunk_spec = SimpleNamespace(shape=shape_attendu, dtype=dtype_attendu)

        # On instancie votre codec avec les métadonnées de ce record spécifique
        codec = FSTCodec(
            nomvar=record.nomvar,
            typvar=record.typvar,
            pack_bits=record.pack_bits,
            data_type=record.data_type,
            # ... ajoutez les autres champs nécessaires
        )

    # ---------------------------------------------------------
    # ÉTAPE 2 : Simuler ce que VirtualiZarr va faire (passer des bytes)
    # ---------------------------------------------------------
    # VirtualiZarr lit un bloc de bytes sur le disque. Pour simuler ça,
    # on utilise votre propre helper qui écrit le record en mémoire et sort les bytes.
    raw_bytes = _write_record_to_bytes(record)

    # On encapsule ces octets dans un Buffer Zarr v3
    input_buffer = Buffer.create_zero_length().__class__.from_bytes(raw_bytes)

    # ---------------------------------------------------------
    # ÉTAPE 3 : Décodage via VOTRE logique Zarr
    # ---------------------------------------------------------
    decoded_ndbuffer = await codec._decode_single(input_buffer, chunk_spec)
    donnees_decodees = decoded_ndbuffer.as_numpy_array()

    # ---------------------------------------------------------
    # ÉTAPE 4 : Le verdict
    # ---------------------------------------------------------
    # On compare les shapes
    assert (
        donnees_decodees.shape == donnees_officielles.shape
    ), "Les dimensions ne correspondent pas"

    # On compare les dtypes
    assert (
        donnees_decodees.dtype == donnees_officielles.dtype
    ), "Les types de données ne correspondent pas"

    # On compare les valeurs exactes (ou presque exactes si c'est du float compressé)
    np.testing.assert_array_almost_equal(
        donnees_officielles,
        donnees_decodees,
        err_msg="Les données décodées par le codec diffèrent de la librairie officielle !",
    )
