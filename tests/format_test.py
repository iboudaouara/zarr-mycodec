
def detect_fst_format(filepath: str) -> str:
    """
    Ouvre un fichier FST, lit l'en-tête global (les 16 premiers octets)
    et retourne dynamiquement le format ('xdf' ou 'rsf').
    """
    with open(filepath, "rb") as f:
        header = f.read(16)

    if b"XDF" in header:
        return "xdf"
    elif b"RSF" in header:
        return "rsf"
    else:
        raise ValueError(
            f"Fichier invalide ou corrompu. Signature 'XDF' ou 'RSF' "
            f"introuvable dans l'en-tête : {header!r}"
        )


# detected_format = detect_fst_format(fst_file)
# print(f"   -> Format détecté dynamiquement : {detected_format.upper()}")
# assert detected_format == "xdf", "Le fichier devrait être détecté comme XDF"
