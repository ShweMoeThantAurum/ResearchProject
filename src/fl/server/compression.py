"""Server-side placeholder for model decompression (future Lambda support)."""

def maybe_decompress(states):
    """
    Return states unchanged.

    Clients currently upload dense models.
    This function exists for future support of compressed uploads.
    """
    return states