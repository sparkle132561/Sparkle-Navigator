class SnapshotManifest:
    """A snapshot manifest connects a source commit with file hashes."""

    source_commit: str
    file_hashes: tuple[str, ...]
