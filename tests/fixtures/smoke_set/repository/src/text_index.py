class SnapshotTextIndex:
    """The FTS5 index stores searchable text for each snapshot path."""

    snapshot_id: str
    searchable_text: str
