"""Data ingestion: video helpers, dataset discovery, manifests."""

from .video import (
    iter_video_frames,
    read_frames,
    read_frame_indices,
    read_representative_frame,
    video_frame_count,
)
from .dataset import (
    DatasetIndex,
    DatasetLayout,
    default_layout_from_repo,
    discover_captures,
)
from .manifests import read_manifest, write_manifest

__all__ = [
    "iter_video_frames",
    "read_frames",
    "read_frame_indices",
    "read_representative_frame",
    "video_frame_count",
    "DatasetIndex",
    "DatasetLayout",
    "default_layout_from_repo",
    "discover_captures",
    "read_manifest",
    "write_manifest",
]
