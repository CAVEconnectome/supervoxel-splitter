# supervoxel-splitter

[![codecov](https://codecov.io/gh/CAVEconnectome/supervoxel-splitter/graph/badge.svg)](https://codecov.io/gh/CAVEconnectome/supervoxel-splitter)

Pluggable supervoxel-splitting algorithms behind a uniform `Splitter` interface.

## What it does

Given a 3-D foreground mask, source seeds, and sink seeds, return a labeled
volume that partitions the foreground into a source side and a sink side.
Useful for proofreading workflows that need to split a single supervoxel
without touching the rest of the segmentation.

The package ships multiple techniques behind the same `Splitter` protocol:

- **GeodesicSplitter** — anisotropic geodesic carve via an EDT-derived speed
  field. Default technique; preserves the neck-aware behavior used in
  PyChunkedGraph today.
- **WatershedSplitter** — seeded watershed on the distance transform.
- **NoopSplitter** — reference implementation; documents the protocol shape
  for downstream plugin authors (e.g. a learned splitter).

## Install

Requires Python 3.14 or newer.

```
pip install supervoxel-splitter           # core
pip install "supervoxel-splitter[fast]"   # adds dijkstra3d as the geodesic backend
```

## The uniform interface

Every splitter satisfies the same `Splitter` protocol. Technique-specific
knobs live on the constructor; `split()` itself stays clean.

```python
from supervoxel_splitter import GeodesicSplitter

splitter = GeodesicSplitter(backend="dj3d")
result = splitter.split(
    mask=foreground_3d_bool,
    sources=src_seeds_xyz,
    sinks=snk_seeds_xyz,
    voxel_size=(4.0, 4.0, 40.0),
)

# result.labels: uint8 in {0, SOURCE, SINK, STRAY}
# result.side_of_label: {SOURCE: 1, SINK: 2}
# result.diagnostics: per-stage timings, label counts, etc.
```

Plugin authors implement the protocol structurally — no base class to
inherit:

```python
class MyLearnedSplitter:
    def __init__(self, *, model_path):
        ...
    def split(self, mask, sources, sinks, *,
              voxel_size=(1.0, 1.0, 1.0),
              vol_order="xyz", vox_order="xyz", seed_order="xyz"):
        ...
        return SplitResult(...)

assert isinstance(MyLearnedSplitter(model_path="..."), Splitter)
```

## Boundary

The package knows nothing about ChunkedGraph, BigTable, IDs, or edges. It
takes voxels in and gives labels back. New-ID minting, edge updates, and
persistence are the consumer's responsibility.

## Status

v0.1.0 covers `GeodesicSplitter`. `WatershedSplitter` and `NoopSplitter` ship
as small reference impls demonstrating the protocol; production-quality
parity with geodesic is out of scope for v0.1.0.

## Release

Published to PyPI by a one-click workflow — no manual tag or version edit.

- **Release:** Actions → **CI** → **Run workflow** → choose `part` (`major`/`minor`/`patch`), or
  `gh workflow run test.yml -f part=patch`. It computes the next version from the latest `vX.Y.Z`
  tag, tags it, builds, and publishes to PyPI (OIDC trusted publishing).
- **Preview:** `dry-run=true` prints the next version without tagging or publishing.

The version is derived from the git tag by `setuptools_scm`; there is no version literal to bump.

## License

MIT.
