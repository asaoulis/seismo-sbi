"""Read and write AxiSEM external background models (``*.bm``).

The format is a small ASCII file: comment lines, ``NAME`` / ``ANELASTIC`` / ``ANISOTROPIC`` /
``UNITS`` headers, a ``COLUMNS`` header naming the fields, then one row per node ordered by
descending radius. A discontinuity is represented by two consecutive rows sharing a radius, the
values just above and just below the boundary; the ``# Discontinuity`` comments AxiSEM emits are
regenerated on write but carry no information. Reading is keyed on ``COLUMNS``, so isotropic and
anisotropic models both round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

#: Surface radius in m the discontinuity-depth annotations are measured from.
DEFAULT_SURFACE_RADIUS_M = 6371000.0

_META_KEYS = ("NAME", "ANELASTIC", "ANISOTROPIC", "UNITS")


@dataclass
class BackgroundModel:
    """An AxiSEM external one-dimensional background model.

    ``columns`` names the fields of ``data``, a ``(n_rows, n_columns)`` table ordered by
    descending radius; ``meta`` holds the header key-value pairs; ``header_comments`` holds the
    leading free-form comment lines, kept verbatim on write.
    """

    columns: list
    data: np.ndarray
    meta: dict = field(default_factory=dict)
    header_comments: list = field(default_factory=list)

    def col_index(self, name: str) -> int:
        return self.columns.index(name)

    def column(self, name: str) -> np.ndarray:
        return self.data[:, self.col_index(name)]

    @property
    def radius(self) -> np.ndarray:
        return self.column("radius")

    @property
    def n_rows(self) -> int:
        return self.data.shape[0]

    def discontinuity_rows(self) -> list:
        """Indices ``i`` where row ``i`` and ``i+1`` share the same radius."""
        r = self.radius
        return [i for i in range(len(r) - 1) if r[i] == r[i + 1]]

    def copy(self) -> "BackgroundModel":
        return BackgroundModel(
            columns=list(self.columns),
            data=self.data.copy(),
            meta=dict(self.meta),
            header_comments=list(self.header_comments),
        )


def read_bm(path) -> BackgroundModel:
    """Parse an AxiSEM ``*.bm`` external background model file."""
    meta: dict = {}
    header_comments: list = []
    columns: list = []
    rows: list = []
    seen_columns = False

    with open(path, "r") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                # Only the leading comments are kept; the discontinuity markers below the
                # COLUMNS header are regenerated on write.
                if not seen_columns:
                    header_comments.append(stripped.lstrip("#").strip())
                continue

            tokens = stripped.split()
            key = tokens[0].upper()
            if key in _META_KEYS:
                meta[key] = " ".join(tokens[1:])
                continue
            if key == "COLUMNS":
                columns = tokens[1:]
                seen_columns = True
                continue

            rows.append([float(t) for t in tokens])

    if not columns:
        raise ValueError(f"No COLUMNS header found in {path}")
    if not rows:
        raise ValueError(f"No data rows found in {path}")

    data = np.array(rows, dtype=float)
    if data.shape[1] != len(columns):
        raise ValueError(
            f"{path}: {data.shape[1]} data columns but COLUMNS header lists "
            f"{len(columns)} ({columns})"
        )
    if "radius" not in columns:
        raise ValueError(f"{path}: COLUMNS header has no 'radius' column")

    return BackgroundModel(
        columns=columns, data=data, meta=meta, header_comments=header_comments
    )


def _format_value(name: str, value: float) -> str:
    """One numeric cell as text: whole metres for a radius, two decimals otherwise."""
    if name == "radius":
        return f"{value:12.1f}"
    if name in ("qka", "qmu"):
        return f"{value:13.1f}"
    return f"{value:11.2f}"


def write_bm(model: BackgroundModel, path, surface_radius_m: float | None = None) -> None:
    """Write ``model`` to ``path`` in AxiSEM ``*.bm`` format.

    The discontinuity markers are regenerated from the duplicated-radius rows, with depths
    measured from ``surface_radius_m``.
    """
    if surface_radius_m is None:
        r = model.radius
        surface_radius_m = float(r.max()) if len(r) else DEFAULT_SURFACE_RADIUS_M

    columns = model.columns
    radius_idx = columns.index("radius")
    disc_rows = set(model.discontinuity_rows())

    lines: list = []
    for c in model.header_comments:
        lines.append(f"# {c}")
    for key in _META_KEYS:
        if key in model.meta:
            lines.append(f"{key:<13s}{model.meta[key]}")
    col_header = "COLUMNS" + "".join(
        f"{name:>11s}" if name not in ("qka", "qmu") and name != "radius"
        else (f"{name:>12s}" if name == "radius" else f"{name:>13s}")
        for name in columns
    )
    lines.append(col_header)

    disc_counter = 0
    for i in range(model.n_rows):
        # The marker goes before the second row of a duplicated-radius pair.
        if i > 0 and model.data[i, radius_idx] == model.data[i - 1, radius_idx]:
            disc_counter += 1
            depth_km = (surface_radius_m - model.data[i, radius_idx]) / 1000.0
            lines.append(
                f"#          Discontinuity {disc_counter:3d}, depth: {depth_km:10.2f} km"
            )
        row_cells = "".join(
            _format_value(name, model.data[i, j]) for j, name in enumerate(columns)
        )
        lines.append(row_cells)

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
