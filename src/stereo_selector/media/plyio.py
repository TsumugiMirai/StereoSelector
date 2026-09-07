"""Memory-friendly PLY vertex reading with early sampling.

``plyfile`` loads every vertex into memory before any downsampling, which is
wasteful for clouds with millions of points when only a fraction is shown.
Binary PLYs are read through ``numpy.memmap`` so only the sampled rows are
materialized; ASCII files and layouts with variable-length properties fall
back to ``plyfile``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_PLY_SCALAR_TYPES = {
    "char": np.int8,
    "int8": np.int8,
    "uchar": np.uint8,
    "uint8": np.uint8,
    "short": np.int16,
    "int16": np.int16,
    "ushort": np.uint16,
    "uint16": np.uint16,
    "int": np.int32,
    "int32": np.int32,
    "uint": np.uint32,
    "uint32": np.uint32,
    "float": np.float32,
    "float32": np.float32,
    "double": np.float64,
    "float64": np.float64,
}
_PLY_ENDIAN = {"binary_little_endian": "<", "binary_big_endian": ">"}


def _parse_header(path: Path) -> tuple[str, list[dict[str, object]], int]:
    """Return (format, element descriptors, byte length of the header)."""
    with open(path, "rb") as handle:
        header_bytes = bytearray()
        while True:
            line = handle.readline()
            if not line:
                raise ValueError("PLY 文件缺少 end_header")
            header_bytes += line
            if line.strip() == b"end_header":
                break
    text = bytes(header_bytes).decode("ascii", errors="replace")
    lines = [line.strip() for line in text.splitlines()]
    if not lines or lines[0] != "ply":
        raise ValueError("PLY 文件头无效")
    format_name: str | None = None
    elements: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for line in lines[1:]:
        if not line or line.startswith(("comment", "obj_info")):
            continue
        parts = line.split()
        if parts[0] == "format" and len(parts) >= 3:
            format_name = parts[1]
        elif parts[0] == "element" and len(parts) >= 3:
            current = {
                "name": parts[1],
                "count": int(parts[2]),
                "properties": [],
                "list_property": False,
            }
            elements.append(current)
        elif parts[0] == "property" and current is not None:
            if parts[1] == "list":
                current["list_property"] = True
            elif len(parts) >= 3 and parts[1] in _PLY_SCALAR_TYPES:
                current["properties"].append(
                    (parts[2], _PLY_SCALAR_TYPES[parts[1]])
                )
    if format_name is None:
        raise ValueError("PLY 缺少 format 声明")
    return format_name, elements, len(header_bytes)


def _vertex_layout(
    format_name: str,
    elements: list[dict[str, object]],
) -> tuple[int, np.dtype] | None:
    """Return (byte offset, vertex dtype) or None when layout is not memmappable."""
    endian = _PLY_ENDIAN.get(format_name)
    if endian is None:
        return None
    cursor = 0
    for element in elements:
        if element["name"] == "vertex":
            properties = element["properties"]
            if element["list_property"] or not properties:
                return None
            dtype = np.dtype(
                [
                    (name, endian + np.dtype(kind).str[1:])
                    for name, kind in properties
                ]
            )
            return cursor, dtype
        if element["list_property"]:
            return None
        count = int(element["count"])
        itemsize = sum(np.dtype(kind).itemsize for _name, kind in element["properties"])
        cursor += count * itemsize
    return None


def read_ply_vertices(path: Path, sample_target: int) -> tuple[np.ndarray, int]:
    """Read the vertex element, sampling uniformly when it exceeds a target.

    Returns ``(vertices, total_vertex_count)`` where ``vertices`` is a
    structured numpy array and ``total_vertex_count`` is the full element size
    (used to report the true source size after early sampling).
    """
    format_name, elements, header_size = _parse_header(path)
    vertex = next(
        (element for element in elements if element["name"] == "vertex"),
        None,
    )
    if vertex is None:
        raise ValueError("PLY 文件不包含 vertex 元素")
    layout = _vertex_layout(format_name, elements)
    if layout is None:
        from plyfile import PlyData

        ply = PlyData.read(str(path))
        data = ply["vertex"].data
        return data, len(data)

    offset, dtype = layout
    count = int(vertex["count"])
    stride = 1
    if sample_target > 0 and count > sample_target:
        stride = max(1, math.ceil(count / sample_target))
    mapped = np.memmap(
        path,
        dtype=dtype,
        mode="r",
        offset=header_size + offset,
        shape=(count,),
    )
    if stride == 1:
        return np.array(mapped), count
    return np.array(mapped[::stride]), count
