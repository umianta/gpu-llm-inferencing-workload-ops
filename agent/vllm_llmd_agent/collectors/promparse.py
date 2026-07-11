"""Minimal Prometheus text-exposition parser.

Just enough to read vLLM / llm-d ``/metrics`` without pulling a heavy
dependency. Returns a list of (name, labels, value) samples. Histograms are
exposed by Prometheus as ``_bucket``/``_sum``/``_count`` series, which this
parser returns verbatim; callers derive averages from ``_sum`` / ``_count``.
"""

from __future__ import annotations

from typing import Iterator, NamedTuple


class Sample(NamedTuple):
    name: str
    labels: dict[str, str]
    value: float


def _parse_labels(blob: str) -> dict[str, str]:
    labels: dict[str, str] = {}
    # blob is the content between { and }, e.g. model_name="x",le="0.1"
    i = 0
    n = len(blob)
    while i < n:
        eq = blob.find("=", i)
        if eq == -1:
            break
        key = blob[i:eq].strip()
        # value is quoted
        q1 = blob.find('"', eq)
        q2 = blob.find('"', q1 + 1)
        # handle escaped quotes
        while q2 != -1 and blob[q2 - 1] == "\\":
            q2 = blob.find('"', q2 + 1)
        if q1 == -1 or q2 == -1:
            break
        val = blob[q1 + 1 : q2].replace('\\"', '"').replace("\\\\", "\\")
        labels[key] = val
        comma = blob.find(",", q2)
        if comma == -1:
            break
        i = comma + 1
    return labels


def parse(text: str) -> Iterator[Sample]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # NAME{labels} VALUE [TIMESTAMP]  |  NAME VALUE
        brace = line.find("{")
        if brace != -1:
            name = line[:brace]
            close = line.rfind("}")
            labels = _parse_labels(line[brace + 1 : close])
            rest = line[close + 1 :].split()
        else:
            parts = line.split()
            name = parts[0]
            labels = {}
            rest = parts[1:]
        if not rest:
            continue
        try:
            value = float(rest[0])
        except ValueError:
            continue
        yield Sample(name, labels, value)
