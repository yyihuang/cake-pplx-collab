"""
Copyright (c) 2026, NVIDIA CORPORATION.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Registry of the Cake SiTU-GLU training kernels (SM100 / SM103 / SM107),
deterministic: the forward, the backward and the fused backward of a
checkpointed step.

Every record of ``MODULES`` is one exported program: a CUDA device translation
unit plus a tvm-ffi binding, compiled once per architecture it is delivered
for.  A record is keyed by its program name ``cake_situ_glu_det_train_<20 hex>``
and carries the host plan: the ``stage`` it serves (``fwd``, ``bwd`` or
``ckpt``), the architectures it is delivered for, ``rows_per_cta`` (the rows a
CTA owns per block) and the ``run`` entry; ``library`` is its key.  The derived
leaves (``kernel_symbol``, ``launch``, ``arg_plan``) are written by the
exporter from the rendered binding.  ``MODULES`` and ``ROUTES`` are literal
tables (the exporter and the manifest checker read them with
``ast.literal_eval``).  Every launch is a plain stream-ordered launch: nothing
cooperative, no PDL, no dynamic shared memory.

The stages, per row ``r < rows`` and column ``c < width``, from the BF16 gate
``g`` and up ``u`` and the FP32 scalars ``beta`` and ``lb``:

- ``fwd``: ``a = beta tanh(g / beta) sigmoid(g)``, ``u' = lb tanh(u / lb)``
  and ``h = bf16(a u')``, the one rounding.
- ``bwd``: from ``g``, ``u`` and the BF16 ``dh``, with ``s = sigmoid(g)``,
  ``tg = tanh(g / beta)`` and ``tu = tanh(u / lb)``:
  ``d_gate = bf16(dh u' ((1 - tg^2) s + a (1 - s)))`` and
  ``d_up = bf16(dh a (1 - tu^2))``; the activation is recomputed.
- ``ckpt``: the backward of a checkpointed step, in one launch: ``d_gate`` and
  ``d_up`` with the bits ``bwd`` stores, and ``h`` with the bits ``fwd``
  stores.

A stage may ship several programs on an architecture.  They differ in their
schedule only: the CTAs an SM holds, the rows a CTA owns, when the loads are
issued, and whether the host passes ``k1`` and ``k2`` (the FP32 quotients
``-2 log2(e) / beta`` and ``-2 log2(e) / lb``, an ``arg_plan`` that names
them) or the kernel divides.  Every program of a stage stores the bits of the
stage's other programs on that architecture, for every input, row count and
grid, so the route changes no output bit.

``ROUTES`` picks the program of a call: per stage and architecture, a list of
routes, each ``{"program", "min_rows_per_sm", "width", "ctas_per_sm"}``.  A
route holds when ``rows >= min_rows_per_sm * sm_count`` (the device's SM
count) and, when ``width`` is a band ``[lo, hi]``, when ``lo <= width <= hi``.
The host runs the first route that holds (:func:`route`).  The last route of
each stage and architecture holds at every row count and width, and no other
route does.  The grid is one CTA per block of ``rows_per_cta`` rows, capped at
``ctas_per_sm * sm_count`` CTAs when the route states a cap (:func:`grid`).
The CTAs stride over the blocks, so a capped grid runs every row.  One program
may run in two routes with different caps.

The host finds a program through ``ROUTES`` and never spells a program name:
retuning a kernel renames its program (the record's key and ``library``) and
leaves the host code as it is.  Every kernel takes the row count, the width,
``beta``, ``lb`` and every BF16 row stride (in elements) as launch arguments,
so one library per program and architecture serves every call routed to it.
The host launches nothing for ``rows == 0``.
"""

from __future__ import annotations

from typing import Any, Optional

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
PACKAGE = "cake_pplx._backends.cake_situ_glu_det_train"
LIBRARY_PREFIX = "cake_situ_glu_det_train"
MANIFEST = "build.json"

# The stages, in launch order: the forward, the backward, and the backward of a checkpointed step (which also stores
# the forward's h).
STAGES = ("fwd", "bwd", "ckpt")

MODULES: dict[str, dict[str, Any]] = {
    # The forward with the host's k1 and k2: one row a CTA, four warps a row.
    "cake_situ_glu_det_train_e5d3075ea15fefbb9d43": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_e5d3075ea15fefbb9d43",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_e5d3075ea15fefbb9d43",
        "rows_per_cta": 1,
        "stage": "fwd",
    },
    # The forward: h = bf16(beta tanh(g / beta) sigmoid(g) lb tanh(u / lb)), one row a CTA, four warps a row, one
    # segment of 256 columns a warp per pass.
    "cake_situ_glu_det_train_9b922273e948685fab2d": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "h"], ["parameter", "rows"], ["parameter", "width"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "h_stride"], ["parameter", "beta"], ["parameter", "lb"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_9b922273e948685fab2d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_9b922273e948685fab2d",
        "rows_per_cta": 1,
        "stage": "fwd",
    },
    # The forward that issues the next row's loads before it stores a row: one row a CTA, four warps a row.
    "cake_situ_glu_det_train_599ccd99cf9f7a41dae9": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_599ccd99cf9f7a41dae9",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_599ccd99cf9f7a41dae9",
        "rows_per_cta": 1,
        "stage": "fwd",
    },
    # The backward with the host's k1 and k2: one row a CTA, four warps a row.
    "cake_situ_glu_det_train_8ba54d18b9b06905b89e": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_8ba54d18b9b06905b89e",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_8ba54d18b9b06905b89e",
        "rows_per_cta": 1,
        "stage": "bwd",
    },
    # The backward: d_gate and d_up (BF16) from gate, up and dh, the activation recomputed, one row a CTA, four warps
    # a row, one segment of 256 columns a warp per pass.
    "cake_situ_glu_det_train_2ad79222570bbd0c2cef": {
        "arches": ["sm_107a"],
        "arg_plan": [["buffer", "gate"], ["buffer", "up"], ["buffer", "dh"], ["buffer", "d_gate"], ["buffer", "d_up"], ["parameter", "rows"], ["parameter", "width"], ["parameter", "gate_stride"], ["parameter", "up_stride"], ["parameter", "dh_stride"], ["parameter", "d_gate_stride"], ["parameter", "d_up_stride"], ["parameter", "beta"], ["parameter", "lb"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_2ad79222570bbd0c2cef",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_2ad79222570bbd0c2cef",
        "rows_per_cta": 1,
        "stage": "bwd",
    },
    # The backward with the host's k1 and k2, built for 8 CTAs an SM: one row a CTA, four warps a row.
    "cake_situ_glu_det_train_1819a1f2c9babb2a7303": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_1819a1f2c9babb2a7303",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_1819a1f2c9babb2a7303",
        "rows_per_cta": 1,
        "stage": "bwd",
    },
    # The backward of a checkpointed step: d_gate, d_up and h, one row a CTA, four warps a row.
    "cake_situ_glu_det_train_0f6427ea51531187fe83": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_0f6427ea51531187fe83",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_0f6427ea51531187fe83",
        "rows_per_cta": 1,
        "stage": "ckpt",
    },
    # The backward of a checkpointed step with two warps a row: two rows a CTA.
    "cake_situ_glu_det_train_0dff177a39167613b5b1": {
        "arches": ["sm_100a", "sm_103a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "kernel_symbol": "kernel_cake_situ_glu_det_train_0dff177a39167613b5b1",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_situ_glu_det_train_0dff177a39167613b5b1",
        "rows_per_cta": 2,
        "stage": "ckpt",
    },
}

# Stage -> architecture -> the routes the host tries in order; the first that holds runs.  Same-GPU sweeps on B200
# (148 SMs), GB300 (152) and VR200 (212) timed every program against its stage's other programs at R from 1024 to
# 262144 and I from 2048 to 33792: a conditional route runs where its launch measured faster than the next route's.
ROUTES: dict[str, dict[str, list[dict[str, Any]]]] = {
    "fwd": {
        "sm_100a": [
            {
                "program": "cake_situ_glu_det_train_e5d3075ea15fefbb9d43",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
        "sm_103a": [
            {
                "program": "cake_situ_glu_det_train_e5d3075ea15fefbb9d43",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
        "sm_107a": [
            {
                "program": "cake_situ_glu_det_train_599ccd99cf9f7a41dae9",
                "min_rows_per_sm": 67,
                "width": [2048, 3072],
                "ctas_per_sm": 32,
            },
            {
                "program": "cake_situ_glu_det_train_9b922273e948685fab2d",
                "min_rows_per_sm": 0,
                "width": [3072, 6144],
                "ctas_per_sm": 16,
            },
            {
                "program": "cake_situ_glu_det_train_9b922273e948685fab2d",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
    },
    "bwd": {
        "sm_100a": [
            {
                "program": "cake_situ_glu_det_train_8ba54d18b9b06905b89e",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
        "sm_103a": [
            {
                "program": "cake_situ_glu_det_train_8ba54d18b9b06905b89e",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
        "sm_107a": [
            {
                "program": "cake_situ_glu_det_train_1819a1f2c9babb2a7303",
                "min_rows_per_sm": 154,
                "width": [3072, 33792],
                "ctas_per_sm": 96,
            },
            {
                "program": "cake_situ_glu_det_train_1819a1f2c9babb2a7303",
                "min_rows_per_sm": 19,
                "width": [3072, 33792],
                "ctas_per_sm": 48,
            },
            {
                "program": "cake_situ_glu_det_train_2ad79222570bbd0c2cef",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
    },
    "ckpt": {
        "sm_100a": [
            {
                "program": "cake_situ_glu_det_train_0dff177a39167613b5b1",
                "min_rows_per_sm": 83,
                "width": [3072, 5120],
                "ctas_per_sm": None,
            },
            {
                "program": "cake_situ_glu_det_train_0f6427ea51531187fe83",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
        "sm_103a": [
            {
                "program": "cake_situ_glu_det_train_0dff177a39167613b5b1",
                "min_rows_per_sm": 80,
                "width": [3072, 5120],
                "ctas_per_sm": None,
            },
            {
                "program": "cake_situ_glu_det_train_0f6427ea51531187fe83",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
        "sm_107a": [
            {
                "program": "cake_situ_glu_det_train_0f6427ea51531187fe83",
                "min_rows_per_sm": 0,
                "width": [3072, 5120],
                "ctas_per_sm": 32,
            },
            {
                "program": "cake_situ_glu_det_train_0f6427ea51531187fe83",
                "min_rows_per_sm": 0,
                "width": None,
                "ctas_per_sm": None,
            },
        ],
    },
}

# The fields of a route of ``ROUTES``, in the order the table spells them.
ROUTE_FIELDS = ("program", "min_rows_per_sm", "width", "ctas_per_sm")

# The width (I, the columns of gate, up, h and the gradients) is a run-time value: a positive multiple of one segment
# of 256 columns (a warp's 32 lanes of 8 BF16 values).  ``width`` is a 32-bit kernel parameter.
SEGMENT = 256
MAX_WIDTH = 2**31 - SEGMENT

# The kernels load 8 BF16 values (16 bytes) at a time: every BF16 row stride is a multiple of 8 elements and every
# BF16 base address is aligned to 16 bytes.  One rule for every tensor of the family, so a forward's output is a
# backward's input as it is.
VECTOR_ELEMENTS = 8
ALIGNMENT_BYTES = 16

# ``rows`` is a 32-bit kernel parameter, and a CTA steps over its blocks of rows by the grid size times its rows in 32
# bits.  With one CTA per block, a CTA's first row plus its step is at most ``2 * rows - 1`` for one row a CTA and
# ``4 * ceil(rows / 2) - 2`` for two, so up to this many rows no step passes ``2**31 - 1`` (the largest is
# ``2**31 - 2`` for two rows a CTA).  A capped grid has a smaller step.
MAX_ROWS = 2**30


def routes(stage: str, arch: str) -> list[dict[str, Any]]:
    """``ROUTES[stage][arch]``, checked: each route states exactly :data:`ROUTE_FIELDS`, names a record of ``stage``
    delivered for ``arch``, has an integer ``min_rows_per_sm >= 0``, a ``width`` that is None or an inclusive band
    ``[lo, hi]`` of integers with ``0 < lo <= hi``, and a ``ctas_per_sm`` that is None or an integer of at least 1.  The
    last route holds at every row count and width (``min_rows_per_sm`` 0, ``width`` None), and no other route does.
    Anything else is a ``RuntimeError`` naming the route; an unknown stage or architecture is a ``ValueError``."""
    if stage not in STAGES:
        raise ValueError(f"stage is one of {STAGES}, got {stage!r}")
    if arch not in ARCHES:
        raise ValueError(f"arch is one of {ARCHES}, got {arch!r}")
    table = ROUTES[stage][arch]
    for index, entry in enumerate(table):
        at = f"ROUTES[{stage!r}][{arch!r}][{index}]"
        if not isinstance(entry, dict) or tuple(entry) != ROUTE_FIELDS:
            raise RuntimeError(f"{at} must state exactly {ROUTE_FIELDS} in that order, got {entry!r}")
        record = MODULES.get(entry["program"])
        if record is None or record["stage"] != stage or arch not in record["arches"]:
            raise RuntimeError(f"{at}: {entry['program']!r} is not a {stage!r} record delivered for {arch}")
        least, band, cap = entry["min_rows_per_sm"], entry["width"], entry["ctas_per_sm"]
        if type(least) is not int or least < 0:
            raise RuntimeError(f"{at}: min_rows_per_sm {least!r} is not an integer of at least 0")
        if band is not None and (
            len(band) != 2 or any(type(bound) is not int for bound in band) or not 0 < band[0] <= band[1]
        ):
            raise RuntimeError(f"{at}: width {band!r} is not None or an inclusive band [lo, hi], 0 < lo <= hi")
        if cap is not None and (type(cap) is not int or cap < 1):
            raise RuntimeError(f"{at}: ctas_per_sm {cap!r} is not None or an integer of at least 1")
        unconditional = least == 0 and band is None
        if unconditional != (index == len(table) - 1):
            raise RuntimeError(f"{at}: only the last route of {stage!r} on {arch} holds at every row count and width")
    return table


def route(stage: str, arch: str, sm_count: int, rows: int, width: int) -> dict[str, Any]:
    """The route of ``stage`` that runs a launch of ``rows`` rows of width ``width`` on a device of ``arch`` with
    ``sm_count`` SMs: the first route of :func:`routes` with ``rows >= min_rows_per_sm * sm_count`` and, when it states
    a band ``[lo, hi]``, ``lo <= width <= hi``."""
    for entry in routes(stage, arch):
        band = entry["width"]
        if rows >= entry["min_rows_per_sm"] * sm_count and (band is None or band[0] <= width <= band[1]):
            return entry
    raise AssertionError("unreachable: routes() ends with an unconditional route")


def select(stage: str, arch: str, sm_count: int, rows: int, width: int) -> str:
    """The program name that runs ``stage`` for ``rows`` rows of width ``width`` on ``arch`` with ``sm_count`` SMs
    (:func:`route`)."""
    return route(stage, arch, sm_count, rows, width)["program"]


def grid(name: str, rows: int, sm_count: int, ctas_per_sm: Optional[int] = None) -> tuple[int, int, int]:
    """The launch grid of program ``name`` for ``1 <= rows <= MAX_ROWS`` rows on a device of ``sm_count`` SMs: one CTA
    per block of the record's ``rows_per_cta`` rows on the x axis, at most ``ctas_per_sm * sm_count`` CTAs when a cap
    is given (the route's ``ctas_per_sm``).  The kernels stride over the blocks from ``blockIdx.x``, so a smaller grid
    runs every row and stores the same bits."""
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"a launch needs 1 to {MAX_ROWS} rows, got {rows}")
    x = -(-rows // MODULES[name]["rows_per_cta"])
    if ctas_per_sm is not None:
        x = min(x, ctas_per_sm * sm_count)
    return (x, 1, 1)
