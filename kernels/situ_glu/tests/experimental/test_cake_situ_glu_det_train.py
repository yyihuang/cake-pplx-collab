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

Backend suite of the Cake deterministic SiTU-GLU training package (``cake_pplx._backends.cake_situ_glu_det_train``,
the backend of ``cake_pplx.cake_situ_glu_det_train``): ``h = bf16(beta tanh(g / beta) sigmoid(g) lb tanh(u / lb))``, its
gradients ``d_gate`` and ``d_up``, and the fused backward of a checkpointed step (``h`` and both gradients in one
launch), at any width that is a multiple of 256, on SM100, SM103 and SM107.

Without a device: the registry (the eight programs of the three stages and their architectures, the launch facts, the
argument plans, the routes: literal copies of the measured ones), the program of a launch at one row below, at and above
every rows-per-SM edge and at every width-band edge on 148, 152 and 212 SMs (the pure route and the package's
per-device plan through a fake device), the launch grids with and without a cap, the host's ``k1`` and ``k2`` against
the FP32 division, the build manifest (every record's library, the recorded source digests), the capability map, the
``beta`` and ``lb`` keywords and their defaults, the refusal of CPU tensors and no environment read outside the JIT
shim.

On SM100 / SM103 / SM107, every training output (``h``, ``d_gate`` and ``d_up``) judged against the naive BF16 tier
(``tests/cake/test_helpers/cake_naive.py``: the error against an FP64 oracle of the same inputs over 32 independent
draws, in the L2 and the max norm, median and largest, beside the error of the same operation in BF16,
:mod:`tests.cake.test_helpers.situ_glu_reference`) over row counts 1 to 4097, widths 256 to 33,792 (one segment,
partial passes of the CTA's warps, both sides of a full pass, and the issue's widths), the merged, two-tensor, sliced
and copied layouts, saturated, deep negative, small and wide values, zero gates and ups, every BF16 gate and scalars
other than the model's, and tiny gates or ups, one magnitude band a case, down to ``2^-60 beta`` and ``2^-60 lb``.
Beside the judge, exact facts of the contract (no tolerance): every program of a stage on the device's architecture
stores the bits of the stage's last route (forced through a private hook, at the route edges, odd row counts, both
layouts, two values of ``beta``, uncapped and capped grids); the route changes no bit at a row or width edge; the fused
backward stores the forward's ``h`` and the backward's gradients; ``g = 0`` or ``u = 0`` gives ``h = 0`` exactly (and
``d_up = 0`` at ``g = 0``, ``d_gate = 0`` at ``u = 0``); saturated inputs give finite outputs with the oracle's signs
outside the flushed range; ``R = 0`` returns empty outputs and launches nothing; one library per routed program serves
every call; the same bits at every call, in every layout, at large row counts whose offsets pass 2**31 bytes, on a
side stream with no host synchronization at every route, in CUDA graph replays (across route edges) and under
``torch.use_deterministic_algorithms``; the autograd function matches the steps bit for bit, saves only its inputs,
gives the same bits under non-reentrant checkpointing and has no double backward; the documentation's checkpointed
``SituDown`` example saves only its inputs and gives the autograd path's gradient bits; inputs the kernels can read are
read in place, others copied once; unfit outputs, operands and scalars are refused.

The suite names its package once, through the shared alias helper, and imports nothing else from ``cake_pplx``.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import inspect
import itertools
import json
import math
import random
import re
import subprocess
import sys
import types
import warnings
from pathlib import Path

import pytest
import torch
import torch.utils.checkpoint

from tests.cake.test_helpers.cake_package import cake_package
from tests.cake.test_helpers.situ_glu_reference import (
    BETA,
    LB,
    Case,
    arrange,
    fp32,
    judge,
    make_inputs,
    oracle,
)
from tests.cake.test_helpers.situ_glu_reference import outputs as step_outputs

pkg = cake_package("cake_situ_glu_det_train")
cake_jit = cake_package("cake_situ_glu_det_train", "cake_jit")

forward = pkg.cake_situ_glu_det_train_forward
backward = pkg.cake_situ_glu_det_train_backward
recompute = pkg.cake_situ_glu_det_train_backward_recompute
situ_glu = pkg.cake_situ_glu_det

PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
PACKAGE_IMPORT = cake_jit.PACKAGE
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a", (10, 7): "sm_107a"}
STAGES = ("fwd", "bwd", "ckpt")
# The kernel parameters of each stage's programs, in launch order (the kernels' signatures): a program that takes the
# host's scalars takes ``k1`` and ``k2`` after ``lb``.
FWD_PARAMETERS = ("gate", "up", "h", "rows", "width", "gate_stride", "up_stride", "h_stride", "beta", "lb")
BWD_PARAMETERS = (
    "gate",
    "up",
    "dh",
    "d_gate",
    "d_up",
    "rows",
    "width",
    "gate_stride",
    "up_stride",
    "dh_stride",
    "d_gate_stride",
    "d_up_stride",
    "beta",
    "lb",
)
CKPT_PARAMETERS = (*BWD_PARAMETERS[:5], "h", *BWD_PARAMETERS[5:12], "h_stride", "beta", "lb")
HOST_SCALARS = ("k1", "k2")

# The programs (literal copies of the shipped build identities: a retuned program has another name, and this table
# changes with the registry): stage, architectures, rows a CTA owns, and whether it takes the host's k1 and k2.
FWD_HS = "cake_situ_glu_det_train_e5d3075ea15fefbb9d43"  # host scalars
FWD = "cake_situ_glu_det_train_9b922273e948685fab2d"  # the 0.2.1 forward
FWD_RP = "cake_situ_glu_det_train_599ccd99cf9f7a41dae9"  # the next row's loads before a row's stores
BWD_HS = "cake_situ_glu_det_train_8ba54d18b9b06905b89e"  # host scalars
BWD = "cake_situ_glu_det_train_2ad79222570bbd0c2cef"  # the 0.2.1 backward
BWD_MB8_HS = "cake_situ_glu_det_train_1819a1f2c9babb2a7303"  # host scalars, built for 8 CTAs an SM
CKPT_H = "cake_situ_glu_det_train_0f6427ea51531187fe83"  # the backward that also stores h
CKPT_H_W2 = "cake_situ_glu_det_train_0dff177a39167613b5b1"  # the same with two warps a row
PROGRAMS = {
    FWD_HS: ("fwd", ("sm_100a", "sm_103a"), 1, True),
    FWD: ("fwd", ("sm_107a",), 1, False),
    FWD_RP: ("fwd", ("sm_107a",), 1, False),
    BWD_HS: ("bwd", ("sm_100a", "sm_103a"), 1, True),
    BWD: ("bwd", ("sm_107a",), 1, False),
    BWD_MB8_HS: ("bwd", ("sm_107a",), 1, True),
    CKPT_H: ("ckpt", ("sm_100a", "sm_103a", "sm_107a"), 1, False),
    CKPT_H_W2: ("ckpt", ("sm_100a", "sm_103a"), 2, False),
}
# The SM counts of the delivered GPUs: B200, GB300 and VR200.
SM_COUNTS = (148, 152, 212)
# The issue's (R, I) points, the benchmark's.
OWNER_SHAPES = ((262144, 3072), (16380, 6144), (16380, 33792), (16380, 3072), (65536, 3072), (1024, 3072))


def expected_route(stage: str, arch: str, sm_count: int, rows: int, width: int) -> tuple[str, int | None]:
    """``(program, ctas_per_sm)`` of a launch: the measured routes, spelled out (the first that holds)."""
    if stage == "fwd":
        if arch != "sm_107a":
            return FWD_HS, None
        if rows >= 67 * sm_count and 2048 <= width <= 3072:
            return FWD_RP, 32
        if 3072 <= width <= 6144:
            return FWD, 16
        return FWD, None
    if stage == "bwd":
        if arch != "sm_107a":
            return BWD_HS, None
        if rows >= 154 * sm_count and 3072 <= width <= 33792:
            return BWD_MB8_HS, 96
        if rows >= 19 * sm_count and 3072 <= width <= 33792:
            return BWD_MB8_HS, 48
        return BWD, None
    least = {"sm_100a": 83, "sm_103a": 80}.get(arch)
    if least is not None:
        return (CKPT_H_W2, None) if rows >= least * sm_count and 3072 <= width <= 5120 else (CKPT_H, None)
    return (CKPT_H, 32) if 3072 <= width <= 5120 else (CKPT_H, None)


# The rows-per-SM thresholds and the width-band edges of every route (each tried on every architecture and SM count).
ROWS_PER_SM_EDGES = (19, 67, 80, 83, 154)
WIDTH_EDGES = (2048, 3072, 5120, 6144, 33792)
BUFFERS = {"gate", "up", "h", "dh", "d_gate", "d_up"}
# The threads of a CTA of each stage's program.
THREADS = 128
# The smallest number that FP32 and BF16 hold as a normal number, and the largest finite FP32 number.
MIN_NORMAL = 2.0**-126
FP32_MAX = 3.4028234663852886e38

# The judged cases: row counts 1 to 4097, widths 256 to 33792, every layout, the data kinds and other scalars.  A pass
# of either stage's CTA covers 4 segments of 256 columns (1024 columns).
CASES = [
    Case(1),
    Case(2, 256),  # one segment: three warps of the CTA idle
    Case(5, 1280, "merged"),  # five segments: a full pass of the CTA, then a partial one
    Case(33, 1792),  # just below two passes of the CTA
    Case(33, 2048, "merged"),  # two passes of the CTA
    Case(33, 2304),  # just above it
    Case(33, 2816, "merged"),  # just below three passes of the CTA
    Case(33, 3328),  # just above three
    Case(3, 33792, "sliced"),
    Case(129),
    Case(129, layout="merged"),
    Case(129, layout="sliced"),
    Case(129, layout="copied"),
    Case(4097),
    Case(4097, layout="merged"),
    Case(1025, 6144),
    Case(1025, 6144, "merged"),
    Case(257, 33792),
    Case(257, 33792, "merged"),
    Case(257, data="wide"),
    Case(257, data="saturated"),
    Case(257, layout="merged", data="saturated"),
    Case(257, 6144, data="saturated"),
    Case(257, data="deep_negative"),
    Case(257, data="small"),
    # Tiny gates or ups, one band a case: where the two forms of beta (1 - E) meet in both stages (7, y = 2 |g| / beta
    # near 2^-5.8), just below it (8), where 1 - E from E alone is off by percents (20), and deep in the FP32 range,
    # where E rounds to 1 (60).
    Case(65, data="tiny_gate", band=7),
    Case(65, data="tiny_gate", band=8),
    Case(65, data="tiny_gate", band=20),
    Case(65, layout="merged", data="tiny_gate", band=60),
    Case(65, data="tiny_up", band=7),
    Case(65, data="tiny_up", band=20),
    Case(65, 6144, data="tiny_up", band=60),
    Case(65, data="tiny_gate", band=20, beta=2.5, lb=10.0),
    Case(257, data="zero_gate"),
    Case(257, layout="merged", data="zero_up"),
    Case(22, data="all_gates"),  # 22 x 3072 >= 65536 gate patterns
    Case(257, beta=2.5, lb=10.0),  # a beta other than 4: P takes its own exponential
    Case(257, data="saturated", beta=3.0, lb=40.0),
    Case(257, data="deep_negative", beta=3.0),
]


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not pkg.is_cake_situ_glu_det_train_supported(torch.device("cuda")):
        pytest.skip(f"the package ships {ARCHES}; this device is {torch.cuda.get_device_capability()}")
    if not (PACKAGE_DIR / "build.json").is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture(scope="module")
def device() -> torch.device:
    dev = _device_or_skip()
    # Build the local architecture's libraries before the first judged call.
    for spec in pkg.specs_for_arch(pkg.device_arch(dev.index)):
        spec.build_and_load()
    return dev


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.is_file():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture
def launches(monkeypatch: pytest.MonkeyPatch) -> list:
    """``(stage, rows, width, values)`` of every launch, in order."""
    launched: list = []
    run_module = pkg.run_module

    def spy(stage, values, *, rows, width, device_index):
        launched.append((stage, rows, width, dict(values)))
        return run_module(stage, values, rows=rows, width=width, device_index=device_index)

    monkeypatch.setattr(pkg, "run_module", spy)
    return launched


@pytest.fixture
def host_device(monkeypatch: pytest.MonkeyPatch):
    """With no GPU: ``place(arch, sm_count)`` makes every device an ``arch`` device of ``sm_count`` SMs (the
    architecture and the SM count the routes read), and each launch is recorded in ``calls`` as ``(name, arch, args)``
    instead of loading a library.  ``properties`` counts the SM-count queries.  The per-device caches are cleared at
    each placement and after the test."""
    calls: list = []
    properties: list = []

    def clear() -> None:
        pkg._plans.cache_clear()
        pkg._sm_count.cache_clear()

    def place(arch: str, sm_count: int) -> None:
        def query(index):
            properties.append(index)
            return types.SimpleNamespace(multi_processor_count=sm_count)

        monkeypatch.setattr(pkg, "device_arch", lambda index: arch)
        monkeypatch.setattr(torch.cuda, "get_device_properties", query)
        properties.clear()
        clear()

    monkeypatch.setattr(pkg, "_entry", lambda name, arch: lambda *args: calls.append((name, arch, args)))
    clear()
    yield types.SimpleNamespace(place=place, calls=calls, properties=properties)
    clear()


def _programs_of(stage: str, arch: str) -> list[str]:
    """The programs of ``stage`` delivered for ``arch``, in registry order."""
    return [name for name, record in cake_jit.MODULES.items() if record["stage"] == stage and arch in record["arches"]]


def _device_facts(device: torch.device) -> tuple[str, int]:
    """``(arch, sm_count)`` of ``device``."""
    return pkg.device_arch(device.index), int(torch.cuda.get_device_properties(device).multi_processor_count)


def _stage_outputs(stage: str, t: types.SimpleNamespace) -> dict:
    """The outputs of one call of ``stage`` on ``t``: ``h`` (``fwd``), the gradients (``bwd``) or both (``ckpt``, the
    fused backward), the gradients keyed as :func:`_functional` keys them."""
    found = {}
    if stage == "fwd":
        found["h"] = forward(t.first, t.second, beta=t.beta, lb=t.lb)
        return found
    if stage == "bwd":
        d_first, d_up = backward(t.dh, t.first, t.second, beta=t.beta, lb=t.lb)
    else:
        found["h"], d_first, d_up = recompute(t.dh, t.first, t.second, beta=t.beta, lb=t.lb)
    found["d_first"] = d_first
    if d_up is not None:
        found["d_up"] = d_up
    return found


@contextlib.contextmanager
def _forced(monkeypatch: pytest.MonkeyPatch, name: str, ctas_per_sm=None):
    """Within the block, every launch of the stage of program ``name`` runs ``name`` on its grid capped at
    ``ctas_per_sm`` CTAs an SM (``pkg._launch_program``), whatever the route."""
    stage = cake_jit.MODULES[name]["stage"]
    run_module = pkg.run_module

    def run(launched, values, *, rows, width, device_index):
        assert launched == stage, (launched, name)
        pkg._launch_program(name, values, rows=rows, device_index=device_index, ctas_per_sm=ctas_per_sm)

    monkeypatch.setattr(pkg, "run_module", run)
    try:
        yield
    finally:
        monkeypatch.setattr(pkg, "run_module", run_module)


def _route_edges(stage: str, arch: str, sm_count: int) -> list[tuple[int, int]]:
    """``(rows, width)`` of each rows-per-SM edge of ``stage`` on ``arch``: the first row count a route with a row
    threshold holds at, with a width in its band."""
    edges = []
    for entry in cake_jit.ROUTES[stage][arch]:
        if entry["min_rows_per_sm"] > 0:
            width = entry["width"][0] if entry["width"] is not None else 3072
            edges.append((entry["min_rows_per_sm"] * sm_count, width))
    return edges


def _width_edges(stage: str, arch: str, sm_count: int) -> list[tuple[int, int, int]]:
    """``(rows, inside, outside)`` of each width-band edge of ``stage`` on ``arch``: a row count at which the banded
    route's row threshold holds, the band's edge width and the width one segment past it."""
    edges = []
    for entry in cake_jit.ROUTES[stage][arch]:
        if entry["width"] is not None:
            rows = max(129, entry["min_rows_per_sm"] * sm_count + 1)
            lo, hi = entry["width"]
            if lo > cake_jit.SEGMENT:
                edges.append((rows, lo, lo - cake_jit.SEGMENT))
            edges.append((rows, hi, hi + cake_jit.SEGMENT))
    return edges


def assert_same_bits(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape, (what, got.dtype, want.dtype, got.shape, want.shape)
    view = torch.int16 if got.dtype == torch.bfloat16 else torch.int32
    assert torch.equal(got.contiguous().view(view), want.contiguous().view(view)), what


def assert_same_outputs(got: dict, want: dict, what: str) -> None:
    assert set(got) == set(want), what
    for name in want:
        assert_same_bits(got[name], want[name], f"{what}: {name}")


def _relayout(t: types.SimpleNamespace, layout: str) -> types.SimpleNamespace:
    """The inputs ``t`` (same values) in ``layout``."""
    return arrange(t.gate, t.up, t.dh, layout, beta=t.beta, lb=t.lb)


def _rows(t: types.SimpleNamespace, rows: int, offset: int = 0) -> types.SimpleNamespace:
    """Rows ``[offset, offset + rows)`` of the inputs ``t`` (views)."""
    span = slice(offset, offset + rows)
    out = types.SimpleNamespace(R=rows, I=t.I, beta=t.beta, lb=t.lb, layout=t.layout)
    out.gate, out.up, out.dh = t.gate[span], t.up[span], t.dh[span]
    if t.second is None:
        out.first, out.second = t.first[span], None
    else:
        out.first, out.second = out.gate, out.up
    return out


def _leaves(t: types.SimpleNamespace) -> tuple:
    """Fresh leaves ``(first, second)`` of the inputs ``t`` that require grad (None stays None)."""
    return tuple(None if x is None else x.detach().clone().requires_grad_(True) for x in (t.first, t.second))


def _autograd_step(t: types.SimpleNamespace, *, checkpoint: bool = False) -> dict:
    """``h`` and the gradients of the leaves of one autograd step on ``t`` (the forward under non-reentrant
    checkpointing when asked)."""
    leaves = _leaves(t)
    if checkpoint:
        h = torch.utils.checkpoint.checkpoint(situ_glu, *leaves, beta=t.beta, lb=t.lb, use_reentrant=False)
    else:
        h = situ_glu(*leaves, beta=t.beta, lb=t.lb)
    h.backward(t.dh)
    found = {"h": h.detach(), "d_first": leaves[0].grad}
    if leaves[1] is not None:
        found["d_up"] = leaves[1].grad
    return found


def _functional(t: types.SimpleNamespace) -> dict:
    """The steps' ``h`` and gradients on ``t``, keyed as :func:`_autograd_step` keys them."""
    h = forward(t.first, t.second, beta=t.beta, lb=t.lb)
    d_first, d_up = backward(t.dh, t.first, t.second, beta=t.beta, lb=t.lb)
    found = {"h": h, "d_first": d_first}
    if d_up is not None:
        found["d_up"] = d_up
    return found


def _large_inputs(rows: int, width: int, *, seed: int, device: torch.device) -> types.SimpleNamespace:
    """A merged ``[rows, 2 width]`` input and its ``dh``, drawn directly in BF16 (no FP32 copy of a large tensor)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    gate_up = torch.randn(rows, 2 * width, device=device, dtype=torch.bfloat16, generator=gen).mul_(2.0)
    dh = torch.randn(rows, width, device=device, dtype=torch.bfloat16, generator=gen)
    t = types.SimpleNamespace(R=rows, I=width, beta=fp32(BETA), lb=fp32(LB), layout="merged", gate_up=gate_up, dh=dh)
    t.gate, t.up = gate_up[:, :width], gate_up[:, width:]
    t.first, t.second = gate_up, None
    return t


# ----------------------------------------------------------------------------------------------------------------
# Without a device


def test_registry_records_are_well_formed() -> None:
    assert tuple(cake_jit.STAGES) == STAGES
    assert tuple(cake_jit.ARCHES) == ARCHES
    assert dict(cake_jit.ARCH_BY_CAPABILITY) == ARCH_BY_CAPABILITY
    assert cake_jit.SEGMENT == 256
    assert cake_jit.PACKAGE == "cake_pplx._backends.cake_situ_glu_det_train"
    # The eight programs, keyed by program name (their library): stage, architectures (in ARCHES order), rows a CTA.
    assert set(cake_jit.MODULES) == set(PROGRAMS)
    prefix = cake_jit.LIBRARY_PREFIX + "_"
    for name, record in cake_jit.MODULES.items():
        stage, arches, rows_per_cta, _ = PROGRAMS[name]
        assert set(record) == {
            "arches",
            "arg_plan",
            "ffi_entry",
            "kernel_symbol",
            "launch",
            "library",
            "rows_per_cta",
            "stage",
        }, name
        assert record["library"] == name
        assert re.fullmatch(re.escape(prefix) + "[0-9a-f]{20}", name), name
        assert record["stage"] == stage, name
        assert tuple(record["arches"]) == arches, (name, record["arches"])
        assert type(record["rows_per_cta"]) is int and record["rows_per_cta"] == rows_per_cta, name
        assert record["ffi_entry"] == "run", name
        assert record["kernel_symbol"] == "kernel_" + record["library"], name
        launch = record["launch"]
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["dynamic_smem_bytes"] == 0, name
        assert tuple(launch["block"]) == (THREADS, 1, 1), name
    # The BF16 stride and alignment rule is one 16-byte load of 8 values.
    assert cake_jit.VECTOR_ELEMENTS == 8 and cake_jit.ALIGNMENT_BYTES == 16


def test_routes_are_well_formed_and_reach_every_program() -> None:
    """Every stage routes on every architecture: each route states exactly the four fields, names a program of its
    stage delivered for that architecture, and only the last route of a (stage, architecture) holds at every row count
    and width (``cake_jit.routes`` checks them).  Every program runs in some route on each architecture it is
    delivered for (no dead library), and the program of each stage's last route is the one a launch of one row at the
    smallest width runs."""
    assert set(cake_jit.ROUTES) == set(STAGES)
    reached = set()
    for stage in STAGES:
        assert set(cake_jit.ROUTES[stage]) == set(ARCHES), stage
        for arch in ARCHES:
            table = cake_jit.routes(stage, arch)
            assert table is cake_jit.ROUTES[stage][arch]
            for entry in table:
                assert tuple(entry) == cake_jit.ROUTE_FIELDS == ("program", "min_rows_per_sm", "width", "ctas_per_sm")
                reached.add((entry["program"], arch))
            assert table[-1]["min_rows_per_sm"] == 0 and table[-1]["width"] is None, (stage, arch)
            for sm_count in SM_COUNTS:
                assert cake_jit.select(stage, arch, sm_count, 1, 256) == table[-1]["program"], (stage, arch)
    assert reached == {(name, arch) for name, record in cake_jit.MODULES.items() for arch in record["arches"]}
    for stage, arch in (("gate", "sm_100a"), ("fwd", "sm_90a")):
        with pytest.raises(ValueError):
            cake_jit.routes(stage, arch)


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"program": FWD_HS, "min_rows_per_sm": 0, "width": None}, "exactly"),
        ({"program": BWD_HS, "min_rows_per_sm": 0, "width": None, "ctas_per_sm": None}, "record"),
        ({"program": FWD, "min_rows_per_sm": 0, "width": None, "ctas_per_sm": None}, "record"),
        ({"program": FWD_HS, "min_rows_per_sm": -1, "width": [3072, 6144], "ctas_per_sm": None}, "min_rows_per_sm"),
        ({"program": FWD_HS, "min_rows_per_sm": True, "width": [3072, 6144], "ctas_per_sm": None}, "min_rows_per_sm"),
        ({"program": FWD_HS, "min_rows_per_sm": 4, "width": [6144, 3072], "ctas_per_sm": None}, "band"),
        ({"program": FWD_HS, "min_rows_per_sm": 4, "width": [3072], "ctas_per_sm": None}, "band"),
        ({"program": FWD_HS, "min_rows_per_sm": 4, "width": None, "ctas_per_sm": 0}, "ctas_per_sm"),
        ({"program": FWD_HS, "min_rows_per_sm": 0, "width": None, "ctas_per_sm": None}, "only the last"),
    ],
    ids=["fields", "stage", "arch", "negative", "bool", "reversed", "short", "cap", "early-unconditional"],
)
def test_the_routes_refuse_a_malformed_route(row, message, monkeypatch) -> None:
    """A route placed before the last route of ``fwd`` on ``sm_100a`` that states other fields, names a program of
    another stage or architecture, or has a bad bound, band or cap (or holds everywhere) is a ``RuntimeError``."""
    table = {stage: dict(arches) for stage, arches in cake_jit.ROUTES.items()}
    table["fwd"]["sm_100a"] = [row, *cake_jit.ROUTES["fwd"]["sm_100a"]]
    monkeypatch.setattr(cake_jit, "ROUTES", table)
    with pytest.raises(RuntimeError, match=message):
        cake_jit.routes("fwd", "sm_100a")


def _boundary_rows(sm_count: int) -> list[int]:
    rows = {1, 2, 3, 129, 4097, cake_jit.MAX_ROWS}
    rows |= {r for r, _ in OWNER_SHAPES}
    for least in ROWS_PER_SM_EDGES:
        rows |= {least * sm_count - 1, least * sm_count, least * sm_count + 1}
    return sorted(rows)


def _boundary_widths() -> list[int]:
    widths = {256, 1024, 33792 + 256, cake_jit.MAX_WIDTH} | {w for _, w in OWNER_SHAPES}
    for edge in WIDTH_EDGES:
        widths |= {edge - cake_jit.SEGMENT, edge, edge + cake_jit.SEGMENT}
    return sorted(widths)


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stage", STAGES)
def test_routes_at_the_boundaries(stage, arch, sm_count) -> None:
    """``cake_jit.route`` / ``select`` at one row below, at and above every rows-per-SM threshold, at every width-band
    edge and one segment on either side, and at the issue's shapes give the measured program and cap
    (:func:`expected_route`), on each architecture with 148, 152 and 212 SMs."""
    seen = set()
    for rows, width in itertools.product(_boundary_rows(sm_count), _boundary_widths()):
        want = expected_route(stage, arch, sm_count, rows, width)
        entry = cake_jit.route(stage, arch, sm_count, rows, width)
        assert (entry["program"], entry["ctas_per_sm"]) == want, (rows, width)
        assert cake_jit.select(stage, arch, sm_count, rows, width) == want[0]
        seen.add(want)
    assert seen == {(e["program"], e["ctas_per_sm"]) for e in cake_jit.ROUTES[stage][arch]}


@pytest.mark.parametrize("sm_count", SM_COUNTS)
@pytest.mark.parametrize("arch", ARCHES)
@pytest.mark.parametrize("stage", STAGES)
def test_the_package_launches_the_routed_program_on_its_grid(stage, arch, sm_count, host_device) -> None:
    """Through a fake device of ``arch`` with ``sm_count`` SMs: ``run_module`` runs the measured program (with its
    parameters, ``k1`` and ``k2`` only where its argument plan names them) on ``ceil(R / rows_per_cta)`` CTAs, capped at
    ``ctas_per_sm * SMs``, at every boundary shape; the SM count is read once per device."""
    host_device.place(arch, sm_count)
    names = set(CKPT_PARAMETERS) | set(FWD_PARAMETERS) | set(HOST_SCALARS)
    values = {key: f"<{key}>" for key in names}
    widths = [w for w in _boundary_widths() if w <= 2 * 33792]
    for rows, width in itertools.product(_boundary_rows(sm_count), widths):
        name, cap = expected_route(stage, arch, sm_count, rows, width)
        per = PROGRAMS[name][2]
        grid_x = -(-rows // per) if cap is None else min(-(-rows // per), cap * sm_count)
        host_device.calls.clear()
        pkg.run_module(stage, values, rows=rows, width=width, device_index=0)
        want = (*(values[key] for key in pkg.parameters(name)), grid_x, 1, 1)
        assert host_device.calls == [(name, arch, want)], (rows, width)
        assert cake_jit.grid(name, rows, sm_count, cap) == (grid_x, 1, 1)
    assert host_device.properties == [0]
    with pytest.raises(ValueError, match="needs argument"):
        pkg.run_module(stage, {"gate": None}, rows=1, width=256, device_index=0)
    with pytest.raises(ValueError, match="row"):
        pkg.run_module(stage, values, rows=0, width=256, device_index=0)


def test_argument_plans_are_the_kernel_parameters() -> None:
    """Each record takes its kernel's parameters in order (tensors as buffers, scalars as parameters), then the
    grid; the host names every one of them.  ``k1`` and ``k2`` follow ``lb`` exactly in the programs that take the
    host's scalars; the ``ckpt`` programs take ``h`` after ``d_up`` and ``h_stride`` after ``d_up_stride``."""
    base = {"fwd": FWD_PARAMETERS, "bwd": BWD_PARAMETERS, "ckpt": CKPT_PARAMETERS}
    for name, record in cake_jit.MODULES.items():
        stage, _, _, host_scalars = PROGRAMS[name]
        want = base[stage] + (HOST_SCALARS if host_scalars else ())
        assert pkg.parameters(name) == want, name
        for kind, key in record["arg_plan"]:
            if kind == "grid":
                continue
            assert kind == ("buffer" if key in BUFFERS else "parameter"), (name, kind, key)
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == ["grid_x", "grid_y", "grid_z"]


def test_grids_are_one_cta_per_block_capped_per_sm() -> None:
    """One CTA per block of ``rows_per_cta`` rows on the x axis (``ceil(R / 2)`` for two rows a CTA, odd ``R``
    included), at most ``ctas_per_sm * SMs`` CTAs with a cap (one below, at and one above it); at the largest row
    count a CTA's first row plus its step over the rows stays in 32 bits."""
    for name, record in cake_jit.MODULES.items():
        per = record["rows_per_cta"]
        for sm_count in SM_COUNTS:
            for rows in (1, 2, 3, 129, 4097, 16380, 16381, 262144, 1048575, cake_jit.MAX_ROWS):
                assert cake_jit.grid(name, rows, sm_count) == (-(-rows // per), 1, 1), (name, rows)
            for cap in (1, 16, 32, 48, 96):
                full = cap * sm_count * per
                for rows in (1, full - per - 1, full - 1, full, full + 1, full + per + 1, cake_jit.MAX_ROWS):
                    want = min(-(-rows // per), cap * sm_count)
                    assert cake_jit.grid(name, rows, sm_count, cap) == (want, 1, 1), (name, rows, cap)
        x = cake_jit.grid(name, cake_jit.MAX_ROWS, max(SM_COUNTS))[0]
        assert per * (x - 1) + per * x <= 2**31 - 1, name
        for rows in (0, -1, cake_jit.MAX_ROWS + 1):
            with pytest.raises(ValueError):
                cake_jit.grid(name, rows, 148)
    assert cake_jit.grid(CKPT_H_W2, 1, 148) == (1, 1, 1)
    assert cake_jit.grid(CKPT_H_W2, 12285, 148) == (6143, 1, 1)
    assert cake_jit.grid(BWD_MB8_HS, 96 * 212 + 1, 212, 96) == (96 * 212, 1, 1)
    assert cake_jit.grid(FWD, 16 * 212 - 1, 212, 16) == (16 * 212 - 1, 1, 1)


def test_host_scalars_are_the_fp32_division() -> None:
    """``k1`` and ``k2`` of the package are ``np.float32(-2 log2(e)) / np.float32(beta)`` and the same of ``lb`` (FP32
    divisions, round to nearest) for random ``beta`` and ``lb`` over the FP32 range, the model's 4 and 25 and the
    edges where a constant leaves FP32; a pair whose constants (``k1``, ``k2``, ``beta lb``, ``beta (1 + 2^-21)``,
    ``lb (1 + 2^-21)``) are not all finite in FP32 is refused.  ``TWO_LOG2E_F32`` is ``2 log2(e)`` rounded to FP32."""
    np = pytest.importorskip("numpy")
    two = pkg.TWO_LOG2E_F32
    assert float(np.float32(two)) == two
    assert two == float(np.float32(2.0 / math.log(2.0)))
    rng = random.Random(7)
    values = [4.0, 25.0, 2.5, 10.0, 40.0, 3.0, 1.0, 0.5, 7.3, 1e-38, 1e-39, 1e-44, 1e38, 3.4e38]
    values += [rng.uniform(0.1, 100.0) for _ in range(40)]
    values += [math.ldexp(rng.uniform(1.0, 2.0), e) for e in range(-149, 128, 3)]
    # beta where -2 log2(e) / beta leaves FP32 (a subnormal beta near 8.48e-39), and where beta (1 + 2^-21) does.
    edge = two / 3.4028234663852886e38
    values += [edge * (1.0 + d) for d in (-1e-3, -1e-6, 0.0, 1e-6, 1e-3)]
    values += [3.4028234663852886e38 / (1.0 + d) for d in (2.0**-22, 2.0**-21, 2.0**-20)]
    c = np.float32(1.0 + 2.0**-21)
    checked = refused = 0
    for beta in values:
        for lb in (25.0, 1.0, rng.choice(values)):
            b, lv = np.float32(beta), np.float32(lb)
            with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
                k1, k2 = np.float32(-two) / b, np.float32(-two) / lv
                constants = (k1, k2, b * lv, b * c, lv * c)
            ok = b > 0 and lv > 0 and np.isfinite(b) and np.isfinite(lv) and all(np.isfinite(x) for x in constants)
            if ok:
                assert pkg._scalars(beta, lb) == (float(b), float(lv), float(k1), float(k2)), (beta, lb)
                checked += 1
            else:
                with pytest.raises(ValueError):
                    pkg._scalars(beta, lb)
                refused += 1
    assert checked > 100 and refused > 5, (checked, refused)
    model = (float(np.float32(-two) / np.float32(4.0)), float(np.float32(-two) / np.float32(25.0)))
    assert pkg._scalars(4.0, 25.0)[2:] == model


def test_no_record_is_delivered_outside_its_architectures() -> None:
    for name, record in cake_jit.MODULES.items():
        for arch in ARCHES:
            if arch in record["arches"]:
                assert pkg.library_name(name, arch) == f"{record['library']}_{arch}"
            else:
                with pytest.raises(RuntimeError):
                    pkg.library_name(name, arch)
        for arch in ("sm_90a", "sm_110a", "sm_120a"):
            with pytest.raises(RuntimeError):
                pkg.library_name(name, arch)


def test_capabilities_map_to_the_three_architectures(monkeypatch) -> None:
    """(10, 0), (10, 3) and (10, 7) map to ``sm_100a``, ``sm_103a`` and ``sm_107a``; any other capability is no target,
    and a device of one is a ``RuntimeError`` naming the delivered architectures."""
    assert tuple(pkg.ARCHES) == ARCHES
    assert tuple(pkg.SUPPORTED_COMPUTE_CAPABILITIES) == ((10, 0), (10, 3), (10, 7))
    assert pkg.WIDTH_MULTIPLE == 256
    for capability, arch in ARCH_BY_CAPABILITY.items():
        assert pkg.arch_for_capability(capability) == arch
    for capability in ((9, 0), (10, 1), (11, 0), (12, 0), (12, 1)):
        assert pkg.arch_for_capability(capability) is None, capability
    for capability, want in (((10, 7), "sm_107a"), ((9, 0), None)):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None, c=capability: c)
        pkg._arch_of_index.cache_clear()
        try:
            if want is None:
                with pytest.raises(RuntimeError, match="sm_100a.*sm_103a.*sm_107a"):
                    pkg.device_arch(0)
            else:
                assert pkg.device_arch(0) == want
        finally:
            monkeypatch.undo()
            pkg._arch_of_index.cache_clear()


def test_scalars_are_keywords_with_the_model_defaults() -> None:
    """``beta`` and ``lb`` are keyword-only everywhere, with the model's values 4.0 and 25.0 as defaults."""
    for entry in (forward, backward, recompute, situ_glu):
        for name, default in (("beta", 4.0), ("lb", 25.0)):
            parameter = inspect.signature(entry).parameters[name]
            assert parameter.kind is parameter.KEYWORD_ONLY, (entry.__name__, name)
            assert parameter.default == default, (entry.__name__, name)
    x = torch.zeros(2, 2 * 3072, dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        forward(x, None, 4.0)


def test_cpu_tensors_are_refused() -> None:
    x = torch.zeros(2, 2 * 3072, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="CUDA"):
        forward(x)
    with pytest.raises(ValueError, match="CUDA"):
        backward(x[:, :3072], x)
    with pytest.raises(ValueError, match="CUDA"):
        recompute(x[:, :3072], x)
    with pytest.raises(ValueError, match="CUDA"):
        situ_glu(x)


def test_manifest_covers_every_record_and_its_sources() -> None:
    """One library per record and architecture it is delivered for (13: four on ``sm_100a`` and ``sm_103a``, five on
    ``sm_107a``), each with its minimum CUDA toolkit and the digests of its sources."""
    manifest = _manifest_or_skip()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    entries = {entry["name"]: entry for entry in manifest["libraries"]}
    wanted = {f"{record['library']}_{arch}" for record in cake_jit.MODULES.values() for arch in record["arches"]}
    assert set(entries) == wanted
    assert len(entries) == len(manifest["libraries"]) == 13
    per_arch = {arch: sum(entry["arch"] == arch for entry in entries.values()) for arch in ARCHES}
    assert per_arch == {"sm_100a": 4, "sm_103a": 4, "sm_107a": 5}
    min_cuda = {"sm_100a": "12.8", "sm_103a": "12.9", "sm_107a": "13.5"}
    for name, entry in entries.items():
        assert entry["arch"] in ARCHES, name
        assert name.endswith("_" + entry["arch"]), name
        assert entry["min_cuda"] == min_cuda[entry["arch"]], name
        assert entry["path"] == f"aot/{name}.so", name
        assert entry["entry"] == "run", name
        for rel, digest in entry["sha256"].items():
            assert hashlib.sha256((PACKAGE_DIR / rel).read_bytes()).hexdigest() == digest, (name, rel)
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


def test_the_loader_imports_standalone_by_path() -> None:
    """The vendorable form: ``cake_build.py`` loaded by path in a fresh process reads the manifest."""
    _manifest_or_skip()
    loader = str(PACKAGE_DIR / "cake_build.py")
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {loader!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'], len(m['libraries']))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    records = sum(len(record["arches"]) for record in cake_jit.MODULES.values())
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT} {records}"


# The variables the JIT shim (``_cake_pplx_jit.py``) reads: the toolchain, cache and JIT switches.
JIT_SHIM_ENVIRONMENT_VARIABLES = frozenset(
    {
        "CUDA_HOME",
        "CUDA_PATH",
        "CAKE_PPLX_CACHE_DIR",
        "CAKE_PPLX_NVCC",
        "CXX",
        "CAKE_PPLX_JIT_WORKERS",
        "CAKE_PPLX_DISABLE_JIT",
    }
)


def _environment_reads(tree: ast.AST) -> tuple[set[str], list[int]]:
    """``(names, unresolved)``: the variable names ``tree`` reads through ``os.environ.get(...)``,
    ``os.environ[...]``, ``... in os.environ`` or ``os.getenv(...)`` -- a string literal, or a loop variable over a
    tuple / list of string literals -- and the lines of every other ``environ`` / ``getenv`` use."""
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def keys(expr: ast.AST, at: ast.AST) -> set[str] | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.Name):
            node = parents.get(at)
            while node is not None:
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == expr.id
                    and isinstance(node.iter, (ast.Tuple, ast.List))
                    and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in node.iter.elts)
                ):
                    return {e.value for e in node.iter.elts}
                node = parents.get(node)
        return None

    names: set[str] = set()
    unresolved: list[int] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in ("environ", "getenv")):
            continue
        parent = parents.get(node)
        found = None
        if node.attr == "getenv":
            if isinstance(parent, ast.Call) and parent.func is node and parent.args:
                found = keys(parent.args[0], parent)
        elif isinstance(parent, ast.Attribute) and parent.attr == "get":
            call = parents.get(parent)
            if isinstance(call, ast.Call) and call.func is parent and call.args:
                found = keys(call.args[0], call)
        elif isinstance(parent, ast.Subscript) and parent.value is node:
            found = keys(parent.slice, parent)
        elif (
            isinstance(parent, ast.Compare)
            and len(parent.ops) == 1
            and isinstance(parent.ops[0], (ast.In, ast.NotIn))
            and parent.comparators[0] is node
        ):
            found = keys(parent.left, parent)
        if found is None:
            unresolved.append(node.lineno)
        else:
            names |= found
    return names, unresolved


def test_the_package_reads_no_environment_variable() -> None:
    """No module of the package reads an environment variable, except the JIT shim (``_cake_pplx_jit.py``), which
    reads exactly :data:`JIT_SHIM_ENVIRONMENT_VARIABLES`, each by a literal name."""
    offenders = []
    shim = None
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        names, unresolved = _environment_reads(ast.parse(path.read_text()))
        if path.name == "_cake_pplx_jit.py":
            shim = names
            offenders += [f"{path.name}:{line} (variable not resolved)" for line in unresolved]
        else:
            offenders += [f"{path.name}:{line}" for line in unresolved]
            offenders += [f"{path.name}: {name}" for name in sorted(names)]
    assert not offenders, offenders
    assert shim == JIT_SHIM_ENVIRONMENT_VARIABLES, (
        sorted(shim - JIT_SHIM_ENVIRONMENT_VARIABLES) if shim is not None else "no _cake_pplx_jit.py",
        sorted(JIT_SHIM_ENVIRONMENT_VARIABLES - shim) if shim is not None else None,
    )


# ----------------------------------------------------------------------------------------------------------------
# On SM100 / SM103 / SM107: accuracy, judged against the naive BF16 tier


def test_the_records_load(device) -> None:
    """Every record delivered for the device's architecture loads; the others are not built for it."""
    arch = pkg.device_arch(device.index)
    assert arch in ARCHES
    for name, record in cake_jit.MODULES.items():
        if arch in record["arches"]:
            assert callable(getattr(pkg.load(name, arch), record["ffi_entry"])), name
        else:
            with pytest.raises(RuntimeError):
                pkg.load(name, arch)


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_outputs_meet_the_naive_tier(case, device, record_property) -> None:
    """``h``, ``d_gate`` and ``d_up`` hold their contract (the oracle's shape, finite where it is finite: every input
    here is finite) and meet the naive BF16 tier over 32 draws."""
    judge(record_property, case, forward, backward, device=device)


# ----------------------------------------------------------------------------------------------------------------
# Exact facts of the contract


@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize("width", [256, 3072, 33792])
def test_a_zero_gate_or_up_gives_exact_zeros(layout, width, device) -> None:
    """Where ``g = 0`` or ``u = 0`` (a random third of the gates and of the ups), ``h`` is exactly 0; ``d_up`` is
    exactly 0 where ``g = 0`` (``a = 0``) and ``d_gate`` where ``u = 0`` (``u' = 0``), also with other scalars.
    Elsewhere the outputs are not all 0."""
    for beta, lb, seed in ((BETA, LB, 1), (2.5, 10.0, 2)):
        gen = torch.Generator(device=device).manual_seed(seed)
        rows = 33
        g = 2.0 * torch.randn(rows, width, device=device, generator=gen)
        u = 2.0 * torch.randn(rows, width, device=device, generator=gen)
        g = torch.where(torch.rand(rows, width, device=device, generator=gen) < 1 / 3, 0.0, g).bfloat16()
        u = torch.where(torch.rand(rows, width, device=device, generator=gen) < 1 / 3, 0.0, u).bfloat16()
        dh = torch.randn(rows, width, device=device, generator=gen).bfloat16()
        t = arrange(g, u, dh, layout, beta=fp32(beta), lb=fp32(lb))
        found = step_outputs(forward, backward, t)
        zero_g, zero_u = g == 0, u == 0
        assert bool(zero_g.any() and zero_u.any())
        assert bool((found["h"][zero_g | zero_u] == 0).all()), "h where g = 0 or u = 0"
        assert bool((found["d_up"][zero_g] == 0).all()), "d_up where g = 0"
        assert bool((found["d_gate"][zero_u] == 0).all()), "d_gate where u = 0"
        live = ~(zero_g | zero_u)
        for name in ("h", "d_gate", "d_up"):
            assert bool((found[name][live] != 0).any()), name


@pytest.mark.parametrize("layout", ["split", "merged"])
@pytest.mark.parametrize(("beta", "lb"), [(BETA, LB), (3.0, 40.0)], ids=["model", "other"])
def test_saturated_inputs_are_finite_with_the_oracle_signs(layout, beta, lb, device) -> None:
    """Gates uniform in ``[-200, 200]`` and ups in ``[-1000, 1000]``, with the probes ``g`` in
    ``{-200, -20, 0, 20, 200}`` and ``u`` in ``{-1000, -50, 0, 50, 1000}`` crossed in the first row: every output is
    finite, and has the sign of the float64 oracle wherever the oracle is at least the smallest normal number (for
    ``d_gate``, where ``|g| >= 20``: there the gate derivative has no zero, so its sign is the operation's, not a
    rounding's).  Past ``|g| = 43 beta`` (``d_gate``) and ``|u| = 43 lb`` (``d_up``) the kernels flush the
    exponential, so a gradient under ``2^-124 |dh| lb`` or ``2^-124 |dh| beta`` is stored as 0: those elements are
    only checked to be finite."""
    gen = torch.Generator(device=device).manual_seed(5)
    rows, width = 129, 3072
    g = -200.0 + 400.0 * torch.rand(rows, width, device=device, generator=gen)
    u = -1000.0 + 2000.0 * torch.rand(rows, width, device=device, generator=gen)
    probes_g = torch.tensor([-200.0, -20.0, 0.0, 20.0, 200.0], device=device)
    probes_u = torch.tensor([-1000.0, -50.0, 0.0, 50.0, 1000.0], device=device)
    g[0, :25] = probes_g.repeat_interleave(5)
    u[0, :25] = probes_u.repeat(5)
    dh = torch.randn(rows, width, device=device, generator=gen).bfloat16()
    t = arrange(g.bfloat16(), u.bfloat16(), dh, layout, beta=fp32(beta), lb=fp32(lb))
    found = step_outputs(forward, backward, t)
    want = oracle(t)
    gate = t.gate.float()
    for name in ("h", "d_gate", "d_up"):
        value = found[name]
        assert bool(value.isfinite().all()), name
        signed = want[name].abs() >= MIN_NORMAL
        if name == "d_gate":
            signed &= (gate.abs() >= 20.0) & (gate.abs() <= 43.0 * beta)
        if name == "d_up":
            signed &= t.up.float().abs() <= 43.0 * lb
        assert bool(signed.any()), name
        assert torch.equal(torch.sign(value.double())[signed], torch.sign(want[name])[signed]), name


@pytest.mark.parametrize("layout", ["split", "merged", "sliced"])
def test_no_rows_return_empty_outputs_and_launch_nothing(layout, device, monkeypatch) -> None:
    """``R = 0``: the steps, the fused backward and the autograd function return empty outputs of the right shapes
    and dtypes, and nothing is launched or loaded."""

    def refuse(*args, **kwargs):
        raise AssertionError("a call with no rows launched or loaded a kernel")

    monkeypatch.setattr(pkg, "run_module", refuse)
    monkeypatch.setattr(pkg, "load", refuse)
    monkeypatch.setattr(pkg, "_entry", refuse)
    t = make_inputs(Case(0, 3072, layout), seed=1, device=device)
    h = forward(t.first, t.second)
    assert h.shape == (0, 3072) and h.dtype == torch.bfloat16 and h.device == device
    d_first, d_up = backward(t.dh, t.first, t.second)
    if layout == "merged":
        assert d_first.shape == (0, 2 * 3072) and d_up is None
    else:
        assert d_first.shape == (0, 3072) and d_up.shape == (0, 3072)
    assert d_first.dtype == torch.bfloat16
    h, d_first, d_up = recompute(t.dh, t.first, t.second)
    assert h.shape == (0, 3072) and h.dtype == torch.bfloat16 and h.device == device
    if layout == "merged":
        assert d_first.shape == (0, 2 * 3072) and d_up is None
    else:
        assert d_first.shape == (0, 3072) and d_up.shape == (0, 3072)
    leaves = _leaves(t)
    out = situ_glu(*leaves)
    assert out.shape == (0, 3072)
    out.sum().backward()
    for leaf in leaves:
        if leaf is not None:
            assert leaf.grad.shape == leaf.shape


def test_one_library_per_program_serves_every_call(device, monkeypatch) -> None:
    """Row counts 1 to 16380, widths 256 to 33792, every layout and other scalars load one library per program that a
    call routes to, once: no recompilation (``rows``, ``width``, ``beta``, ``lb`` and the strides are launch
    arguments)."""
    cake_build = cake_package("cake_situ_glu_det_train", "cake_build")
    loaded: list[str] = []
    load_library = cake_build.load_library

    def spy(name):
        loaded.append(name)
        return load_library(name)

    monkeypatch.setattr(cake_build, "load_library", spy)
    pkg.load.cache_clear()
    pkg._entry.cache_clear()
    cases = (
        Case(1),
        Case(4097, 3072, "merged"),
        Case(3, 256, "sliced", beta=2.5, lb=10.0),
        Case(129, 6144, "copied"),
        Case(2, 33792, "merged", beta=3.0),
        Case(16380, 6144),
        Case(16380, 3072, "merged"),
    )
    try:
        for case in cases:
            t = make_inputs(case, seed=5, device=device)
            for stage in STAGES:
                _stage_outputs(stage, t)
    finally:
        pkg.load.cache_clear()
        pkg._entry.cache_clear()
    arch, sm_count = _device_facts(device)
    want = {
        pkg.library_name(cake_jit.select(stage, arch, sm_count, case.R, case.I), arch)
        for case in cases
        for stage in STAGES
    }
    assert sorted(loaded) == sorted(want)


@pytest.mark.parametrize(
    "case",
    [Case(1), Case(129, 3072, "merged", "saturated"), Case(257, 33792, "sliced"), Case(4097, 6144)],
    ids=lambda case: case.id,
)
def test_outputs_are_the_same_bits_at_every_call(case, device) -> None:
    t = make_inputs(case, seed=21, device=device)
    want = step_outputs(forward, backward, t)
    for k in range(4):
        assert_same_outputs(step_outputs(forward, backward, t), want, f"call {k + 1}")


@pytest.mark.parametrize("width", [256, 3072, 33792])
def test_layouts_give_the_same_bits(width, device) -> None:
    """The same values as two contiguous tensors, one merged tensor, row slices of wider buffers (read in place) and
    rows the host copies give the same bits."""
    t = make_inputs(Case(65, width, "split", "wide"), seed=4, device=device)
    want = step_outputs(forward, backward, t)
    for layout in ("merged", "sliced", "copied"):
        assert_same_outputs(step_outputs(forward, backward, _relayout(t, layout)), want, layout)


def _stage_shapes(stage: str, arch: str, sm_count: int) -> list[tuple[int, int]]:
    """Shapes of ``stage`` that every program of it runs in the suite: one row, odd row counts (two rows a CTA leave
    one alone), a partial pass of the CTA, the issue's dense width, and each rows-per-SM edge (one row below, at and
    above it, at a width in the route's band)."""
    shapes = [(1, 3072), (3, 2048), (129, 256), (1025, 6144), (257, 33792), (4097, 5120)]
    for rows, width in _route_edges(stage, arch, sm_count):
        shapes += [(rows - 1, width), (rows, width), (rows + 1, width)]
    return shapes


@pytest.mark.parametrize("stage", STAGES)
def test_every_program_stores_the_bits_of_its_stage(stage, device, monkeypatch) -> None:
    """Every program of ``stage`` on the device's architecture, forced through the private hook (``_launch_program``)
    on its full grid, on a grid of one CTA an SM and on each cap its routes state, stores the bits of the stage's last
    route's program on its full grid: at the stage's shapes (:func:`_stage_shapes`), merged and two tensors, at the
    model's scalars and at ``beta`` 2.5, ``lb`` 10.  Programs of other architectures are not run."""
    arch, sm_count = _device_facts(device)
    names = _programs_of(stage, arch)
    default = cake_jit.ROUTES[stage][arch][-1]["program"]
    caps = {name: {None, 1} for name in names}
    for entry in cake_jit.ROUTES[stage][arch]:
        caps[entry["program"]].add(entry["ctas_per_sm"])
    for (rows, width), layout, (beta, lb) in itertools.product(
        _stage_shapes(stage, arch, sm_count), ("split", "merged"), ((BETA, LB), (2.5, 10.0))
    ):
        t = make_inputs(Case(rows, width, layout, beta=beta, lb=lb), seed=rows % 97, device=device)
        with _forced(monkeypatch, default):
            want = _stage_outputs(stage, t)
        for name in names:
            for cap in sorted(caps[name], key=lambda c: -1 if c is None else c):
                with _forced(monkeypatch, name, cap):
                    got = _stage_outputs(stage, t)
                assert_same_outputs(got, want, f"{name} cap {cap} at R {rows} I {width} {layout} beta {beta}")


@pytest.mark.parametrize("stage", STAGES)
def test_the_route_changes_no_bit(stage, device) -> None:
    """At each rows-per-SM edge ``e`` of the stage on the device: the first ``e - 1`` rows of calls on ``e - 1``,
    ``e`` and ``e + 1`` rows (views of one input) are the same bits, though the calls run different programs or caps.
    At each width-band edge: the columns of the narrower call are the same bits as those columns of a call one segment
    wider (the narrower inputs are column slices of the wider ones, read in place)."""
    arch, sm_count = _device_facts(device)
    rows_edges = _route_edges(stage, arch, sm_count)
    width_edges = _width_edges(stage, arch, sm_count)
    if not rows_edges and not width_edges:
        pytest.skip(f"{stage} has one route on {arch}")
    for layout in ("split", "merged"):
        for edge, width in rows_edges:
            t = make_inputs(Case(edge + 1, width, layout), seed=edge % 89, device=device)
            programs = {cake_jit.select(stage, arch, sm_count, r, width) for r in (edge - 1, edge)}
            want = _stage_outputs(stage, _rows(t, edge - 1))
            for rows in (edge, edge + 1):
                got = _stage_outputs(stage, _rows(t, rows))
                for name, value in got.items():
                    assert_same_bits(value[: edge - 1], want[name], f"{name}: R {rows} vs {edge - 1} {programs}")
    for rows, inside, outside in width_edges:
        narrow, wide = sorted((inside, outside))
        t = make_inputs(Case(rows, wide, "split"), seed=wide % 83, device=device)
        part = types.SimpleNamespace(R=rows, I=narrow, beta=t.beta, lb=t.lb, layout="sliced")
        part.gate, part.up, part.dh = t.gate[:, :narrow], t.up[:, :narrow], t.dh[:, :narrow]
        part.first, part.second = part.gate, part.up
        programs = {cake_jit.select(stage, arch, sm_count, rows, w) for w in (narrow, wide)}
        want = _stage_outputs(stage, t)
        got = _stage_outputs(stage, part)
        for name, value in got.items():
            assert_same_bits(value, want[name][:, :narrow], f"{name}: I {narrow} vs {wide} at R {rows} {programs}")


@pytest.mark.parametrize(
    ("rows", "width"),
    [(262144, 3072), (1048576, 3072), (65536, 33792)],
    ids=["R262144-I3072", "R1048576-I3072", "R65536-I33792"],
)
def test_large_row_counts(rows, width, device) -> None:
    """The issue's routed call (262,144 rows), 1,048,576 rows, and 65,536 rows of the dense width, merged: byte
    offsets past 2**31 in every case, element offsets past 2**32 in the last two.  Finite outputs, the same bits at a
    second call and in the fused backward, and the first and the last 4097 rows the bits of a call on those 4097 rows
    (views of the inputs, the last ones at a large row offset)."""
    t = _large_inputs(rows, width, seed=31, device=device)
    want = step_outputs(forward, backward, t)
    for name, value in want.items():
        assert bool(value.isfinite().all()), name
    again = step_outputs(forward, backward, t)
    assert_same_outputs(again, want, "second call")
    del again
    h, d_first, _ = recompute(t.dh, t.first, None)
    assert_same_bits(h, want["h"], "fused backward: h")
    assert_same_bits(d_first[:, :width], want["d_gate"], "fused backward: d_gate")
    assert_same_bits(d_first[:, width:], want["d_up"], "fused backward: d_up")
    del h, d_first
    for offset in (0, rows - 4097):
        part = step_outputs(forward, backward, _rows(t, 4097, offset=offset))
        for name, value in part.items():
            assert_same_bits(value, want[name][offset : offset + 4097], f"{name} of rows {offset} to {offset + 4096}")


def test_beta_and_lb_are_rounded_to_fp32_once(device) -> None:
    """An integer scalar gives the bits of the same float; a scalar between two FP32 values gives the bits of its FP32
    rounding."""
    t = make_inputs(Case(33, 3072, "merged"), seed=2, device=device)
    want = forward(t.first, None, beta=4.0, lb=25.0)
    assert_same_bits(forward(t.first, None, beta=4, lb=25), want, "integer scalars")
    near = 3.0 + 2.0**-30  # rounds to 3.0 in FP32
    assert_same_bits(forward(t.first, None, beta=near, lb=near), forward(t.first, None, beta=3.0, lb=3.0), "rounding")


# ----------------------------------------------------------------------------------------------------------------
# Autograd, streams, graphs and the determinism setting


@pytest.mark.parametrize("layout", ["merged", "split", "sliced"])
def test_autograd_matches_the_steps_and_checkpoint(layout, device) -> None:
    """The autograd function gives the steps' bits (a merged input's gradient is one ``[R, 2I]`` tensor), saves only
    its inputs (no activation, no FP32 intermediate), and gives the same bits under non-reentrant checkpointing."""
    t = make_inputs(Case(513, 3072, layout), seed=8, device=device)
    want = _functional(t)
    if layout == "merged":
        assert want["d_first"].shape == (t.R, 2 * t.I)
    assert_same_outputs(_autograd_step(t), want, "autograd")
    assert_same_outputs(_autograd_step(t, checkpoint=True), want, "checkpoint")

    saved = []

    def pack(x):
        saved.append((str(x.dtype), x.numel(), x.data_ptr()))
        return x

    leaves = _leaves(t)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        situ_glu(*leaves, beta=t.beta, lb=t.lb)
    expected = [(str(x.dtype), x.numel(), x.data_ptr()) for x in leaves if x is not None]
    assert sorted(saved) == sorted(expected)


def test_autograd_returns_the_gradients_that_are_needed(device) -> None:
    """Only the inputs that require grad get one; ``beta`` and ``lb`` get none; an expanded upstream gradient
    (``h.sum()``) gives the bits of the steps with a gradient of ones."""
    t = make_inputs(Case(129, 3072, "split"), seed=12, device=device)
    want = _functional(t)
    gate = t.gate.detach().clone().requires_grad_(True)
    up = t.up.detach().clone()
    situ_glu(gate, up).backward(t.dh)
    assert_same_bits(gate.grad, want["d_first"], "d_gate alone")
    assert up.grad is None

    m = make_inputs(Case(129, 6144, "merged"), seed=13, device=device)
    leaves = _leaves(m)
    situ_glu(*leaves).sum().backward()
    ones = torch.ones(m.R, m.I, dtype=torch.bfloat16, device=device)
    d_first, _ = backward(ones, m.first, None)
    assert_same_bits(leaves[0].grad, d_first, "merged gradient of h.sum()")


def _ckpt_shapes(arch: str, sm_count: int) -> list[tuple[int, int]]:
    """Row counts 0 and 1, odd counts, the fused backward's rows-per-SM edges on ``arch`` (one row below, at and above)
    and both sides of its width bands."""
    shapes = [(0, 3072), (1, 3072), (1, 256), (129, 6144), (1025, 5120)]
    for rows, width in _route_edges("ckpt", arch, sm_count):
        shapes += [(rows - 1, width), (rows, width), (rows + 1, width)]
    for rows, inside, outside in _width_edges("ckpt", arch, sm_count):
        shapes += [(rows, inside), (rows, outside)]
    return shapes


@pytest.mark.parametrize(("beta", "lb"), [(BETA, LB), (2.5, 10.0)], ids=["model", "other"])
@pytest.mark.parametrize("layout", ["split", "merged"])
def test_recompute_h_is_the_forward_bits(layout, beta, lb, device) -> None:
    """The fused backward of a checkpointed step stores ``h`` with the forward's bits and ``d_gate`` and ``d_up`` with
    the backward's bits (one ``[R, 2I]`` gradient for a merged input), at ``R`` 0 and 1, odd row counts, its route
    edges and both sides of its width bands, at the model's and other scalars."""
    arch, sm_count = _device_facts(device)
    for rows, width in _ckpt_shapes(arch, sm_count):
        t = make_inputs(Case(rows, width, layout, beta=beta, lb=lb), seed=rows % 71, device=device)
        want = _functional(t)
        got = _stage_outputs("ckpt", t)
        assert_same_outputs(got, want, f"R {rows} I {width} ({cake_jit.select('ckpt', arch, sm_count, rows, width)})")
        if layout == "merged":
            assert got["d_first"].shape == (rows, 2 * width) and "d_up" not in got


def test_recompute_buffers_and_refusals(device, launches) -> None:
    """The fused backward writes preallocated outputs (contiguous, or row slices of wider aligned buffers) and returns
    them, with the bits of new outputs; it reads the inputs in place as the backward does (the halves of a merged
    input, ``dh``) and passes ``h`` and its row stride; unfit outputs, ``grad_up`` with a merged input and wrong
    operands are refused before any launch."""
    width = 3072
    t = make_inputs(Case(65, width, "merged"), seed=16, device=device)
    want = _functional(t)

    def wide(rows, cols, pad=8):
        return torch.empty(rows, cols + 2 * pad, dtype=torch.bfloat16, device=device)[:, pad : pad + cols]

    for h, d_gu in (
        (torch.empty(t.R, width, dtype=torch.bfloat16, device=device), None),
        (wide(t.R, width), wide(t.R, 2 * width)),
    ):
        result = recompute(t.dh, t.first, None, out=h, grad_gate_up_or_gate=d_gu)
        assert result[0] is h and (d_gu is None or result[1] is d_gu) and result[2] is None
        assert_same_bits(h, want["h"], "out")
        assert_same_bits(result[1], want["d_first"], "grad_gate_up_or_gate")
    stage, rows, cols, values = launches[-1]
    assert (stage, rows, cols) == ("ckpt", t.R, width)
    assert values["gate"].data_ptr() == t.gate_up.data_ptr()
    assert values["up"].data_ptr() == t.gate_up.data_ptr() + 2 * width
    assert values["h"].data_ptr() == h.data_ptr() and values["h_stride"] == h.stride(0) == width + 16
    assert values["d_up"].data_ptr() == values["d_gate"].data_ptr() + 2 * width

    s = make_inputs(Case(65, width, "split"), seed=17, device=device)
    want_s = _functional(s)
    h, d_gate, d_up = wide(s.R, width), wide(s.R, width), wide(s.R, width)
    result = recompute(s.dh, s.gate, s.up, out=h, grad_gate_up_or_gate=d_gate, grad_up=d_up)
    assert result[0] is h and result[1] is d_gate and result[2] is d_up
    assert_same_outputs({"h": h, "d_first": d_gate, "d_up": d_up}, want_s, "split buffers")

    launches.clear()
    raw = torch.empty(t.R * 2 * width + 64, dtype=torch.bfloat16, device=device)
    unfit = {
        "an FP32 output": torch.empty(t.R, width, device=device),
        "another shape": torch.empty(t.R + 1, width, dtype=torch.bfloat16, device=device),
        "a base that is not 16-byte aligned": raw[1 : 1 + t.R * width].view(t.R, width),
        "a row stride that is not a multiple of 8": raw[: t.R * (width + 2)].view(t.R, width + 2)[:, :width],
        "overlapping rows": raw.as_strided((t.R, width), (8, 1)),
        "a non-unit last stride": raw[: t.R * width * 2].view(t.R, width, 2)[:, :, 0],
    }
    for what, buffer in unfit.items():
        for kwargs in ({"out": buffer}, {"grad_gate_up_or_gate": buffer}, {"grad_up": buffer}):
            with pytest.raises(ValueError):
                recompute(s.dh, s.gate, s.up, **kwargs)
    with pytest.raises(ValueError, match="grad_up"):
        recompute(t.dh, t.first, None, grad_up=torch.empty_like(t.dh))
    with pytest.raises(ValueError):
        recompute(t.dh[:, :256], t.first, None)
    with pytest.raises(ValueError):
        recompute(t.dh, t.first, None, beta=1e-39)
    with pytest.raises(TypeError):
        recompute(t.dh, t.first, None, beta=torch.tensor(4.0))
    with pytest.raises(TypeError):
        recompute(t.dh, t.first, None, 4.0)
    assert launches == []


class SituDown(torch.autograd.Function):
    """The documentation's example: SiTU-GLU and the down projection, checkpointed by hand.  The forward keeps only
    ``gate_up`` and ``w_down``; the backward takes the down projection's input gradient first, then ``h`` and the
    gradients of ``gate_up`` from one fused call, then the weight gradient from that ``h``."""

    @staticmethod
    def forward(ctx, gate_up, w_down):
        h = forward(gate_up)
        ctx.save_for_backward(gate_up, w_down)  # h is not kept
        return torch.nn.functional.linear(h, w_down)

    @staticmethod
    def backward(ctx, dy):
        gate_up, w_down = ctx.saved_tensors
        dy2 = dy.reshape(-1, dy.shape[-1])
        dh = dy2 @ w_down  # 1. dgrad of down: needs no h
        h, d_gate_up, _ = recompute(dh, gate_up)  # 2. h + grads
        d_w_down = dy2.t() @ h  # 3. wgrad with the recomputed h
        return d_gate_up, d_w_down


def test_autograd_example_recompute_order(device) -> None:
    """The documentation's ``SituDown`` example saves only ``gate_up`` and ``w_down`` (no ``h``), gives the output and
    the ``gate_up`` gradient bits of the package's autograd function followed by the same down projection, and the
    weight gradient ``dy^T h`` of the forward's ``h``."""
    rows, width, hidden = 1031, 3072, 1024
    gen = torch.Generator(device=device).manual_seed(77)
    gate_up = (2.0 * torch.randn(rows, 2 * width, device=device, generator=gen)).bfloat16()
    w_down = (torch.randn(hidden, width, device=device, generator=gen) / width**0.5).bfloat16()
    dy = torch.randn(rows, hidden, device=device, generator=gen).bfloat16()

    saved = []

    def pack(x):
        saved.append((x.data_ptr(), tuple(x.shape), x.dtype))
        return x

    gu, w = gate_up.clone().requires_grad_(True), w_down.clone().requires_grad_(True)
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
        y = SituDown.apply(gu, w)
    assert sorted(saved) == sorted([(x.data_ptr(), tuple(x.shape), x.dtype) for x in (gu, w)])
    y.backward(dy)

    gu2, w2 = gate_up.clone().requires_grad_(True), w_down.clone().requires_grad_(True)
    y2 = torch.nn.functional.linear(situ_glu(gu2), w2)
    y2.backward(dy)
    assert_same_bits(y.detach(), y2.detach(), "output")
    assert_same_bits(gu.grad, gu2.grad, "d_gate_up")
    h_fwd = forward(gate_up)
    assert_same_bits(w.grad, dy.t() @ h_fwd, "d_w_down = dy^T h")


@pytest.mark.parametrize("layout", ["split", "merged"])
def test_double_backward_raises(layout, device) -> None:
    """Gradients computed with ``create_graph=True`` and an upstream gradient that requires grad have the steps' bits,
    and differentiating them raises ``RuntimeError``."""
    t = make_inputs(Case(33, 3072, layout), seed=3, device=device)
    want = _functional(t)
    leaves = _leaves(t)
    dh = t.dh.detach().clone().requires_grad_(True)
    wants = [leaf for leaf in leaves if leaf is not None]
    grads = torch.autograd.grad(situ_glu(*leaves), wants, dh, create_graph=True)
    names = ["d_first"] if t.second is None else ["d_first", "d_up"]
    for name, grad in zip(names, grads, strict=True):
        assert_same_bits(grad.detach(), want[name], f"{name} with create_graph=True")
    with pytest.raises(RuntimeError):
        grads[0].float().sum().backward()


def test_side_stream_and_no_host_sync(device) -> None:
    """On a side stream under ``torch.cuda.set_sync_debug_mode("error")``: the steps and an autograd step run without
    synchronizing and give the bits of the default stream."""
    t = make_inputs(Case(2049, 3072, "merged"), seed=9, device=device)
    want = step_outputs(forward, backward, t)
    want_autograd = _autograd_step(t)
    leaves = _leaves(t)
    stream = torch.cuda.Stream(device)
    stream.wait_stream(torch.cuda.current_stream(device))
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.cuda.stream(stream):
            got = step_outputs(forward, backward, t)
            h = situ_glu(*leaves)
            h.backward(t.dh)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream(device).wait_stream(stream)
    torch.cuda.synchronize(device)
    assert_same_outputs(got, want, "side stream")
    assert_same_bits(h.detach(), want_autograd["h"], "autograd h on a side stream")
    assert_same_bits(leaves[0].grad, want_autograd["d_first"], "autograd gradient on a side stream")


def _route_shapes(stage: str, arch: str, sm_count: int) -> list[tuple[int, int]]:
    """One ``(rows, width)`` per route of ``stage`` on ``arch`` that runs it: the route's least row count (at least
    one) at the low edge of its band, or one row at width 256 for the last route."""
    shapes = []
    for entry in cake_jit.ROUTES[stage][arch]:
        rows = max(1, entry["min_rows_per_sm"] * sm_count)
        width = entry["width"][0] if entry["width"] is not None else 256
        assert cake_jit.route(stage, arch, sm_count, rows, width) is entry, (stage, arch, entry)
        shapes.append((rows, width))
    return shapes


@pytest.mark.parametrize("stage", STAGES)
def test_no_host_sync_at_every_route(stage, device) -> None:
    """At a shape of each route of the stage on the device, a call on a side stream under
    ``torch.cuda.set_sync_debug_mode("error")`` neither synchronizes nor reads the device (after one warm-up call that
    loads the library), and gives the bits of the same call on the default stream."""
    arch, sm_count = _device_facts(device)
    for rows, width in _route_shapes(stage, arch, sm_count):
        t = make_inputs(Case(rows, width, "merged"), seed=rows % 61, device=device)
        want = _stage_outputs(stage, t)
        torch.cuda.synchronize(device)
        stream = torch.cuda.Stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        torch.cuda.set_sync_debug_mode("error")
        try:
            with torch.cuda.stream(stream):
                got = _stage_outputs(stage, t)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        assert_same_outputs(got, want, f"{stage} R {rows} I {width}")


def test_cuda_graph_capture_across_route_edges(device) -> None:
    """The forward, the backward and the fused backward at both sides of every rows-per-SM and width-band edge of the
    device (merged, preallocated outputs), each shape warmed up by one eager call, captured once in one graph and
    replayed twice on new inputs copied into the captured tensors: each replay gives the bits of eager calls on those
    inputs (the programs the capture chose run again)."""
    arch, sm_count = _device_facts(device)
    shapes = set()
    for stage in STAGES:
        for rows, width in _route_edges(stage, arch, sm_count):
            shapes |= {(rows - 1, width), (rows, width)}
        for rows, inside, outside in _width_edges(stage, arch, sm_count):
            shapes |= {(rows, inside), (rows, outside)}
    shapes = sorted(shapes)
    slots = []
    for rows, width in shapes:
        t = make_inputs(Case(rows, width, "merged"), seed=1, device=device)
        slot = types.SimpleNamespace(rows=rows, width=width, gu=t.gate_up.clone(), dh=t.dh.clone())
        slot.h = torch.empty(rows, width, dtype=torch.bfloat16, device=device)
        slot.d = torch.empty(rows, 2 * width, dtype=torch.bfloat16, device=device)
        slot.h2 = torch.empty_like(slot.h)
        slot.d2 = torch.empty_like(slot.d)
        slots.append(slot)

    def step():
        for slot in slots:
            forward(slot.gu, None, out=slot.h)
            backward(slot.dh, slot.gu, None, grad_gate_up_or_gate=slot.d)
            recompute(slot.dh, slot.gu, None, out=slot.h2, grad_gate_up_or_gate=slot.d2)

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        step()  # loads every library a shape routes to, before the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for k in range(2):
        news = []
        for slot in slots:
            n = make_inputs(Case(slot.rows, slot.width, "merged"), seed=100 + k, device=device)
            slot.gu.copy_(n.gate_up)
            slot.dh.copy_(n.dh)
            for buffer in (slot.h, slot.d, slot.h2, slot.d2):
                buffer.zero_()
            news.append(n)
        graph.replay()
        torch.cuda.synchronize(device)
        for slot, n in zip(slots, news, strict=True):
            want = _functional(n)
            what = f"replay {k}: R {slot.rows} I {slot.width}"
            assert_same_bits(slot.h, want["h"], f"{what}: h")
            assert_same_bits(slot.d, want["d_first"], f"{what}: d_gate_up")
            assert_same_bits(slot.h2, want["h"], f"{what}: fused h")
            assert_same_bits(slot.d2, want["d_first"], f"{what}: fused d_gate_up")


def test_cuda_graph_capture_and_replay(device) -> None:
    """A merged step at the model's scalars and a two-tensor step (with the fused backward) at another width and
    other scalars, with preallocated outputs, captured once after a warm-up call and replayed three times on new inputs
    copied into the captured tensors: each replay gives the bits of eager calls on those inputs."""
    R = 300
    a_case = Case(R, 3072, "merged")
    b_case = Case(R, 6144, "split", beta=2.5, lb=10.0)
    a = make_inputs(a_case, seed=40, device=device)
    b = make_inputs(b_case, seed=41, device=device)
    gu, dh_a = a.gate_up.clone(), a.dh.clone()
    gate, up, dh_b = b.gate.clone(), b.up.clone(), b.dh.clone()
    h_a = torch.empty(R, 3072, dtype=torch.bfloat16, device=device)
    d_gu = torch.empty(R, 2 * 3072, dtype=torch.bfloat16, device=device)
    h_b = torch.empty(R, 6144, dtype=torch.bfloat16, device=device)
    d_gate, d_up = torch.empty_like(h_b), torch.empty_like(h_b)

    h_c, d_gate_c, d_up_c = torch.empty_like(h_b), torch.empty_like(h_b), torch.empty_like(h_b)

    def step():
        forward(gu, None, out=h_a)
        backward(dh_a, gu, None, grad_gate_up_or_gate=d_gu)
        forward(gate, up, beta=b.beta, lb=b.lb, out=h_b)
        backward(dh_b, gate, up, beta=b.beta, lb=b.lb, grad_gate_up_or_gate=d_gate, grad_up=d_up)
        recompute(dh_b, gate, up, beta=b.beta, lb=b.lb, out=h_c, grad_gate_up_or_gate=d_gate_c, grad_up=d_up_c)

    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        step()  # loads the libraries before the capture
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for k in range(3):
        na = make_inputs(a_case, seed=50 + k, device=device)
        nb = make_inputs(b_case, seed=60 + k, device=device)
        gu.copy_(na.gate_up)
        dh_a.copy_(na.dh)
        gate.copy_(nb.gate)
        up.copy_(nb.up)
        dh_b.copy_(nb.dh)
        for buffer in (h_a, d_gu, h_b, d_gate, d_up, h_c, d_gate_c, d_up_c):
            buffer.zero_()
        graph.replay()
        torch.cuda.synchronize(device)
        want_a = _functional(na)
        want_b = _functional(nb)
        assert_same_bits(h_a, want_a["h"], f"replay {k}: h (merged)")
        assert_same_bits(d_gu, want_a["d_first"], f"replay {k}: d_gate_up")
        assert_same_bits(h_b, want_b["h"], f"replay {k}: h (two tensors)")
        assert_same_bits(d_gate, want_b["d_first"], f"replay {k}: d_gate")
        assert_same_bits(d_up, want_b["d_up"], f"replay {k}: d_up")
        assert_same_bits(h_c, want_b["h"], f"replay {k}: fused h")
        assert_same_bits(d_gate_c, want_b["d_first"], f"replay {k}: fused d_gate")
        assert_same_bits(d_up_c, want_b["d_up"], f"replay {k}: fused d_up")


@pytest.mark.parametrize("warn_only", [False, True], ids=["strict", "warn_only"])
def test_steps_neither_raise_nor_warn_under_deterministic_algorithms(warn_only, device) -> None:
    """Under ``torch.use_deterministic_algorithms(True)`` (strict and ``warn_only``) with warnings as errors, the
    steps, an autograd step and a non-reentrant checkpointed step raise nothing, warn nothing and give the bits of the
    same calls with the setting off."""
    t = make_inputs(Case(1031, 3072, "merged"), seed=35, device=device)
    u = make_inputs(Case(77, 6144, "split"), seed=36, device=device)

    def run():
        return [
            step_outputs(forward, backward, t),
            step_outputs(forward, backward, u),
            _stage_outputs("ckpt", t),
            _stage_outputs("ckpt", u),
            _autograd_step(t),
            _autograd_step(t, checkpoint=True),
        ]

    want = run()
    enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            got = run()
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=was_warn_only)
    for i, (a, b) in enumerate(zip(got, want, strict=True)):
        assert_same_outputs(a, b, f"warn_only={warn_only} call {i}")


# ----------------------------------------------------------------------------------------------------------------
# Layouts, buffers and refusals


def test_fit_inputs_are_read_in_place_and_others_copied_once(device, launches) -> None:
    """The halves of a merged input and row slices of wider buffers reach the kernels as views of the caller's memory
    (their row strides), with no copy; rows the kernels cannot read are copied once to contiguous rows; a single row's
    stride is passed as 0; the width, the FP32 scalars and the host's ``k1`` and ``k2`` are launch arguments."""
    m = make_inputs(Case(33, 3072, "merged"), seed=2, device=device)
    forward(m.first, None)
    stage, rows, width, values = launches[-1]
    assert stage == "fwd" and rows == 33 and width == 3072
    assert values["gate"].data_ptr() == m.gate_up.data_ptr()
    assert values["up"].data_ptr() == m.gate_up.data_ptr() + 2 * 3072
    assert values["gate_stride"] == values["up_stride"] == 2 * 3072
    assert values["width"] == 3072 and values["h_stride"] == 3072
    assert values["beta"] == 4.0 and values["lb"] == 25.0
    assert (values["k1"], values["k2"]) == pkg._scalars(4.0, 25.0)[2:]
    backward(m.dh, m.first, None)
    stage, _, _, values = launches[-1]
    assert stage == "bwd"
    assert values["gate"].data_ptr() == m.gate_up.data_ptr()
    assert values["dh"].data_ptr() == m.dh.data_ptr()
    assert values["d_up"].data_ptr() == values["d_gate"].data_ptr() + 2 * 3072
    assert values["d_gate_stride"] == values["d_up_stride"] == 2 * 3072

    s = make_inputs(Case(33, 6144, "sliced"), seed=2, device=device)
    backward(s.dh, s.gate, s.up, beta=2.5, lb=10.0)
    _, _, _, values = launches[-1]
    for name, tensor in (("gate", s.gate), ("up", s.up), ("dh", s.dh)):
        assert values[name].data_ptr() == tensor.data_ptr(), name
        assert values[f"{name}_stride"] == tensor.stride(0) != 6144, name
    assert values["width"] == 6144 and values["beta"] == 2.5 and values["lb"] == 10.0
    assert (values["k1"], values["k2"]) == pkg._scalars(2.5, 10.0)[2:]

    c = make_inputs(Case(33, 3072, "copied"), seed=2, device=device)
    backward(c.dh, c.gate, c.up, beta=7.3)
    _, _, _, values = launches[-1]
    for name, tensor in (("gate", c.gate), ("up", c.up), ("dh", c.dh)):
        assert values[name].data_ptr() != tensor.data_ptr(), name
        assert values[name].is_contiguous() and values[f"{name}_stride"] == 3072, name
    assert values["beta"] == fp32(7.3)

    one = make_inputs(Case(1, 3072, "merged"), seed=2, device=device)
    recompute(one.dh, one.first, None)
    _, _, _, values = launches[-1]
    for name in ("gate", "up", "dh", "d_gate", "d_up", "h"):
        assert values[f"{name}_stride"] == 0, name


def test_preallocated_outputs(device) -> None:
    """Preallocated outputs, contiguous or row slices of wider aligned buffers, receive the bits of new outputs and are
    returned; outputs the kernels cannot write in place are refused, never copied."""
    width = 3072
    t = make_inputs(Case(65, width, "merged"), seed=14, device=device)
    want = _functional(t)

    def wide(rows, cols, pad=8):
        return torch.empty(rows, cols + 2 * pad, dtype=torch.bfloat16, device=device)[:, pad : pad + cols]

    def new(rows, cols):
        return torch.empty(rows, cols, dtype=torch.bfloat16, device=device)

    for h, d_gu in ((new(t.R, width), new(t.R, 2 * width)), (wide(t.R, width), wide(t.R, 2 * width))):
        assert forward(t.first, None, out=h) is h
        result = backward(t.dh, t.first, None, grad_gate_up_or_gate=d_gu)
        assert result[0] is d_gu and result[1] is None
        assert_same_bits(h, want["h"], "out")
        assert_same_bits(d_gu, want["d_first"], "grad_gate_up_or_gate")

    s = make_inputs(Case(65, width, "split"), seed=15, device=device)
    want_s = _functional(s)
    d_gate, d_up = wide(s.R, width), wide(s.R, width)
    result = backward(s.dh, s.gate, s.up, grad_gate_up_or_gate=d_gate, grad_up=d_up)
    assert result[0] is d_gate and result[1] is d_up
    assert_same_bits(d_gate, want_s["d_first"], "grad_gate")
    assert_same_bits(d_up, want_s["d_up"], "grad_up")

    raw = torch.empty(t.R * 2 * width + 64, dtype=torch.bfloat16, device=device)
    unfit = {
        "an FP32 output": torch.empty(t.R, width, device=device),
        "another shape": torch.empty(t.R + 1, width, dtype=torch.bfloat16, device=device),
        "a base that is not 16-byte aligned": raw[1 : 1 + t.R * width].view(t.R, width),
        "a row stride that is not a multiple of 8": raw[: t.R * (width + 2)].view(t.R, width + 2)[:, :width],
        "overlapping rows": raw.as_strided((t.R, width), (8, 1)),
        "a non-unit last stride": raw[: t.R * width * 2].view(t.R, width, 2)[:, :, 0],
    }
    for what, out in unfit.items():
        with pytest.raises(ValueError):
            forward(t.first, None, out=out)
        with pytest.raises(ValueError):
            backward(s.dh, s.gate, s.up, grad_gate_up_or_gate=out)
        with pytest.raises(ValueError):
            backward(s.dh, s.gate, s.up, grad_up=out)
    with pytest.raises(ValueError, match="grad_up"):
        backward(t.dh, t.first, None, grad_up=torch.empty_like(t.dh))


def test_operand_and_scalar_errors(device, launches) -> None:
    """Wrong types, dtypes, shapes, widths and devices, and scalars that are not finite positive Python numbers or whose
    kernel constants are past FP32, are refused before any launch."""
    t = make_inputs(Case(8, 3072, "split"), seed=1, device=device)
    gate, up, dh = t.gate, t.up, t.dh
    merged = torch.cat([gate, up], dim=1)
    value_errors = {
        "an FP16 gate": lambda: forward(gate.half(), up),
        "an FP32 up": lambda: forward(gate, up.float()),
        "a width that is not a multiple of 256": lambda: forward(gate[:, :3000], up[:, :3000]),
        "a width below 256": lambda: forward(gate[:, :128], up[:, :128]),
        "a merged input of an odd number of columns": lambda: forward(merged[:, :6143]),
        "a merged input whose half is not a multiple of 256": lambda: forward(merged[:, :6000]),
        "an up of another width": lambda: forward(gate, up[:, :2816]),
        "another row count of up": lambda: forward(gate, up[:7]),
        "a 3-D gate": lambda: forward(gate[None], up[None]),
        "a 1-D merged input": lambda: forward(merged[0]),
        "up on the CPU": lambda: forward(gate, up.cpu()),
        "dh of another shape": lambda: backward(dh[:, :256], gate, up),
        "an FP32 dh": lambda: backward(dh.float(), gate, up),
        "dh on the CPU": lambda: backward(dh.cpu(), gate, up),
        "a NaN beta": lambda: forward(gate, up, beta=float("nan")),
        "an infinite lb": lambda: forward(gate, up, lb=float("inf")),
        "a zero beta": lambda: forward(gate, up, beta=0.0),
        "a negative lb": lambda: forward(gate, up, lb=-1.0),
        "a beta past FP32": lambda: forward(gate, up, beta=1e39),
        "an lb that rounds to 0 in FP32": lambda: forward(gate, up, lb=1e-50),
        "a beta whose k1 = -2 log2(e) / beta is past FP32": lambda: forward(gate, up, beta=1e-39),
        "an lb whose k2 = -2 log2(e) / lb is past FP32 in the backward": lambda: backward(dh, gate, up, lb=1e-39),
        "an lb whose k2 is past FP32 in the fused backward": lambda: recompute(dh, gate, up, lb=1e-39),
        "an FP32 dh in the fused backward": lambda: recompute(dh.float(), gate, up),
        "an up of another width in the fused backward": lambda: recompute(dh, gate, up[:, :2816]),
        "a beta lb product past FP32 in the autograd entry": lambda: situ_glu(gate, up, beta=1e20, lb=1e20),
        "a beta whose beta (1 + 2^-21) is past FP32": lambda: forward(gate, up, beta=FP32_MAX, lb=0.5),
        "an lb whose lb (1 + 2^-21) is past FP32 in the backward": lambda: backward(
            dh, gate, up, beta=0.5, lb=FP32_MAX
        ),
        "a negative beta in the backward": lambda: backward(dh, gate, up, beta=-4.0),
        "a negative lb in the autograd entry": lambda: situ_glu(gate, up, lb=-25.0),
    }
    type_errors = {
        "a tensor beta": lambda: forward(gate, up, beta=torch.tensor(4.0, device=device)),
        "a boolean lb": lambda: forward(gate, up, lb=True),
        "a string beta": lambda: forward(gate, up, beta="4"),
        "a list gate": lambda: forward([1.0], up),
        "a tensor lb in the autograd entry": lambda: situ_glu(gate, up, lb=torch.tensor(25.0)),
        "a tensor beta in the backward": lambda: backward(dh, gate, up, beta=torch.tensor(4.0)),
        "a tensor lb in the fused backward": lambda: recompute(dh, gate, up, lb=torch.tensor(25.0)),
        "a list dh in the fused backward": lambda: recompute([1.0], gate, up),
    }
    misses = []
    for kind, calls in ((ValueError, value_errors), (TypeError, type_errors)):
        for what, call in calls.items():
            try:
                call()
            except kind:
                continue
            except Exception as e:
                misses.append(f"{what}: {type(e).__name__} ({e}), not {kind.__name__}")
            else:
                misses.append(f"{what}: no {kind.__name__}")
    assert not misses, misses
    assert launches == []


def test_unsupported_devices_raise(device, monkeypatch) -> None:
    """A device of another architecture is a ``RuntimeError`` naming the delivered architectures, also with no rows,
    and is reported unsupported."""
    t = make_inputs(Case(5, 3072, "merged"), seed=1, device=device)
    monkeypatch.setattr(pkg, "_arch_of_index", lambda index: None)
    assert not pkg.is_cake_situ_glu_det_train_supported(device)
    for rows in (0, 5):
        with pytest.raises(RuntimeError, match="sm_100a"):
            forward(t.first[:rows], None)
        with pytest.raises(RuntimeError, match="sm_100a"):
            backward(t.dh[:rows], t.first[:rows], None)
        with pytest.raises(RuntimeError, match="sm_100a"):
            recompute(t.dh[:rows], t.first[:rows], None)


def test_operands_on_another_device(device) -> None:
    """Operands on a second GPU run there and give the bits of the first; mixed devices are refused."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two GPUs")
    other = torch.device("cuda", 1 if device.index != 1 else 0)
    if not pkg.is_cake_situ_glu_det_train_supported(other):
        pytest.skip(f"{other} is not a supported device")
    t = make_inputs(Case(129, 3072, "merged"), seed=6, device=device)
    want = step_outputs(forward, backward, t)
    o = make_inputs(Case(129, 3072, "merged"), seed=6, device=other)
    got = step_outputs(forward, backward, o)
    for name, value in got.items():
        assert value.device == other, name
        assert_same_bits(value.to(device), want[name], f"{name} on {other}")
    with pytest.raises(ValueError):
        backward(t.dh, o.first, None)
