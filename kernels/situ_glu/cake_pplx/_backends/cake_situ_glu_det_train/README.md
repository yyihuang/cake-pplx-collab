# Deterministic SiTU-GLU training backend (SM100 / SM103 / SM107)

Backend of `cake_pplx.cake_situ_glu_det_train`: the SiTU-GLU activation of an MLP in training, forward, backward and
the fused backward of a checkpointed step, at any width `I` that is a positive multiple of 256 (the widths 3072, 6144
and 33,792 among them).  The math is FP32
on the BF16 inputs, with one BF16 rounding of each output.  The activation is unweighted.

The facade `cake_pplx.cake_situ_glu_det_train` is a stable API.  This backend package is experimental, and its internal
names may change.  The facade imports the backend on first use.  The supported architectures are SM100 (B200 / GB200),
SM103 (B300 / GB300) and SM107 (Rubin).  Another device raises `RuntimeError`.  The package loads its libraries prebuilt
from `aot/` when a builder compiled them.  Otherwise the package JIT-builds them from the sources and flags that
`build.json` records.  No environment variable changes a call; every control is an argument.  A JIT build (no prebuilt
`aot/` library) reads the JIT variables listed in the repository README, section JIT build and cache.

## Usage

```python
import torch
import torch.utils.checkpoint
from cake_pplx.cake_situ_glu_det_train import cake_situ_glu_det

# gate_up: BF16 [R, 2I] (gate columns [0, I), up [I, 2I)), the output of the gate/up projection
h = cake_situ_glu_det(gate_up)  # BF16 [R, I], the down projection's input; beta = 4.0, lb = 25.0
h.backward(dh)  # gate_up.grad: BF16 [R, 2I]

# two tensors, and other scalars
h = cake_situ_glu_det(gate, up, beta=3.0, lb=40.0)

# under activation checkpointing: only the inputs are kept
h = torch.utils.checkpoint.checkpoint(cake_situ_glu_det, gate_up, use_reentrant=False)
```

The steps without autograd, `cake_situ_glu_det_train_forward` and `cake_situ_glu_det_train_backward`, take the same
operands and optional preallocated outputs (`out=`, `grad_gate_up_or_gate=`, `grad_up=`), for a caller that fuses the
activation into its own `torch.autograd.Function`.  `cake_situ_glu_det_train_backward_recompute` is the fused
backward of a checkpointed step: `(h, d_gate_up, None)` or `(h, d_gate, d_up)` in one launch, `h` with the forward's
bits and the gradients with the backward's (the API documentation's `SituDown` example keeps only `gate_up` and the
down projection's weight, and takes `h` back from this call).  `beta` and `lb` are keywords everywhere, with the
defaults 4.0 and 25.0.  They must be positive and finite in FP32, with the FP32 quotients `k1 = -2 log2(e) / beta`
and `k2 = -2 log2(e) / lb`, `beta lb`, `beta (1 + 2^-21)` and `lb (1 + 2^-21)` finite in FP32 (`ValueError`
otherwise); the tests cover `beta` in `[2.5, 4]` and `lb` in `[10, 40]`.

## Inputs and outputs

| operand | shape | dtype | notes |
|---|---|---|---|
| `gate`, `up` | `[R, I]` each | BF16 | or one `[R, 2I]` tensor (first argument, `up=None`); any row stride |
| `beta`, `lb` | Python numbers | | finite and positive, rounded to FP32; launch arguments |
| `h` | `[R, I]` | BF16 | new and contiguous, or the caller's `out` |
| `dh` | `[R, I]` | BF16 | upstream gradient |
| `d_gate`, `d_up` | `[R, I]` each | BF16 | or one `[R, 2I]` gradient for a merged input |

The fused backward takes `gate`, `up` and `dh` and stores `h`, `d_gate` and `d_up`.

- Every operand is on one CUDA device.  `R` is any count from 0 to `2**30`.  `I` is a positive multiple of 256.
- The kernels read every BF16 tensor through its row stride in elements: a row stride that is a multiple of 8, a unit
  last stride and a 16-byte aligned base are read in place (the halves of a merged tensor, row slices of a wider
  tensor).  Another input layout is copied once.
- A caller's output buffer must already have such a layout, with rows that do not overlap (row stride at least the
  row width); otherwise it is refused (`ValueError`).  An output (`out`, `grad_gate_up_or_gate`, `grad_up`) must not
  overlap any input or another output; this is not checked.
- A wrong type, or a `beta` or `lb` that is not a Python number, raises `TypeError` (a tensor would need a host read).
  A wrong device, dtype or shape, a width that is not a positive multiple of 256, a non-finite or non-positive `beta`
  or `lb`, or an unfit output raises `ValueError`.  An unsupported device raises `RuntimeError`, also with `R = 0`.
  Every argument is checked before an input is copied or an output allocated.
- `R = 0` returns empty outputs and launches nothing.

## Programs

Eight programs (`cake_jit.MODULES`, keyed by program name) serve three stages: `fwd`, `bwd` and `ckpt` (the fused
backward).  Each is compiled with the exact `-gencode` flag of each architecture it is delivered for (`build.json`:
four libraries on `sm_100a`, four on `sm_103a`, five on `sm_107a`).  Each is one kernel.  A row is `I / 256`
segments of 256 columns, and a warp owns a segment: lane `l` loads 8 BF16 values at column `256 s + 8 l`, so each load
of a warp is contiguous.  A CTA has 128 threads (four warps).  Four warps own one row (two warps a row in the
two-rows-a-CTA program) and take the row's segments in turn.  Each warp issues the loads of its segments before it
computes and stores them; a segment past the row's last one reloads the last segment and stores nothing, and a row past
the last one stores nothing.  The CTAs stride over the blocks of rows, so a smaller grid runs every row.  Row offsets
are 64-bit.  No program uses shared memory, a workspace, a cooperative launch or programmatic dependent launch.  The
row count, the width, `beta`, `lb` and the strides are launch arguments, so one library per program and architecture
serves every call.

The programs of a stage differ in their schedule only, and store the same bits for every input:

| program | stage | architectures | schedule |
|---|---|---|---|
| `cake_situ_glu_det_train_e5d3075ea15fefbb9d43` | `fwd` | `sm_100a`, `sm_103a` | the host's `k1` and `k2` |
| `cake_situ_glu_det_train_9b922273e948685fab2d` | `fwd` | `sm_107a` | the 0.2.1 forward |
| `cake_situ_glu_det_train_599ccd99cf9f7a41dae9` | `fwd` | `sm_107a` | the next row's loads issued before a row's stores |
| `cake_situ_glu_det_train_8ba54d18b9b06905b89e` | `bwd` | `sm_100a`, `sm_103a` | the host's `k1` and `k2` |
| `cake_situ_glu_det_train_2ad79222570bbd0c2cef` | `bwd` | `sm_107a` | the 0.2.1 backward |
| `cake_situ_glu_det_train_1819a1f2c9babb2a7303` | `bwd` | `sm_107a` | the host's `k1` and `k2`, built for 8 CTAs per SM |
| `cake_situ_glu_det_train_0f6427ea51531187fe83` | `ckpt` | all three | one row per CTA |
| `cake_situ_glu_det_train_0dff177a39167613b5b1` | `ckpt` | `sm_100a`, `sm_103a` | two warps a row, two rows per CTA |

Caption: the programs; `k1 = -2 log2(e) / beta` and `k2 = -2 log2(e) / lb` are FP32 divisions, the values the other
programs form on the device.

`cake_jit.ROUTES` picks the program of a call: per stage and architecture, the first route that holds runs.  A route
holds when `R >= rows_per_sm * SMs` (the device's SM count, read once per device) and, when it states a width band,
`lo <= I <= hi` (inclusive).  The grid is one CTA per block of the program's `rows_per_cta` rows, capped at
`ctas_per_sm * SMs` when the route states a cap (`cake_jit.route`, `cake_jit.grid`).

| stage | arch | routes, in order (rows per SM, width band, CTAs per SM) |
|---|---|---|
| `fwd` | `sm_100a`, `sm_103a` | `e5d3075e...`, every `R` and `I` |
| `fwd` | `sm_107a` | `599ccd99...` if `R >= 67 SMs` and `2048 <= I <= 3072`, cap 32; `9b922273...` if `3072 <= I <= 6144`, cap 16; `9b922273...` uncapped |
| `bwd` | `sm_100a`, `sm_103a` | `8ba54d18...`, every `R` and `I` |
| `bwd` | `sm_107a` | `1819a1f2...` if `R >= 154 SMs` and `3072 <= I <= 33792`, cap 96; the same if `R >= 19 SMs`, same band, cap 48; `2ad79222...` uncapped |
| `ckpt` | `sm_100a` | `0dff177a...` if `R >= 83 SMs` and `3072 <= I <= 5120`; `0f6427ea...` |
| `ckpt` | `sm_103a` | `0dff177a...` if `R >= 80 SMs` and `3072 <= I <= 5120`; `0f6427ea...` |
| `ckpt` | `sm_107a` | `0f6427ea...` if `3072 <= I <= 5120`, cap 32; `0f6427ea...` uncapped |

Caption: the routes, measured on B200 (148 SMs), GB300 (152) and VR200 (212): a conditional route runs where its
launch measured faster than the next route's.  The windows are in rows per SM, so they scale with the SM count.

## Numerics

- The BF16 values are widened to FP32 (exact).  With `E = 2^(-2 |g| log2(e) / beta)`, `P = 2^(-|g| log2(e))` and
  `W = 2^(-2 |u| log2(e) / lb)`, all in `[0, 1]` so that no exponential overflows: `tanh(|g| / beta) = (1 - E) /
  (1 + E)`, `sigmoid(g) = 1 / (1 + P)` for `g >= 0` and `P / (1 + P)` for `g < 0`, and `1 - tanh^2 = 4 E / (1 + E)^2`
  (no cancellation); the up takes the same forms with `W`.
- The approximate exponential is within 2 ulps of the rounded result, so `1 - E` and `1 - W` taken from `E` and `W`
  alone keep only an absolute error of a few `2^-24`, and `E` (`W`) rounds to 1 below about `|g| = 2^-26 beta`
  (`|u| = 2^-26 lb`).  Neither stage takes them that way.
- The forward takes `beta (1 - E)` as the smaller of two upper bounds, without a branch.  With `y = 2 |g| / beta`,
  the first is `|g| (1 + E)`, above the exact value by under `y^2 / 12` of it (`tanh(z) <= z`); the second is
  `beta (1 + 2^-21) - beta E`, above it by more than the error of `E`.  The smaller is within about `2^-14.8` of the
  exact value, relative to it, for every `g`, and `g = 0` gives 0 exactly.  `lb (1 - W)` takes the same form with
  `|u|` and `lb`.  So `h` keeps its accuracy for tiny gates and ups, subnormal ones included, and `g = 0` or `u = 0`
  gives `h = 0` exactly.
- The backward takes `beta (1 - E)` as the larger of two lower bounds, without a branch.  With `y = 2 |g| / beta`,
  the first is `|g| (2 - 2 |g| / beta) = beta (y - y^2 / 2)`, below the exact value by under `beta y^3 / 6`; the
  second is `beta (1 - 2^-20) - beta E`, below it by more than the error of `E`.  The larger is within `2^-13` of the
  exact value, relative to it, for every `g`, and `g = 0` gives 0 exactly.  `lb (1 - W)` takes the same form with
  `|u|` and `lb`.  So `d_gate` and `d_up` keep their accuracy for tiny gates and ups, subnormal ones included; the
  tests judge every output with `|g|` down to `2^-60 beta` and `|u|` down to `2^-60 lb`.
- The bounds of both stages hold in exact arithmetic.  The second bound of each stage also holds in FP32: its margin
  covers the error of `E` and the roundings.  In FP32 the first bound moves by up to a few `2^-24` of the exact value,
  to either side, and the relative errors above include this.
- `beta = 4` takes `P = E E`; any other `beta` takes `P` from its own exponential.  The choice is the same for every
  element of a call.
- The exponentials are the approximate base-2 exponential.  `E` and `W` flush below `2^-126`, only where
  `|g| > 43.67 beta` or `|u| > 43.67 lb`.  `h` takes them only as `1 + E`, `beta (1 + 2^-21) - beta E` and the same
  terms of `W`, which the flush leaves as they are.  In the backward, an up past `43.67 lb` has `d_up` under
  `2^-124 |dh| beta`, and at `beta >= 2.4` a gate past `43.67 beta` has `d_gate` under about `2^-124 |dh| lb`; the
  kernels store 0 there.  At a smaller `beta`, the sigmoid term `beta P` of such a `d_gate` does not flush, and the
  kernels store it.  `P` keeps its subnormal values, so a gate below about -87 keeps a nonzero sigmoid where FP32 holds
  one.
- The reciprocal `R` of `(1 + E)(1 + P)(1 + W)` (a product in `[1, 8]`): the forward takes one per element; the backward takes
  one per pair of elements (of their product, in `[1, 64]`, multiplied back by the other element's factor) and writes
  both gradients with `R^2`, so it needs no other reciprocal.  A negative gate's sigmoid numerator `P` is the last
  factor of each product, so a subnormal result rounds once.
- `h`, `d_gate` and `d_up` are each rounded once to BF16.  Every product and FMA rounds to nearest without flushing
  subnormals; pairs of elements use the packed FP32 instructions, which round each half as the scalar instruction
  does.  Nothing is FP64.

## Determinism

Every output is elementwise, so every call gives the same bits for the same inputs on the same architecture, whatever
the row count, the program a route picks, the width's place in a wider tensor, the layout (merged, two tensors, row
slices), the stream or a CUDA graph replay.  The fused backward stores the forward's `h` and the backward's gradients.  No call alerts under `torch.use_deterministic_algorithms(True)`; with
`torch.utils.deterministic.fill_uninitialized_memory` (on by default), PyTorch fills each new output and each copied
input with NaN before the kernel writes it, one more launch each, and the bits do not change.  No entry point takes a
`deterministic` keyword.  The family has no non-deterministic module: there is no reduction across columns, rows or
CTAs to reorder.  It has no KCP module: rows are independent, with no sequence dimension.

## Workspace and retained state

- Workspace: none.
- Retained state of the autograd function: its inputs only (`gate` and `up` or the merged tensor), references with no
  copy: `R * 2I * 2` bytes of tensors the caller already holds.  No activation is saved; the backward recomputes it.

## Limitations

- No double backward (`create_graph`); not tested under `torch.compile` or `torch.jit.trace`.  No FP8 or FP4 output.
- A program's library builds at its first launch, and the program of a call depends on `R` and `I`: before capturing
  a CUDA graph, run one eager call at each `(R, I)` and stage the graph captures (or build the libraries with
  `specs_for_arch`).  A replay runs the programs the capture chose.
