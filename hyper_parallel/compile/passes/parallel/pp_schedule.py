# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
PP Schedules - Graph-mode pipeline schedule drivers.

Single module per the house convention (``core/pipeline_parallel/scheduler.py``
keeps its eager-mode schedules the same way): shared P2P / microbatch
machinery in :class:`PipelineScheduleBase` plus one class per algorithm —
:class:`ScheduleGPipe` and :class:`Schedule1F1B` today — dispatched by name
from :data:`SCHEDULE_REGISTRY` / ``PassConfig.pp_schedule``.

Both schedules run the per-stage FX subgraphs produced by ``PpPass`` across
microbatches and share the same P2P / loss / gradient contract:

- Model inputs are routed by dataflow: each user input is owned by the single
  stage that consumes it (resolved by ``PpPass``) and is fed to that stage's
  forward subgraph directly — every rank receives the full flattened
  user-input list, slices its OWNED tensor inputs per microbatch, and ignores
  the rest. All tensor user inputs must carry the leading batch dim.
- Boundary values are exchanged as ORDERED LISTS: tensors go as-is, int
  scalars (dynamic-shape ``sym_size`` nodes crossing the cut) are packed as
  0-d int64 tensors and unwrapped with ``item()`` on arrival — mirroring how
  ``torch.distributed.pipelining``'s ``PipelineStage`` ships full argument
  lists. The P2P exchange uses eager ``dist.isend``/``irecv`` on the PP
  process group; sends are async (``Work`` handles waited at the end of the
  step) so the sweeps overlap receive/compute across stages, receives wait
  before first use.
- Microbatch gradients are un-normalized sums while sweeping, then divided by
  ``num_microbatches`` at the end, so the step gradient matches the non-PP
  semantics of a full-batch mean loss.

The schedules differ only in their action order:

``ScheduleGPipe``
    1. Forward sweep: run every microbatch's forward (shipping boundary
       activations downstream), holding all forward outputs.
    2. Backward sweep (reverse order): seed ``ones_like(loss)`` on the last
       stage, run each backward against the activations saved by the forward
       sweep, and pass boundary gradients one stage back.

``Schedule1F1B``
    Warmup forwards (``min(num_microbatches, pp_degree - stage_idx)``), then
    steady-state 1B1F (one backward followed by one forward) until all
    forwards are consumed, then a backward cooldown. Interleaving backward
    earlier shrinks the pipeline bubble relative to GPipe. Requires
    ``num_microbatches >= pp_degree``.
"""

__all__ = [
    "PipelineScheduleBase",
    "ScheduleGPipe",
    "Schedule1F1B",
    "SCHEDULE_REGISTRY",
    "get_schedule_class",
]

from typing import Any, Dict, List, Sequence, Tuple, Type

import torch
import torch.distributed as dist
from torch import nn


class PipelineScheduleBase(nn.Module):
    """Shared P2P / microbatch machinery for the graph-mode PP schedules.

    Installed by ``PpPass`` as a submodule of the compiled GraphModule and
    invoked through a ``call_module`` node, so the trainer's
    ``graph_module(*flat_inputs)`` dispatches to the concrete schedule
    unchanged. Subclasses implement :meth:`forward` (the action order) from
    the per-microbatch building blocks defined here.

    Args:
        fwd_gm: Stage forward subgraph. Signature per stage:
            stage 0    -> ``(*state, *owned_inputs)`` returning
                          ``(*act_out, *saved)``;
            other      -> ``(*state, *act_in, *owned_inputs)`` returning
                          ``(*act_out, *saved)`` for non-last stages and
                          ``(loss, *saved)`` for the last one, where
                          ``owned_inputs`` are the user inputs routed to
                          this stage (may be empty).
        bwd_gm: Stage backward subgraph. Signature:
            ``(*state, *grad_in, *fwd_outs)`` where ``grad_in`` are the
            received boundary gradients (``ones_like(loss)`` on the last
            stage) and ``fwd_outs`` are the forward subgraph's outputs
            verbatim. Returns ``(*param_grads,)`` plus the boundary
            gradients for every stage except stage 0.
        stage_idx: This rank's stage index (0-based).
        pp_degree: Number of pipeline stages.
        num_state: Number of leading state tensors (stage params/buffers).
        num_trainable: Number of trainable parameters on this stage; the
            backward subgraph returns exactly this many parameter gradients
            first.
        num_send: Number of forward boundary values to ship downstream
            (the forward outputs' prefix); 0 on the last stage.
        grad_send_count: Number of backward boundary values to ship
            upstream (the backward outputs' suffix); 0 on stage 0.
        microbatch_size: Samples per microbatch (leading dim).
        pp_group: PP process group object used for isend/irecv.
        recv_spec: Per-value descriptors for the incoming forward
            boundary: ``("tensor", shape, dtype, device)`` or
            ``("scalar",)``. Empty on stage 0.
        grad_recv_spec: Same, for the incoming backward boundary
            gradients. Empty on the last stage.
        user_input_stages: Per flattened user input (stub arg order), the
            stage that consumes it. This stage slices and feeds the inputs
            mapped to itself and ignores the rest.
    """

    def __init__(
        self,
        fwd_gm: nn.Module,
        bwd_gm: nn.Module,
        stage_idx: int,
        pp_degree: int,
        num_state: int,
        num_trainable: int,
        num_send: int,
        grad_send_count: int,
        microbatch_size: int,
        pp_group: Any = None,
        recv_spec: Sequence[Tuple[Any, ...]] = (),
        grad_recv_spec: Sequence[Tuple[Any, ...]] = (),
        user_input_stages: Sequence[int] = (),
    ) -> None:
        """Store the stage subgraphs, P2P group, and boundary specs."""
        super().__init__()
        self.fwd_gm = fwd_gm
        self.bwd_gm = bwd_gm
        self.stage_idx = stage_idx
        self.pp_degree = pp_degree
        self.num_state = num_state
        self.num_trainable = num_trainable
        self.num_send = num_send
        self.grad_send_count = grad_send_count
        self.microbatch_size = microbatch_size
        self.pp_group = pp_group
        self.recv_spec = list(recv_spec)
        self.grad_recv_spec = list(grad_recv_spec)
        self.user_input_stages = list(user_input_stages)
        self.is_first = stage_idx == 0
        self.is_last = stage_idx == pp_degree - 1
        # Work handles of in-flight isend ops; tensors are kept referenced
        # so the underlying buffers stay alive until the send completes.
        self._pending_sends: List[Tuple[Any, torch.Tensor]] = []

    # ------------------------------------------------------------------
    # Input / microbatch resolution
    # ------------------------------------------------------------------

    def _split_inputs(
        self, flat_inputs: Sequence[Any]
    ) -> Tuple[Sequence[Any], List[Any]]:
        """Split ``(*state, *user_inputs)`` and validate the input arity."""
        state = flat_inputs[: self.num_state]
        user_inputs = list(flat_inputs[self.num_state :])
        if len(user_inputs) != len(self.user_input_stages):
            raise RuntimeError(
                f"PP schedule expects {len(self.user_input_stages)} user "
                f"inputs (per its routing) but received {len(user_inputs)}"
            )
        return state, user_inputs

    def _resolve_microbatches(
        self, user_inputs: Sequence[Any]
    ) -> Tuple[int, List[Any]]:
        """Derive the microbatch count and this stage's owned user inputs.

        Every rank receives every user input, so the batch size is derived
        identically on all stages from the first tensor input.
        """
        batch_size = next(
            (v.shape[0] for v in user_inputs if isinstance(v, torch.Tensor)), None
        )
        if batch_size is None:
            raise ValueError(
                "PP requires at least one tensor model input to derive the "
                "microbatch split"
            )
        if batch_size % self.microbatch_size != 0:
            raise ValueError(
                f"PP microbatch mismatch: batch size {batch_size} is not "
                f"divisible by pp_microbatch_size {self.microbatch_size}"
            )
        num_microbatches = batch_size // self.microbatch_size
        owned_inputs = [
            v
            for v, s in zip(user_inputs, self.user_input_stages)
            if s == self.stage_idx
        ]
        return num_microbatches, owned_inputs

    def _slice_inputs(self, owned_inputs: Sequence[Any], mb_index: int) -> List[Any]:
        """Slice this stage's owned inputs for microbatch ``mb_index``.

        Tensors are sliced along the leading batch dim; non-tensors pass
        through unchanged.
        """
        lo = mb_index * self.microbatch_size
        hi = lo + self.microbatch_size
        return [v[lo:hi] if isinstance(v, torch.Tensor) else v for v in owned_inputs]

    # ------------------------------------------------------------------
    # Per-microbatch building blocks
    # ------------------------------------------------------------------

    def _forward_microbatch(
        self, state: Sequence[Any], owned_inputs: Sequence[Any], mb_index: int
    ) -> Tuple[Any, ...]:
        """Run one microbatch's forward subgraph, receiving activations first.

        ``owned_inputs`` are this stage's user inputs (see
        ``user_input_stages``): tensors are sliced per microbatch along the
        leading batch dim, non-tensors pass through. They are appended after
        the received activations in the fwd subgraph's arg order.
        """
        owned_mb = self._slice_inputs(owned_inputs, mb_index)
        if self.is_first:
            out = self.fwd_gm(*state, *owned_mb)
        else:
            # Forward activations arrive from the PREVIOUS stage.
            act_in = self._recv_values(self.recv_spec, src=self.stage_idx - 1)
            out = self.fwd_gm(*state, *act_in, *owned_mb)
        return tuple(out)

    def _send_forward(self, out: Sequence[Any]) -> None:
        """Ship the boundary activations (forward outputs' prefix) downstream."""
        if self.num_send:
            self._send_values(out[: self.num_send], dst=self.stage_idx + 1)

    def _backward_microbatch(
        self, state: Sequence[Any], fwd_outs: Sequence[Any]
    ) -> Sequence[torch.Tensor]:
        """Run one microbatch's backward subgraph, returning param grads.

        Seeds ``ones_like(loss)`` on the last stage; otherwise receives the
        boundary gradients from the NEXT stage (they flow upstream during the
        backward sweep). Ships this stage's boundary gradients to the previous
        stage. The traced backward graph seeds its own loss gradient
        internally, so the seed on the last stage is a placeholder — the grads
        that come back are un-normalized sums over the microbatches.
        """
        if self.is_last:
            # Seed shape only; the traced graph ignores the value.
            grad_in: List[Any] = [torch.ones_like(fwd_outs[0])]
        else:
            grad_in = self._recv_values(self.grad_recv_spec, src=self.stage_idx + 1)
        out = self.bwd_gm(*state, *grad_in, *fwd_outs)

        param_grads = out[: self.num_trainable]
        if self.grad_send_count:
            # Gradients flow UPSTREAM: the previous stage is the destination,
            # mirroring the forward direction.
            self._send_values(out[self.num_trainable :], dst=self.stage_idx - 1)
        return param_grads

    @staticmethod
    def _accumulate_grads(
        accumulated: Sequence[torch.Tensor], param_grads: Sequence[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Sum one microbatch's param grads into the running accumulation."""
        if not accumulated:
            return list(param_grads)
        return [acc + grad for acc, grad in zip(accumulated, param_grads)]

    def _finalize_loss(
        self, fwd_outs_per_mb: Sequence[Sequence[Any]], user_inputs: Sequence[Any]
    ) -> torch.Tensor:
        """Return the step loss (last stage) or a zero placeholder."""
        if self.is_last:
            # On the last stage each forward output tuple starts with the
            # per-microbatch loss.
            losses = [outs[0] for outs in fwd_outs_per_mb]
            return torch.stack(losses).mean()
        return torch.zeros((), device=self._anchor_device(user_inputs))

    @staticmethod
    def _average_grads(
        grads: Sequence[torch.Tensor], num_microbatches: int
    ) -> List[torch.Tensor]:
        """Turn un-normalized sums over microbatches into the mean."""
        return [grad / num_microbatches for grad in grads]

    # ------------------------------------------------------------------
    # P2P helpers
    # ------------------------------------------------------------------

    def _flush_pending_sends(self) -> None:
        """Wait for in-flight isend ops and drop their buffer references."""
        for work, _ in self._pending_sends:
            work.wait()
        self._pending_sends = []

    @staticmethod
    def _anchor_device(user_inputs: Sequence[Any]) -> torch.device:
        """Device for the zero-loss placeholder: first tensor user input."""
        for value in user_inputs:
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device("cpu")

    def _send_values(self, values: Sequence[Any], dst: int) -> None:
        """Async-send boundary values to the neighbouring stage ``dst``.

        Tensors go as-is (made contiguous, with THAT buffer retained); int
        scalars (dynamic-shape ``sym_size`` nodes) are packed as 0-d int64
        tensors on the first tensor value's device — CPU tensors cannot ride
        an NCCL/NPU-backend group — and unwrapped with ``item()`` on the
        receiving side. The ``Work`` handles are queued (waited at the end of
        the step) and the sent tensors kept referenced so their buffers
        outlive the send.
        """
        dst_rank = self._global_rank(dst)
        anchor_device = next(
            (v.device for v in values if isinstance(v, torch.Tensor)), None
        )
        for value in values:
            if isinstance(value, torch.Tensor):
                tensor = value.contiguous()
            else:
                tensor = torch.tensor(value, dtype=torch.int64, device=anchor_device)
            work = dist.isend(tensor, dst=dst_rank, group=self.pp_group)
            self._pending_sends.append((work, tensor))

    def _recv_values(self, spec: Sequence[Tuple[Any, ...]], src: int) -> List[Any]:
        """Receive boundary values from the neighbouring stage ``src``, in order.

        The buffer for each entry comes from ``spec`` (built by ``PpPass``
        from the boundary values' fake-tensor ``val`` metas, including the
        device each value was traced on); scalar entries arrive as 0-d
        int64 tensors and are unwrapped to Python ints so the graph sees
        the same types it was traced with. Scalar buffers are anchored to
        the first tensor entry's device (``PpPass._value_spec`` guarantees
        a non-empty list carries at least one tensor).
        """
        src_rank = self._global_rank(src)
        anchor_device = next((entry[3] for entry in spec if entry[0] == "tensor"), None)
        values: List[Any] = []
        for entry in spec:
            if entry[0] == "scalar":
                buffer = torch.empty((), dtype=torch.int64, device=anchor_device)
            else:
                _, shape, dtype, device = entry
                buffer = torch.empty(shape, dtype=dtype, device=device)
            work = dist.irecv(buffer, src=src_rank, group=self.pp_group)
            work.wait()
            values.append(int(buffer.item()) if entry[0] == "scalar" else buffer)
        return values

    def _global_rank(self, stage_idx: int) -> int:
        """Map a stage index to its global rank within the PP group."""
        if self.pp_group is None:
            return stage_idx
        return dist.get_global_rank(self.pp_group, stage_idx)


class ScheduleGPipe(PipelineScheduleBase):
    """GPipe driver over two per-stage FX subgraphs (forward / backward).

    Runs ALL forwards before ANY backward, so every microbatch's forward
    outputs (boundary activations + saved values) are held until the backward
    sweep completes — activation memory scales with ``num_microbatches``.
    Larger ``microbatch_size`` trades pipeline bubble for fewer P2P
    round-trips and fewer retained activations.
    """

    def forward(self, *flat_inputs: Any) -> Tuple[torch.Tensor, ...]:
        """Run one full GPipe step (fwd sweep then bwd sweep).

        Args:
            flat_inputs: ``(*state, *user_inputs)`` — the same surface
                ``run_traced_graph`` feeds the compiled graph, with
                ``user_inputs`` the flattened model inputs in trace order.

        Returns:
            ``(loss, *param_grads)``. On non-last stages the loss is a zero
            scalar placeholder (the real loss lives on the last stage);
            parameter gradients are the microbatch-averaged accumulation.
        """
        state, user_inputs = self._split_inputs(flat_inputs)
        num_microbatches, owned_inputs = self._resolve_microbatches(user_inputs)

        fwd_outs_per_mb = self._forward_sweep(state, owned_inputs, num_microbatches)
        grads = self._backward_sweep(state, fwd_outs_per_mb, num_microbatches)

        self._flush_pending_sends()

        loss = self._finalize_loss(fwd_outs_per_mb, user_inputs)
        return (loss, *grads)

    def _forward_sweep(
        self,
        state: Sequence[Any],
        owned_inputs: Sequence[Any],
        num_microbatches: int,
    ) -> List[Tuple[Any, ...]]:
        """Run forward microbatch 0..N-1, shipping boundary values downstream.

        Returns per-microbatch forward output tuples (replayed verbatim
        into the backward subgraph by ``_backward_sweep``).
        """
        fwd_outs_per_mb: List[Tuple[Any, ...]] = []
        for mb_index in range(num_microbatches):
            out = self._forward_microbatch(state, owned_inputs, mb_index)
            fwd_outs_per_mb.append(out)
            self._send_forward(out)
        return fwd_outs_per_mb

    def _backward_sweep(
        self,
        state: Sequence[Any],
        fwd_outs_per_mb: Sequence[Tuple[Any, ...]],
        num_microbatches: int,
    ) -> List[torch.Tensor]:
        """Run backward microbatches N-1..0, accumulating full-batch gradients.

        The traced backward graph seeds its own loss gradient internally
        (an ``ones_like`` over the saved loss), so the grads that come back
        are un-normalized sums over the microbatches. Dividing the
        accumulation by ``num_microbatches`` turns that into the mean, which
        is exactly the full-batch mean-loss gradient the non-PP path
        produces.
        """
        grads: List[torch.Tensor] = []
        for mb_index in reversed(range(num_microbatches)):
            grads = self._accumulate_grads(
                grads,
                self._backward_microbatch(state, fwd_outs_per_mb[mb_index]),
            )
        return self._average_grads(grads, num_microbatches)


class Schedule1F1B(PipelineScheduleBase):
    """1F1B driver over two per-stage FX subgraphs (forward / backward).

    Warmup runs ``min(num_microbatches, pp_degree - stage_idx)`` forwards,
    the steady state alternates one backward then one forward, and the
    cooldown drains the remaining backwards. Running each backward as soon as
    its forward output is available shrinks the pipeline bubble relative to
    GPipe's fill-drain order.

    Requires ``num_microbatches >= pp_degree`` (checked at runtime), the same
    constraint ``torch.distributed.pipelining.Schedule1F1B`` enforces.

    Note:
        Like ``ScheduleGPipe``, in-flight sends are waited at the end of the
        step, so boundary-activation buffers stay referenced until then; this
        class fixes the action order, not yet the activation lifetime.
    """

    def forward(self, *flat_inputs: Any) -> Tuple[torch.Tensor, ...]:
        """Run one full 1F1B step (warmup -> steady 1B1F -> cooldown).

        Args:
            flat_inputs: ``(*state, *user_inputs)`` — the same surface
                ``run_traced_graph`` feeds the compiled graph, with
                ``user_inputs`` the flattened model inputs in trace order.

        Returns:
            ``(loss, *param_grads)``. On non-last stages the loss is a zero
            scalar placeholder (the real loss lives on the last stage);
            parameter gradients are the microbatch-averaged accumulation.

        Raises:
            ValueError: When ``num_microbatches < pp_degree`` (the schedule
                cannot fill the pipeline).
        """
        state, user_inputs = self._split_inputs(flat_inputs)
        num_microbatches, owned_inputs = self._resolve_microbatches(user_inputs)
        if num_microbatches < self.pp_degree:
            raise ValueError(
                f"1F1B requires at least pp_degree={self.pp_degree} "
                f"microbatches, got {num_microbatches} "
                f"(batch {num_microbatches * self.microbatch_size} with "
                f"pp_microbatch_size={self.microbatch_size}); lower "
                f"pp_microbatch_size or use the gpipe schedule"
            )

        fwd_outs_per_mb: List[Any] = [None] * num_microbatches
        grads: List[torch.Tensor] = []
        fwd_index = 0
        bwd_index = 0

        # Warmup: the last stage has 1 forward, the stage before it 2, ...
        warmup = min(num_microbatches, self.pp_degree - self.stage_idx)
        for _ in range(warmup):
            out = self._forward_microbatch(state, owned_inputs, fwd_index)
            fwd_outs_per_mb[fwd_index] = out
            self._send_forward(out)
            fwd_index += 1

        # Steady state + cooldown: one backward, then one forward while any
        # forward remains.
        while bwd_index < num_microbatches:
            grads = self._accumulate_grads(
                grads,
                self._backward_microbatch(state, fwd_outs_per_mb[bwd_index]),
            )
            bwd_index += 1
            if fwd_index < num_microbatches:
                out = self._forward_microbatch(state, owned_inputs, fwd_index)
                fwd_outs_per_mb[fwd_index] = out
                self._send_forward(out)
                fwd_index += 1

        self._flush_pending_sends()

        loss = self._finalize_loss(fwd_outs_per_mb, user_inputs)
        return (loss, *self._average_grads(grads, num_microbatches))


SCHEDULE_REGISTRY: Dict[str, Type[PipelineScheduleBase]] = {
    "gpipe": ScheduleGPipe,
    "1f1b": Schedule1F1B,
}


def get_schedule_class(name: str) -> Type[PipelineScheduleBase]:
    """Resolve a ``PassConfig.pp_schedule`` name to its schedule class.

    Args:
        name: Schedule name, case-insensitive (``"gpipe"`` / ``"1f1b"``).

    Returns:
        The schedule class.

    Raises:
        ValueError: When ``name`` is not a registered schedule.
    """
    key = str(name).lower()
    schedule_cls = SCHEDULE_REGISTRY.get(key)
    if schedule_cls is None:
        raise ValueError(
            f"Unknown pp_schedule {name!r}; expected one of {sorted(SCHEDULE_REGISTRY)}"
        )
    return schedule_cls
