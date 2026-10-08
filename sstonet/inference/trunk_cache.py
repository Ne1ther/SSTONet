"""Reuse a fixed graph's Trunk representation across operating conditions.

This external wrapper does not register buffers, freeze parameters, or change a
model's ``state_dict``. Call ``model.eval()`` explicitly before building a cache.
Inputs, graph topology/weights, and every model parameter/buffer are bound by
identity, version counter, and tensor metadata. Ordinary in-place edits, loading
weights, device/dtype moves, and module replacement require a new cache.

Checks read metadata only, without hashing or copying GPU tensor contents. As a
result, unsupported writes through ``.data``, NumPy aliases, or external storage
can bypass invalidation. Bound tensors must have PyTorch version counters; tensors
created in ``torch.inference_mode()`` must first be cloned outside that context.
Concurrent model mutation, hooks/monkey-patched forwards, and arbitrary changes to
unregistered Python attributes are unsupported. Autocast must be disabled for both
cache construction and use so that the cached precision has an explicit contract.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

from sstonet.models.deeponet import ComponentAwareGNNDeepONet, GCNLayer, GNNDeepONet


SupportedModel = Union[GNNDeepONet, ComponentAwareGNNDeepONet]


class StaleTrunkCacheError(RuntimeError):
    """The cached representation no longer matches its bound model or graph."""


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    """Read metadata only; ``_version`` detects normal aliased in-place edits."""
    try:
        version = tensor._version
    except RuntimeError as exc:
        raise ValueError(
            "Bound tensors need version counters. Clone inference tensors outside "
            "torch.inference_mode() before precompute_trunk()."
        ) from exc
    return (
        version,
        tensor.device,
        tensor.dtype,
        tensor.layout,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(),
    )


@dataclass(frozen=True)
class _TensorBinding:
    name: str
    tensor: torch.Tensor
    signature: tuple

    @classmethod
    def capture(cls, name: str, tensor: torch.Tensor) -> "_TensorBinding":
        return cls(name, tensor, _tensor_signature(tensor))

    def check(self, tensor: torch.Tensor) -> None:
        if tensor is not self.tensor or _tensor_signature(tensor) != self.signature:
            raise StaleTrunkCacheError(
                f"{self.name} changed after Trunk precomputation; rebuild with precompute_trunk()."
            )


def _named_state(model: nn.Module) -> Tuple[Tuple[str, torch.Tensor], ...]:
    return tuple(("parameter " + name, tensor) for name, tensor in model.named_parameters()) + tuple(
        ("buffer " + name, tensor) for name, tensor in model.named_buffers()
    )


def _require_eval(model: nn.Module, *, stale: bool = False) -> None:
    if any(module.training for module in model.modules()):
        error = StaleTrunkCacheError if stale else ValueError
        raise error("Trunk caching requires model.eval() and all submodules in evaluation mode.")


def _require_full_precision(device: torch.device) -> None:
    if torch.is_autocast_enabled(device.type):
        raise ValueError("Disable autocast when constructing or using a Trunk cache.")


class TrunkCache:
    """Inference-only, externally held cache for one frozen model and fixed graph.

    Build with :func:`precompute_trunk`. ``predict`` includes inexpensive Python
    validity checks in its latency; it never reruns the Trunk. The model's
    ``requires_grad`` flags remain unchanged, but all predictions disable autograd.
    The public ``embeddings`` property returns a detached copy to prevent callers
    from corrupting the internal representation. Use ``shape`` and ``nbytes`` for
    metadata without allocating that copy.
    """

    def __init__(
        self,
        model: SupportedModel,
        embeddings: torch.Tensor,
        inputs: Tuple[_TensorBinding, ...],
    ) -> None:
        self._model = model
        self._embeddings = embeddings
        self._inputs = inputs
        self._modules = tuple((name, id(module)) for name, module in model.named_modules())
        self._state = tuple(_TensorBinding.capture(name, tensor) for name, tensor in _named_state(model))

    @property
    def shape(self) -> torch.Size:
        return self._embeddings.shape

    @property
    def nbytes(self) -> int:
        return self._embeddings.numel() * self._embeddings.element_size()

    @property
    def embeddings(self) -> torch.Tensor:
        """A detached copy of the spatial representation, never an internal alias."""
        self._validate()
        return self._embeddings.detach().clone()

    def _validate(self) -> None:
        _require_eval(self._model, stale=True)
        modules = tuple((name, id(module)) for name, module in self._model.named_modules())
        if modules != self._modules:
            raise StaleTrunkCacheError("Model modules changed; rebuild with precompute_trunk().")
        state = _named_state(self._model)
        if tuple(name for name, _ in state) != tuple(binding.name for binding in self._state):
            raise StaleTrunkCacheError("Model state changed; rebuild with precompute_trunk().")
        for binding, (_, tensor) in zip(self._state, state):
            binding.check(tensor)
        for binding in self._inputs:
            binding.check(binding.tensor)

    @torch.inference_mode()
    def predict(self, branch_input: torch.Tensor) -> torch.Tensor:
        """Predict ``(batch, nodes)`` from normalized ``(batch, branch_dim)`` inputs."""
        self._validate()
        _require_full_precision(self._embeddings.device)
        if branch_input.ndim != 2:
            raise ValueError("branch_input must have shape (batch, branch_input_dim).")
        if (
            branch_input.device != self._embeddings.device
            or branch_input.dtype != self._embeddings.dtype
        ):
            raise ValueError("branch_input must have the same device and dtype as the cached model.")
        branch_out = self._model.branch(branch_input)
        return torch.matmul(branch_out, self._embeddings.T) + self._model.bias


@torch.no_grad()
def precompute_trunk(
    model: SupportedModel,
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
) -> TrunkCache:
    """Compute a reusable representation for a fixed, already built graph.

    ``coords`` are exactly the normalized spatial inputs used by the trained
    model, ``edge_index`` has shape ``(2, E)`` and dtype ``torch.long``, and the
    optional ``edge_weight`` has shape ``(E,)``. Graph construction and input
    normalization remain the caller's responsibilities. Changing geometry,
    topology, edge weights, checkpoint, device, or dtype requires a new cache.

    GCN normalization caches are cleared before computing the Trunk. Their legacy
    pointer-based key does not detect an in-place graph/weight edit; clearing here
    makes explicit rebuilding safe without changing the underlying model class.
    This precomputation includes that normalization cost.
    """
    if type(model) not in (GNNDeepONet, ComponentAwareGNNDeepONet):
        raise TypeError("Trunk caching supports GNNDeepONet and ComponentAwareGNNDeepONet only.")
    _require_eval(model)
    _require_full_precision(coords.device)
    if coords.ndim != 2:
        raise ValueError("coords must have shape (nodes, trunk_input_dim).")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise ValueError("edge_index must have shape (2, edges) and dtype torch.long.")
    if edge_weight is not None and (
        edge_weight.ndim != 1 or edge_weight.shape[0] != edge_index.shape[1]
    ):
        raise ValueError("edge_weight must have shape (edges,).")
    reference = next(model.parameters())
    if coords.device != reference.device or coords.dtype != reference.dtype:
        raise ValueError("coords must have the same device and dtype as the model.")
    if edge_index.device != coords.device or (
        edge_weight is not None and edge_weight.device != coords.device
    ):
        raise ValueError("coords, edge_index, and edge_weight must be on the same device.")
    inputs = [_TensorBinding.capture("coords", coords), _TensorBinding.capture("edge_index", edge_index)]
    if edge_weight is not None:
        inputs.append(_TensorBinding.capture("edge_weight", edge_weight))
    # Check model version counters before running any work or clearing norm caches.
    for name, tensor in _named_state(model):
        _TensorBinding.capture(name, tensor)
    for module in model.trunk.modules():
        if isinstance(module, GCNLayer):
            module._cached_norm = None
            module._cached_norm_key = None
    # Even if the caller is in inference_mode, keep the model's transient GCN
    # normalization tensors ordinary tensors so subsequent training still works.
    with torch.inference_mode(False), torch.no_grad():
        embeddings = model.trunk(coords, edge_index, edge_weight).detach()
    return TrunkCache(model, embeddings, tuple(inputs))
