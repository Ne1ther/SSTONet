"""Contract tests for frozen Trunk reuse; all fixtures are small CPU graphs."""

import io

import pytest
import torch
from torch import nn

from sstonet.inference import StaleTrunkCacheError, precompute_trunk
from sstonet.models import ComponentAwareGNNDeepONet, GNNDeepONet, PureGNN


MODEL_CLASSES = (GNNDeepONet, ComponentAwareGNNDeepONet)


def make_inputs(model_class=GNNDeepONet, *, weighted=True, dtype=torch.float32):
    torch.manual_seed(2026)
    model = model_class(
        branch_input_dim=4,
        branch_hidden=[12, 8],
        trunk_hidden=[12, 8],
        output_dim=6,
        num_gnn_layers=2,
        dropout=0.25,
    ).to(dtype=dtype).eval()
    coords = torch.randn(7, 3, dtype=dtype)
    edges = torch.tensor([[0, 1, 2, 3, 4, 5, 1, 2, 3], [1, 2, 3, 4, 5, 6, 0, 1, 2]])
    weights = torch.linspace(0.2, 1.2, edges.shape[1], dtype=dtype) if weighted else None
    return model, coords, edges, weights


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("weighted", (False, True))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_matches_forward_for_single_and_multiple_conditions(model_class, weighted, dtype):
    model, coords, edges, weights = make_inputs(model_class, weighted=weighted, dtype=dtype)
    cache = precompute_trunk(model, coords, edges, weights)
    for batch in (1, 3, 16):
        inputs = torch.randn(batch, 4, dtype=dtype, requires_grad=True)
        with torch.no_grad():
            expected = model(inputs, coords, edges, weights)
        actual = cache.predict(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.shape == (batch, coords.shape[0])
        assert not actual.requires_grad
    assert cache.shape == (7, 6)
    assert cache.nbytes == 7 * 6 * coords.element_size()


def test_predict_does_not_execute_trunk_again(monkeypatch):
    model, coords, edges, weights = make_inputs()
    cache = precompute_trunk(model, coords, edges, weights)

    def forbid_trunk(*args, **kwargs):
        raise AssertionError("Cached prediction recomputed Trunk")

    monkeypatch.setattr(model.trunk, "forward", forbid_trunk)
    assert cache.predict(torch.randn(2, 4)).shape == (2, 7)


def test_embeddings_export_is_not_a_mutable_alias():
    model, coords, edges, weights = make_inputs()
    cache = precompute_trunk(model, coords, edges, weights)
    inputs = torch.randn(2, 4)
    expected = cache.predict(inputs)
    exported = cache.embeddings
    exported.zero_()
    torch.testing.assert_close(cache.predict(inputs), expected, rtol=0, atol=0)
    assert not exported.requires_grad
    with pytest.raises(AttributeError):
        cache.embeddings = exported


@pytest.mark.parametrize("target", ("coords", "edges", "weights"))
def test_bound_input_inplace_edits_invalidate(target):
    model, coords, edges, weights = make_inputs()
    cache = precompute_trunk(model, coords, edges, weights)
    tensors = {"coords": coords, "edges": edges, "weights": weights}
    # Editing a view also increments the shared version counter.
    tensor = tensors[target]
    tensor.view(-1)[0].add_(1)
    with pytest.raises(StaleTrunkCacheError, match="changed"):
        cache.predict(torch.randn(1, 4))


@pytest.mark.parametrize("target", ("trunk", "branch", "bias"))
def test_every_model_parameter_region_is_bound(target):
    model, coords, edges, weights = make_inputs()
    cache = precompute_trunk(model, coords, edges, weights)
    parameter = model.bias if target == "bias" else next(getattr(model, target).parameters())
    with torch.no_grad():
        parameter.add_(0.01)
    with pytest.raises(StaleTrunkCacheError, match="parameter"):
        cache.predict(torch.randn(1, 4))


@pytest.mark.parametrize("change", ("dtype", "device", "parameter", "module", "buffer"))
def test_model_structural_or_storage_changes_invalidate(change):
    model, coords, edges, weights = make_inputs()
    model.register_buffer("test_marker", torch.zeros(1))
    cache = precompute_trunk(model, coords, edges, weights)
    if change == "dtype":
        model.double()
    elif change == "device":
        model.to("meta")
    elif change == "parameter":
        model.bias = nn.Parameter(model.bias.detach().clone())
    elif change == "module":
        model.trunk.activation = nn.Tanh().eval()
    else:
        model.test_marker.add_(1)
    with pytest.raises(StaleTrunkCacheError):
        cache.predict(torch.randn(1, 4))


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
@pytest.mark.parametrize("target", ("edges", "weights"))
def test_rebuild_clears_legacy_gcn_normalization_after_inplace_graph_edit(model_class, target):
    model, coords, edges, weights = make_inputs(model_class)
    inputs = torch.randn(3, 4)
    cache = precompute_trunk(model, coords, edges, weights)
    original = cache.predict(inputs)
    if target == "edges":
        edges[1, 0] = 6  # Changes degree normalization while retaining storage pointer.
    else:
        weights.mul_(2)
    with pytest.raises(StaleTrunkCacheError):
        cache.predict(inputs)
    rebuilt = precompute_trunk(model, coords, edges, weights)
    fresh_model, _, _, _ = make_inputs(model_class)
    fresh_model.load_state_dict(model.state_dict())
    with torch.no_grad():
        expected = fresh_model(inputs, coords, edges, weights)
        direct = model(inputs, coords, edges, weights)
    assert not torch.allclose(expected, original)
    torch.testing.assert_close(rebuilt.predict(inputs), expected, rtol=0, atol=0)
    torch.testing.assert_close(direct, expected, rtol=0, atol=0)


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_state_dict_checkpoint_and_training_remain_compatible(model_class):
    model, coords, edges, weights = make_inputs(model_class)
    state_before = {name: value.clone() for name, value in model.state_dict().items()}
    flags_before = [parameter.requires_grad for parameter in model.parameters()]
    with torch.inference_mode():
        cache = precompute_trunk(model, coords, edges, weights)
        cache.predict(torch.randn(1, 4))
    assert tuple(model.state_dict()) == tuple(state_before)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[name], rtol=0, atol=0)
    assert [parameter.requires_grad for parameter in model.parameters()] == flags_before
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    other, _, _, _ = make_inputs(model_class)
    other.load_state_dict(saved, strict=True)
    model.load_state_dict(saved, strict=True)
    with pytest.raises(StaleTrunkCacheError):
        cache.predict(torch.randn(1, 4))
    rebuilt = precompute_trunk(model, coords, edges, weights)
    assert rebuilt.predict(torch.randn(1, 4)).shape == (1, 7)
    model.train()
    # Building inside inference_mode must not poison transient norm caches for training.
    model(torch.randn(2, 4), coords, edges, weights).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_training_mode_is_rejected_without_changing_mode_or_gradient_flags():
    model, coords, edges, weights = make_inputs()
    model.train()
    with pytest.raises(ValueError, match="eval"):
        precompute_trunk(model, coords, edges, weights)
    assert model.training
    assert all(parameter.requires_grad for parameter in model.parameters())
    model.eval()
    cache = precompute_trunk(model, coords, edges, weights)
    model.branch.train()  # Reject mixed-mode models too.
    with pytest.raises(StaleTrunkCacheError, match="eval"):
        cache.predict(torch.randn(1, 4))
    with pytest.raises(ValueError, match="eval"):
        precompute_trunk(model, coords, edges, weights)


def test_inference_tensor_inputs_are_rejected_but_normal_inputs_in_context_work():
    model, coords, edges, weights = make_inputs()
    with torch.inference_mode():
        inference_coords = coords.clone()
    with pytest.raises(ValueError, match="version counters"):
        precompute_trunk(model, inference_coords, edges, weights)
    with torch.inference_mode():
        cache = precompute_trunk(model, coords, edges, weights)
    assert cache.predict(torch.randn(1, 4)).shape == (1, 7)


def test_autocast_requires_explicit_full_precision():
    model, coords, edges, weights = make_inputs()
    cache = precompute_trunk(model, coords, edges, weights)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(ValueError, match="autocast"):
            precompute_trunk(model, coords, edges, weights)
        with pytest.raises(ValueError, match="autocast"):
            cache.predict(torch.randn(1, 4))


def test_invalid_inputs_and_unsupported_model_are_rejected():
    model, coords, edges, weights = make_inputs()
    with pytest.raises(TypeError, match="supports"):
        precompute_trunk(PureGNN(param_dim=4).eval(), coords, edges, weights)
    with pytest.raises(ValueError, match="edge_index"):
        precompute_trunk(model, coords, edges.float(), weights)
    with pytest.raises(ValueError, match="edge_weight"):
        precompute_trunk(model, coords, edges, weights[:-1])
    with pytest.raises(ValueError, match="dtype"):
        precompute_trunk(model, coords.double(), edges, weights)
    cache = precompute_trunk(model, coords, edges, weights)
    with pytest.raises(ValueError, match="shape"):
        cache.predict(torch.randn(4))
    with pytest.raises(ValueError, match="dtype"):
        cache.predict(torch.randn(1, 4, dtype=torch.float64))
