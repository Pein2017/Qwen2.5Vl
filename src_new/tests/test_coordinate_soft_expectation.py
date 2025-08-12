import torch
import torch.nn.functional as F

from src_new.models.coordinate_loss import SoftExpectationCoordinateLoss


def _make_loss(coord_start: int, coord_count: int) -> SoftExpectationCoordinateLoss:
    # coord_end_id passed to ctor is exclusive
    return SoftExpectationCoordinateLoss(
        coord_start_id=coord_start,
        coord_end_id=coord_start + coord_count,
        temperature=1.0,
        numerical_stability=True,
        device=torch.device("cpu"),
    )


def test_coord_value_mapping_and_validation():
    coord_start = 151667
    coord_count = 16  # small for test
    loss = _make_loss(coord_start, coord_count)

    # Validate coord_values tensor
    values = loss._get_coord_values(torch.device("cpu"))
    assert values.shape[0] == coord_count
    # Should be [0, 1, ..., coord_count-1]
    assert torch.allclose(values, torch.arange(coord_count, dtype=torch.float32))

    # Validate mapping function on a few samples
    for k in [0, 5, coord_count - 1]:
        token_id = coord_start + k
        assert loss._extract_coord_value_from_token_id(token_id) == k

    # Full validator should pass
    loss._validate_coordinate_mapping()


def test_soft_expectation_monotonicity_and_float_output():
    coord_start = 100
    coord_count = 10  # values 0..9
    loss = _make_loss(coord_start, coord_count)

    # Target coordinate value (scalar)
    target = torch.tensor([5.0])

    # Build logits for 1 token over coord vocab
    logits_close = torch.full((1, coord_count), -5.0)
    logits_far = torch.full((1, coord_count), -5.0)

    # Concentrate mass near 5
    logits_close[0, 5] = 5.0
    logits_close[0, 4] = 3.0
    logits_close[0, 6] = 3.0

    # Concentrate mass far from 5, e.g., near 0
    logits_far[0, 0] = 5.0
    logits_far[0, 1] = 3.0

    exp_close = loss.compute_soft_expectation(logits_close)
    exp_far = loss.compute_soft_expectation(logits_far)

    # Expectation should be a float and closer to 5 for logits_close
    assert exp_close.dtype == torch.float32
    assert abs(exp_close.item() - 5.0) < abs(exp_far.item() - 5.0)

    l1_close = F.l1_loss(exp_close, target)
    l1_far = F.l1_loss(exp_far, target)
    assert l1_close.item() < l1_far.item()


def test_gradient_flows_only_to_coordinate_slice_and_positions():
    torch.manual_seed(0)
    coord_start = 10
    coord_count = 8  # coordinate dims 10..17 inclusive in vocab space
    vocab_size = 32

    loss = _make_loss(coord_start, coord_count)

    # logits: [batch=1, seq_len=3, vocab_size]
    logits = torch.randn(1, 3, vocab_size, requires_grad=True)

    # labels: choose coordinate token at position [0,1], ignore others
    labels = torch.full((1, 3), -100, dtype=torch.long)
    target_coord_value = 6  # within 0..7
    labels[0, 1] = coord_start + target_coord_value

    # coord_mask: true only at [0,1]
    coord_mask = torch.zeros((1, 3), dtype=torch.bool)
    coord_mask[0, 1] = True

    coord_loss, _ = loss.compute_coordinate_loss(logits, labels, coord_mask)
    coord_loss.backward()

    # Gradients should exist only at the coordinate slice for the masked position
    grad = logits.grad  # shape [1,3,32]

    # Positions other than [0,1,:] should have zero gradients
    assert torch.allclose(grad[0, 0, :], torch.zeros_like(grad[0, 0, :]))
    assert torch.allclose(grad[0, 2, :], torch.zeros_like(grad[0, 2, :]))

    # Within [0,1,:], non-coordinate dims should have zero gradients
    non_coord_left = slice(0, coord_start)
    non_coord_right = slice(coord_start + coord_count, vocab_size)
    assert torch.allclose(
        grad[0, 1, non_coord_left], torch.zeros_like(grad[0, 1, non_coord_left])
    )
    assert torch.allclose(
        grad[0, 1, non_coord_right], torch.zeros_like(grad[0, 1, non_coord_right])
    )

    # Coordinate dims should receive non-zero gradients (at least one)
    coord_slice_grad = grad[0, 1, coord_start : coord_start + coord_count]
    assert (coord_slice_grad.abs() > 0).any(), "No gradient flowed to coordinate logits"
