"""Cost computation — src/cost.py. Pure functions, no I/O, no mocking needed."""
import pytest

from src.cost import calc_cost


def test_local_inference_is_free_by_default():
    cost = calc_cost(
        input_tokens=1000,
        output_tokens=500,
        price_input_per_mtok=0.0,
        price_output_per_mtok=0.0,
        model_name="gemma3:4b",
    )

    assert cost.cost_usd == 0.0
    assert cost.total_tokens == 1500
    assert cost.model == "gemma3:4b"


def test_cost_is_price_times_tokens_over_a_million():
    cost = calc_cost(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        price_input_per_mtok=0.15,
        price_output_per_mtok=0.60,
    )

    assert cost.cost_usd == pytest.approx(0.75)


def test_zero_tokens_produce_zero_cost():
    cost = calc_cost(
        input_tokens=0,
        output_tokens=0,
        price_input_per_mtok=3.0,
        price_output_per_mtok=15.0,
    )

    assert cost.cost_usd == 0.0
    assert cost.total_tokens == 0
