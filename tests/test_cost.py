"""Cost computation — src/cost.py. Pure functions, no I/O, no mocking needed."""
import pytest

from src.cost import calc_cost, monthly_cost_estimate


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


def test_cost_per_1k_is_zero_when_no_tokens_were_used():
    cost = calc_cost(
        input_tokens=0,
        output_tokens=0,
        price_input_per_mtok=3.0,
        price_output_per_mtok=15.0,
    )

    assert cost.cost_per_1k == 0.0


def test_cost_per_1k_scales_the_total_cost():
    cost = calc_cost(
        input_tokens=1_000_000,
        output_tokens=0,
        price_input_per_mtok=2.0,
        price_output_per_mtok=0.0,
    )

    # $2.00 for 1,000,000 tokens → $0.002 per 1,000 tokens
    assert cost.cost_per_1k == pytest.approx(0.002)


def test_monthly_cost_estimate_projects_daily_cost_times_thirty():
    estimate = monthly_cost_estimate(cost_per_request=0.002, requests_per_day=100)

    assert estimate["daily_usd"] == pytest.approx(0.2)
    assert estimate["monthly_usd"] == pytest.approx(6.0)
    assert estimate["requests_per_day"] == 100
