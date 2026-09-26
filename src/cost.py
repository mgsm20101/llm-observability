"""Pure cost calculations — no I/O, fully deterministic.

Prices are supplied by the caller (see `Settings.price_input_per_mtok` /
`price_output_per_mtok` in `config.py`), not hardcoded here. The default is
0.0/0.0 — local, self-hosted inference, which is what this project actually
runs and actually measures. Any non-zero price turns the same token counts
into a projection of what the traffic would cost against some other,
illustrative rate; it is never a measurement of what this run cost.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestCost:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    model: str


def calc_cost(
    input_tokens: int,
    output_tokens: int,
    price_input_per_mtok: float,
    price_output_per_mtok: float,
    model_name: str = "unknown",
) -> RequestCost:
    """Calculate USD cost for a single LLM request.

    Args:
        input_tokens: Number of prompt tokens.
        output_tokens: Number of completion tokens.
        price_input_per_mtok: Input price in USD per 1M tokens.
        price_output_per_mtok: Output price in USD per 1M tokens.
        model_name: Display name for the model.

    Returns:
        RequestCost with breakdown.
    """
    cost = (input_tokens / 1_000_000) * price_input_per_mtok \
         + (output_tokens / 1_000_000) * price_output_per_mtok

    return RequestCost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=cost,
        model=model_name,
    )
