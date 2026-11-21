"""Exact, offline cost of a query run, per endpoint, before any request is sent.

`build_plan` is pure: it only looks at the phase, the answers already stored and (optionally) the
server-side budget figures. The CLI prints it for `plan` and requires `--yes` on top of it for
every `ep-*` run, so nobody spends budget without having seen the exact numbers.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hackathon.model_equality.client import BudgetPolicy, QueryJob
from hackathon.model_equality.prompts import PromptRepeat, bank_by_id
from hackathon.model_equality.store import Answer, AnswerKey

REFERENCE_PREFIX = "ref-"
SCORED_PREFIX = "ep-"
NUM_SCORED_ENDPOINTS = 33
NUM_REFERENCE_ENDPOINTS = 9
SCORED_ENDPOINTS: tuple[str, ...] = tuple(
    f"{SCORED_PREFIX}{index:02d}" for index in range(1, NUM_SCORED_ENDPOINTS + 1)
)
REFERENCE_ENDPOINTS: tuple[str, ...] = tuple(
    f"{REFERENCE_PREFIX}{index:02d}" for index in range(1, NUM_REFERENCE_ENDPOINTS + 1)
)


@dataclass(frozen=True)
class EndpointCost:
    """Cost of a plan on one endpoint.

    Attributes:
        endpoint: Endpoint id.
        stored: Answers of this endpoint already in the store.
        wanted: (prompt, repeat) units of the phase not yet stored.
        to_request: Requests the plan will really send (wanted, truncated to the allowance).
        truncated: Wanted units dropped because they would eat into the reserve.
        total_after: Stored answers after the run.
        left_after: Answers left on the endpoint after the run, relative to the policy limit
            (None for unlimited endpoints).
    """

    endpoint: str
    stored: int
    wanted: int
    to_request: int
    truncated: int
    total_after: int
    left_after: int | None


@dataclass
class QueryPlan:
    """Jobs to send plus the per-endpoint accounting."""

    jobs: list[QueryJob]
    costs: list[EndpointCost]
    unlimited: bool
    drift: list[str] = field(default_factory=list)

    @property
    def total_requests(self) -> int:
        """Requests the plan will send, all endpoints together."""
        return len(self.jobs)

    def caps(self) -> dict[str, int] | None:
        """Per-endpoint new-answer caps for the runner (None for unlimited endpoints)."""
        if self.unlimited:
            return None
        return {cost.endpoint: cost.to_request for cost in self.costs}


def detect_prompt_drift(stored: Mapping[AnswerKey, Answer]) -> list[str]:
    """Stored answers whose prompt text differs from the bank's text for the same id.

    A drift means the bank was edited after answers were collected: those answers would be
    compared with answers to a different question. Ids outside the bank are ignored.
    """
    bank = bank_by_id()
    drifted = sorted({
        answer.prompt_id for answer in stored.values()
        if answer.prompt_id in bank and answer.prompt and answer.prompt != bank[answer.prompt_id].text
    })
    return [f"prompt {prompt_id}: stored text differs from the bank" for prompt_id in drifted]


def build_plan(
    units: Sequence[PromptRepeat],
    endpoints: Sequence[str],
    stored: Mapping[AnswerKey, Answer],
    policy: BudgetPolicy | None,
    server_left: Mapping[str, int] | None = None,
) -> QueryPlan:
    """Compute the jobs and exact costs of asking `units` to every endpoint.

    Args:
        units: Ordered (prompt, repeat) units (see `expand_phase`).
        endpoints: Endpoints to query, in round-robin order.
        stored: Answers already stored (never asked again).
        policy: Budget policy for limited endpoints, or None for unlimited ones (`ref-*`).
        server_left: Answers left on the server per endpoint, when known (online planning).

    Returns:
        The plan; with a policy, units beyond an endpoint's allowance are counted as truncated
        and never become jobs.
    """
    stored_per_endpoint: Counter[str] = Counter(key[0] for key in stored)
    allowance: dict[str, int | None] = {}
    for endpoint in endpoints:
        if policy is None:
            allowance[endpoint] = None
        else:
            left = None if server_left is None else server_left.get(endpoint)
            allowance[endpoint] = policy.allowance(stored_per_endpoint[endpoint], left)
    planned: Counter[str] = Counter()
    wanted: Counter[str] = Counter()
    truncated: Counter[str] = Counter()
    jobs: list[QueryJob] = []
    for unit in units:
        for endpoint in endpoints:
            if (endpoint, unit.prompt.id, unit.repeat) in stored:
                continue
            wanted[endpoint] += 1
            cap = allowance[endpoint]
            if cap is not None and planned[endpoint] >= cap:
                truncated[endpoint] += 1
                continue
            planned[endpoint] += 1
            jobs.append(QueryJob(endpoint, unit.prompt.id, unit.repeat, unit.prompt.text,
                                 unit.prompt.max_tokens))
    costs = [
        EndpointCost(
            endpoint=endpoint,
            stored=stored_per_endpoint[endpoint],
            wanted=wanted[endpoint],
            to_request=planned[endpoint],
            truncated=truncated[endpoint],
            total_after=stored_per_endpoint[endpoint] + planned[endpoint],
            left_after=(None if policy is None
                        else policy.limit - stored_per_endpoint[endpoint] - planned[endpoint]),
        )
        for endpoint in endpoints
    ]
    return QueryPlan(jobs=jobs, costs=costs, unlimited=policy is None,
                     drift=detect_prompt_drift(stored))


def format_plan(plan: QueryPlan, policy: BudgetPolicy | None) -> str:
    """Render the plan as a fixed-width table plus totals (what `plan` prints)."""
    header = f"{'endpoint':<9}{'stored':>7}{'wanted':>7}{'request':>8}{'cut':>5}{'after':>7}{'left':>6}"
    lines = [header]
    for cost in plan.costs:
        left = "-" if cost.left_after is None else str(cost.left_after)
        lines.append(f"{cost.endpoint:<9}{cost.stored:>7}{cost.wanted:>7}{cost.to_request:>8}"
                     f"{cost.truncated:>5}{cost.total_after:>7}{left:>6}")
    lines.append(f"TOTAL requests to send: {plan.total_requests} "
                 f"over {len(plan.costs)} endpoint(s)")
    if policy is not None:
        lines.append(f"policy: limit {policy.limit} per endpoint, reserve {policy.reserve} "
                     f"(`cut` = units dropped to protect the reserve)")
    lines.extend(f"WARNING: {message}" for message in plan.drift)
    return "\n".join(lines)
