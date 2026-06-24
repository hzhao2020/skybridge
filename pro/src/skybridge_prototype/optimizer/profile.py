from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    PlannerConfig,
    PlanningQuery,
    PlanningScenario,
    RelayLink,
    RuntimeEndpoint,
    RuntimeProfile,
    WorkflowSpec,
)


def load_runtime_profile(path: Path) -> tuple[RuntimeProfile, PlannerConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    workflow_raw = data["workflow"]
    workflow = WorkflowSpec(
        name=workflow_raw.get("name", "video_qa"),
        nodes=list(workflow_raw["nodes"]),
        edges=[tuple(edge) for edge in workflow_raw["edges"]],
    )
    endpoints = [RuntimeEndpoint(**item) for item in data["endpoints"]]
    relay_links = {item["endpoint_id"]: RelayLink(**item) for item in data["relay_links"]}
    queries = [PlanningQuery(**item) for item in data["queries"]]
    scenarios = [PlanningScenario(**item) for item in data["scenarios"]]
    profile = RuntimeProfile(
        workflow=workflow,
        endpoints=endpoints,
        relay_links=relay_links,
        queries=queries,
        scenarios=scenarios,
    )
    config = PlannerConfig(**data.get("planner", {}))
    _validate_profile(profile)
    return profile, config


def profile_from_dict(data: dict[str, Any]) -> tuple[RuntimeProfile, PlannerConfig]:
    tmp = Path("__profile_from_dict_not_used__.json")
    _ = tmp
    workflow_raw = data["workflow"]
    profile = RuntimeProfile(
        workflow=WorkflowSpec(
            name=workflow_raw.get("name", "video_qa"),
            nodes=list(workflow_raw["nodes"]),
            edges=[tuple(edge) for edge in workflow_raw["edges"]],
        ),
        endpoints=[RuntimeEndpoint(**item) for item in data["endpoints"]],
        relay_links={item["endpoint_id"]: RelayLink(**item) for item in data["relay_links"]},
        queries=[PlanningQuery(**item) for item in data["queries"]],
        scenarios=[PlanningScenario(**item) for item in data["scenarios"]],
    )
    _validate_profile(profile)
    return profile, PlannerConfig(**data.get("planner", {}))


def _validate_profile(profile: RuntimeProfile) -> None:
    node_set = set(profile.workflow.nodes)
    for src, dst in profile.workflow.edges:
        if src not in node_set or dst not in node_set:
            raise ValueError(f"Unknown workflow edge {src!r}->{dst!r}")
    roles = {endpoint.role for endpoint in profile.endpoints}
    missing_roles = [node for node in profile.workflow.nodes if node not in roles]
    if missing_roles:
        raise ValueError(f"No endpoint candidates for roles: {', '.join(missing_roles)}")
    missing_links = [
        endpoint.endpoint_id
        for endpoint in profile.endpoints
        if endpoint.endpoint_id not in profile.relay_links
    ]
    if missing_links:
        raise ValueError(f"No relay link profile for endpoints: {', '.join(missing_links)}")
    query_ids = {query.query_id for query in profile.queries}
    bad_scenarios = [scenario.scenario_id for scenario in profile.scenarios if scenario.query_id not in query_ids]
    if bad_scenarios:
        raise ValueError(f"Scenarios reference unknown queries: {', '.join(bad_scenarios)}")
