"""API surface smoke tests (real clock, heavily time-compressed, CPU chains)."""

from __future__ import annotations

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pipeline_kit.config import KitSettings  # noqa: E402
from service.api import create_app  # noqa: E402
from service.service import JobService  # noqa: E402


@pytest.fixture
def client():
    service = JobService(settings=KitSettings(time_scale=200.0))
    app = create_app(service)
    with TestClient(app) as test_client:
        yield test_client


def _poll_terminal(client: TestClient, job_id: str, timeout_s: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = client.get(f"/jobs/{job_id}").json()
        if payload["state"] in {"succeeded", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not settle within {timeout_s}s")


def test_submit_status_trace_roundtrip(client):
    response = client.post(
        "/jobs", json={"chain": "cutout", "image": {"id": "img-api-1"}}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["state"] == "queued"

    final = _poll_terminal(client, body["job_id"])
    assert final["state"] == "succeeded"
    assert final["final_result_id"] is not None

    trace = client.get(f"/jobs/{body['job_id']}/trace").json()
    assert [s["step"] for s in trace["steps"]] == ["segment", "remove_bg"]
    assert all(s["state"] == "succeeded" for s in trace["steps"])

    metrics = client.get("/metrics").json()
    assert metrics["counters"]["jobs_succeeded"] >= 1
    assert "total_cost" in metrics["gpu"]


def test_invalid_chain_is_a_422_naming_the_problem(client):
    response = client.post(
        "/jobs",
        json={"chain": ["segment", "generate_multiview"], "image": {"id": "x"}},
    )
    assert response.status_code == 422
    assert "generate_multiview" in str(response.json()["detail"])


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


def test_agent_tools_schema_and_dispatch(client):
    tools = client.get("/agent/tools").json()
    assert {t["name"] for t in tools} == {
        "submit_chain", "get_job", "cancel_job", "get_system_status",
    }
    assert all(t["description"] and t["input_schema"] for t in tools)

    status = client.post("/agent/tools/get_system_status", json={}).json()
    assert status["accepting_gpu_jobs"] is True
    assert status["gpu_capacity"] == 4

    submitted = client.post(
        "/agent/tools/submit_chain",
        json={"preset": "cutout", "image_id": "img-agent-1"},
    ).json()
    assert "job_id" in submitted
    duplicate = client.post(
        "/agent/tools/submit_chain",
        json={"preset": "cutout", "image_id": "img-agent-1"},
    ).json()
    assert duplicate["job_id"] == submitted["job_id"]
    assert duplicate["dedup"] is True

    final = _poll_terminal(client, submitted["job_id"])
    assert final["state"] == "succeeded"
    assert final["suggestion"] is None
