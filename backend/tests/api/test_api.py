"""The HTTP API end to end, with the stub engine as a real child process."""

from __future__ import annotations

import itertools
import json
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from mlredact.api import app as app_module
from mlredact.api import runner as runner_module
from mlredact.api.engine import EngineResult, kill_tree
from mlredact.api.links import link_key, sign
from mlredact.api.models import JobOptions, JobRecord
from mlredact.api.runner import Runner
from mlredact.api.store import JobStore, iso_from_unix, new_job_id, now_iso

from .conftest import AUTH, ORIGIN, TOKEN, Api, stub_input

# ----------------------------------------------------------------------------------- auth


def test_health_needs_no_token(api: Api) -> None:
    assert api.client.get("/v1/health").json() == {"status": "ok"}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": f"Basic {TOKEN}"}])
@pytest.mark.parametrize(("method", "path"), [("get", "/v1/jobs"), ("get", "/v1/system"), ("post", "/v1/jobs")])
def test_server_routes_need_the_token(api: Api, headers: dict[str, str], method: str, path: str) -> None:
    res = api.client.request(method, path, headers=headers, json={"inputBytes": 1})
    assert res.status_code == 401
    assert res.json() == {"error": "E_API_UNAUTHORIZED"}


def test_errors_are_codes_only(api: Api) -> None:
    assert api.client.get("/v1/nope").json() == {"error": "E_API_NOT_FOUND"}
    assert api.client.get(f"/v1/jobs/{'a' * 32}", headers=AUTH).json() == {"error": "E_API_NOT_FOUND"}
    assert api.client.get("/v1/jobs/../../etc", headers=AUTH).status_code == 404


def test_validation_errors_never_echo_the_request(api: Api) -> None:
    res = api.client.post(
        "/v1/jobs",
        json={"inputBytes": 10, "seeds": [{"type": "nickname", "value": "Jane Citizen"}], "scopeId": "CLAIM-77"},
        headers=AUTH,
    )
    assert res.status_code == 400
    assert res.json() == {"error": "E_API_BAD_REQUEST"}
    assert "Citizen" not in res.text and "CLAIM" not in res.text


# ----------------------------------------------------------------------------------- outcomes


def test_released_job_end_to_end(api: Api) -> None:
    job_id = api.submit("released", profile="broad", style="surrogate", devKey=True)
    view = api.wait(job_id)
    job = view["job"]
    assert job["state"] == "released"
    assert job["exitCode"] == 0
    assert job["input"]["bytes"] == len(stub_input("released"))
    assert job["summary"] == {
        "pages": 1,
        "replaced": 3,
        "kept": 1,
        "regions": 1,
        "lowEvidence": 1,
        "residual": 0,
        "checksPassed": 6,
        "checksTotal": 6,
        "reasons": [],
        "durationMs": 20,
    }
    assert view["files"] == ["redacted.pdf", "manifest.json", "run.json"]
    assert set(view["links"]) == set(view["files"])
    assert view["run"]["workers"] == 2
    assert not (api.jobs_dir / job_id / "in").exists()

    pdf = api.client.get(view["links"]["redacted.pdf"])
    assert pdf.status_code == 200
    assert pdf.content.startswith(b"%PDF-")
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.headers["cache-control"] == "no-store"
    assert pdf.headers["content-disposition"].startswith("inline")
    assert f"mlredact-{job_id[:8]}-redacted.pdf" in pdf.headers["content-disposition"]
    attachment = api.client.get(view["links"]["manifest.json"] + "&download=true")
    assert attachment.headers["content-disposition"].startswith("attachment")
    assert attachment.json()["status"] == "released"


def test_quarantined_after_verification(api: Api) -> None:
    view = api.wait(api.submit("quarantined"))
    assert view["job"]["state"] == "quarantined"
    assert view["job"]["summary"]["reasons"] == ["V4_RESIDUAL"]
    assert view["files"] == ["manifest.json", "run.json", "quarantine.json"]


def test_quarantined_at_intake(api: Api) -> None:
    view = api.wait(api.submit("intake"))
    assert view["job"]["state"] == "quarantined"
    assert view["job"]["summary"]["pages"] == 0


def test_engine_error_keeps_only_the_reason_code(api: Api) -> None:
    job_id = api.submit("error")
    job = api.wait(job_id)["job"]
    assert job["state"] == "failed"
    assert job["errorCode"] == "E_CONFIG_INVALID"
    assert "Citizen" not in api.job_file(job_id)
    assert not (api.jobs_dir / job_id / "in").exists()


def test_engine_crash_gets_an_api_code(api: Api) -> None:
    job_id = api.submit("crash")
    job = api.wait(job_id)["job"]
    assert (job["state"], job["exitCode"], job["errorCode"]) == ("failed", 3, "E_API_ENGINE_FAILED")
    assert "Citizen" not in api.job_file(job_id)


# ----------------------------------------------------------------------------------- privacy


def test_options_reach_the_engine_but_scope_and_seeds_are_never_stored(
    api: Api, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLREDACT_STUB_RECORD_ARGS", "1")
    job_id = api.submit(
        "released",
        profile="maximal",
        style="label",
        workers=3,
        scopeId="MATTER-42",
        seeds=[{"type": "person", "value": "Jane Example"}],
        evaluatorKey="ab" * 32,
    )
    view = api.wait(job_id)
    argv = json.loads((api.jobs_dir / job_id / "out" / "stub-argv.json").read_text())
    value = dict(itertools.pairwise(argv))
    assert argv[0] == "run"
    assert (value["--profile"], value["--style"], value["--workers"], value["--scope-id"]) == (
        "maximal",
        "label",
        "3",
        "MATTER-42",
    )
    assert value["--seeds"].endswith("seeds.json")
    assert value["--sensitive-to"].endswith("evaluator.pub")
    assert "--dev-key" not in argv
    assert "sensitive-manifest.sealed.json" in view["files"]
    record = api.job_file(job_id)
    assert "MATTER-42" not in record and "Jane Example" not in record
    assert view["job"]["options"]["scope"] == "caller"
    assert not (api.jobs_dir / job_id / "in").exists()  # seeds and key went with the input


def test_dev_key_can_be_disabled(api_factory: Callable[..., Api]) -> None:
    api = api_factory(allow_dev_key=False)
    res = api.client.post("/v1/jobs", json={"inputBytes": 5, "devKey": True}, headers=AUTH)
    assert res.json() == {"error": "E_API_DEV_KEY_DISABLED"}


def test_unknown_profile_is_refused(api: Api) -> None:
    res = api.client.post("/v1/jobs", json={"inputBytes": 5, "profile": "lenient"}, headers=AUTH)
    assert res.json() == {"error": "E_API_BAD_PROFILE"}


# ----------------------------------------------------------------------------------- uploads and links


def test_upload_link_works_once_and_only_for_its_job(api: Api) -> None:
    body = stub_input("released")
    created = api.create(body)
    other = api.create(body)
    path = str(created["uploadPath"])
    token = path.split("token=")[1]
    wrong_job = api.client.put(f"/v1/jobs/{other['id']}/input?token={token}", content=body)
    assert wrong_job.json() == {"error": "E_API_LINK_INVALID"}
    tampered = token[:-1] + ("0" if token[-1] != "0" else "1")
    assert api.client.put(path.replace(token, tampered), content=body).json() == {"error": "E_API_LINK_INVALID"}
    assert api.client.put(path, content=body).status_code == 202
    assert api.client.put(path, content=body).json() == {"error": "E_API_UPLOAD_STATE"}


def test_upload_size_limits(api_factory: Callable[..., Api]) -> None:
    api = api_factory(max_upload_bytes=64)
    too_big = api.client.post("/v1/jobs", json={"inputBytes": 65}, headers=AUTH)
    assert too_big.json() == {"error": "E_API_UPLOAD_TOO_LARGE"}
    created = api.create(b"x" * 10)  # declared small, then more is sent
    res = api.client.put(created["uploadPath"], content=b"x" * 100)
    assert (res.status_code, res.json()) == (413, {"error": "E_API_UPLOAD_TOO_LARGE"})
    assert not (api.jobs_dir / created["id"] / "in" / "input").exists()
    empty = api.create(b"x")
    assert api.client.put(empty["uploadPath"], content=b"").json() == {"error": "E_API_EMPTY_UPLOAD"}


def test_download_links_are_bound_to_one_file_and_expire(api: Api) -> None:
    job_id = api.submit("released")
    view = api.wait(job_id)
    link = view["links"]["manifest.json"]
    token = link.split("token=")[1]
    assert api.client.get(f"/v1/jobs/{job_id}/files/redacted.pdf?token={token}").status_code == 403
    assert api.client.get(f"/v1/jobs/{job_id}/files/manifest.json").status_code == 400  # no token
    expired, _ = sign(link_key(TOKEN), "file", job_id, "manifest.json", ttl_s=-5)
    res = api.client.get(f"/v1/jobs/{job_id}/files/manifest.json?token={expired}")
    assert res.json() == {"error": "E_API_LINK_EXPIRED"}
    assert api.client.get(f"/v1/jobs/{job_id}/files/job.json?token={token}").status_code == 404


def test_cors_only_for_the_frontend_origin(api: Api) -> None:
    preflight = {"Access-Control-Request-Method": "PUT", "Access-Control-Request-Headers": "content-type"}
    ok = api.client.options(f"/v1/jobs/{'a' * 32}/input", headers={"Origin": ORIGIN, **preflight})
    assert ok.headers.get("access-control-allow-origin") == ORIGIN
    bad = api.client.options(f"/v1/jobs/{'a' * 32}/input", headers={"Origin": "https://evil.example", **preflight})
    assert "access-control-allow-origin" not in bad.headers


# ----------------------------------------------------------------------------------- queue and control


def test_one_job_at_a_time(api: Api) -> None:
    first = api.submit("released", delay_ms=600)
    second = api.submit("released")
    api.wait(first, until=lambda v: v["job"]["state"] == "running")
    assert api.view(second)["queuePosition"] == 0
    a = api.wait(first)["job"]
    b = api.wait(second)["job"]
    assert b["startedAt"] >= a["finishedAt"]


def test_cancel_queued_and_running(api: Api) -> None:
    blocking = api.submit("hang")
    waiting = api.submit("released")
    api.wait(blocking, until=lambda v: v["job"]["state"] == "running")
    assert api.client.post(f"/v1/jobs/{waiting}/cancel", headers=AUTH).json() == {"ok": True}
    assert api.view(waiting)["job"]["state"] == "cancelled"
    assert api.client.post(f"/v1/jobs/{blocking}/cancel", headers=AUTH).status_code == 200
    job = api.wait(blocking)["job"]
    assert job["state"] == "cancelled"
    assert not (api.jobs_dir / blocking / "in").exists()
    assert api.client.post(f"/v1/jobs/{blocking}/cancel", headers=AUTH).json() == {"error": "E_API_NOT_ACTIVE"}


def test_a_cancel_never_brings_back_a_deleted_job(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = api.client.app.state.runner  # type: ignore[attr-defined]
    blocking = api.submit("hang")
    api.wait(blocking, until=lambda v: v["job"]["state"] == "running")
    waiting = api.submit("released")
    write = JobStore.write
    deleted: list[str] = []
    racer = threading.Thread(target=lambda: deleted.append(runner.delete(waiting)))

    def delete_meanwhile(self: JobStore, job: JobRecord) -> None:
        if job.id == waiting and job.state == "cancelled":
            # A delete arriving between the cancel's read and its write: it must wait for the write.
            racer.start()
            racer.join(timeout=0.5)
        write(self, job)

    monkeypatch.setattr(JobStore, "write", delete_meanwhile)
    assert runner.cancel(waiting) is True
    racer.join(timeout=10)
    assert deleted == ["deleted"]
    assert not (api.jobs_dir / waiting).exists()
    runner.cancel(blocking)


def test_delete_running_and_finished(api: Api) -> None:
    running = api.submit("hang")
    api.wait(running, until=lambda v: v["job"]["state"] == "running")
    res = api.client.delete(f"/v1/jobs/{running}", headers=AUTH)
    assert (res.status_code, res.json()) == (202, {"result": "pending"})
    deadline = time.monotonic() + 20
    while (api.jobs_dir / running).exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not (api.jobs_dir / running).exists()
    done = api.submit("released")
    api.wait(done)
    assert api.client.delete(f"/v1/jobs/{done}", headers=AUTH).json() == {"result": "deleted"}


def test_a_failed_settle_does_not_stop_the_queue(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    run, settle = Runner._run, Runner._settle
    calls = itertools.count()

    def failing_run(self: Runner, active: Any, scope_id: str | None) -> None:
        if next(calls) == 0:
            raise RuntimeError("internal")
        run(self, active, scope_id)

    def failing_settle(self: Runner, active: Any, **kw: Any) -> None:
        if kw["error_code"] == "E_API_INTERNAL":
            raise OSError("a handle is still open")
        settle(self, active, **kw)

    monkeypatch.setattr(Runner, "_run", failing_run)
    monkeypatch.setattr(Runner, "_settle", failing_settle)
    api.submit("released")
    assert api.wait(api.submit("released"))["job"]["state"] == "released"


def test_sweep_fails_a_job_whose_outcome_could_not_be_recorded(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_settle(self: Runner, active: Any, **kw: Any) -> None:
        raise OSError("disk unavailable")

    monkeypatch.setattr(Runner, "_settle", broken_settle)
    runner = api.client.app.state.runner  # type: ignore[attr-defined]
    job_id = api.submit("released")
    deadline = time.monotonic() + 20
    while job_id not in runner._unsettled and time.monotonic() < deadline:
        time.sleep(0.05)
    assert api.view(job_id)["job"]["state"] == "running"
    runner._sweep()
    job = api.view(job_id)["job"]
    assert (job["state"], job["errorCode"]) == ("failed", "E_API_INTERNAL")
    assert not (api.jobs_dir / job_id / "in").exists()
    assert job_id not in runner._unsettled


def test_cancel_and_delete_after_the_outcome_is_recorded(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    settle = Runner._settle
    raced: dict[str, object] = {}

    def settle_then_race(self: Runner, active: Any, **kw: Any) -> None:
        settle(self, active, **kw)
        # Both land after the final record is written, while the job is still the active one.
        raced["cancel"] = self.cancel(active.id)
        raced["delete"] = self.delete(active.id)

    monkeypatch.setattr(Runner, "_settle", settle_then_race)
    job_id = api.submit("released")
    deadline = time.monotonic() + 20
    while (api.jobs_dir / job_id).exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert raced == {"cancel": False, "delete": "pending"}
    assert not (api.jobs_dir / job_id).exists()


def test_an_internal_error_after_spawn_kills_the_engine(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    attach = Runner._attach
    procs: list[subprocess.Popen[bytes]] = []

    def recording_attach(self: Runner, active: Any, proc: subprocess.Popen[bytes]) -> bool:
        procs.append(proc)
        return attach(self, active, proc)

    def broken_watch(stream: Any) -> Any:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(Runner, "_attach", recording_attach)
    monkeypatch.setattr(runner_module, "watch_error_code", broken_watch)
    job = api.wait(api.submit("hang"))["job"]
    try:
        assert (job["state"], job["errorCode"]) == ("failed", "E_API_INTERNAL")
        assert procs and procs[0].poll() is not None
    finally:
        for proc in procs:
            kill_tree(proc)


def test_shutdown_interrupts_the_running_job(api: Api) -> None:
    job_id = api.submit("hang")
    api.wait(job_id, until=lambda v: v["job"]["state"] == "running")
    api.client.app.state.runner.stop()  # type: ignore[attr-defined]
    job = JobStore(api.jobs_dir).read(job_id)
    assert job is not None
    assert (job.state, job.error_code) == ("interrupted", "E_API_INTERRUPTED")
    assert not (api.jobs_dir / job_id / "in").exists()


def test_timeout_kills_the_job(api_factory: Callable[..., Api]) -> None:
    api = api_factory(job_timeout_s=1.0)
    job = api.wait(api.submit("hang"))["job"]
    assert (job["state"], job["errorCode"]) == ("failed", "E_API_TIMEOUT")


def test_missing_engine(api_factory: Callable[..., Api]) -> None:
    api = api_factory(engine=("mlredact-engine-that-does-not-exist",))
    job = api.wait(api.submit("released"))["job"]
    assert (job["state"], job["errorCode"]) == ("failed", "E_API_ENGINE_NOT_FOUND")


def test_list_jobs_newest_first(api: Api) -> None:
    for _ in range(2):
        api.wait(api.submit("released"))
    jobs = api.client.get("/v1/jobs", headers=AUTH).json()["jobs"]
    stamps = [j["createdAt"] for j in jobs]
    assert stamps == sorted(stamps, reverse=True)


# ----------------------------------------------------------------------------------- recovery and sweep


def _record(job_id: str, state: str, **extra: object) -> JobRecord:
    options = JobOptions(
        profile="broad", style="blackout", workers=1, scope="document", seed_count=0, sealed=False, dev_key=False
    )
    return JobRecord(id=job_id, created_at=now_iso(), state=state, options=options, **extra)  # type: ignore[arg-type]


def test_restart_interrupts_unfinished_jobs(tmp_path: Path, api_factory: Callable[..., Api]) -> None:
    jobs = tmp_path / "jobs"
    store = JobStore(jobs)
    ids = {state: new_job_id() for state in ("running", "queued", "awaiting_upload", "released")}
    for state, job_id in ids.items():
        store.write(_record(job_id, state))
        store.input_dir(job_id).mkdir(parents=True, exist_ok=True)
        (store.input_dir(job_id) / "input").write_bytes(b"%PDF document")
    api_factory(jobs_dir=jobs)
    for state in ("running", "queued"):
        job = store.read(ids[state])
        assert job is not None
        assert (job.state, job.error_code) == ("interrupted", "E_API_INTERRUPTED")
        assert not store.input_dir(ids[state]).exists()
    assert not (jobs / ids["awaiting_upload"]).exists()
    released = store.read(ids["released"])
    assert released is not None and released.state == "released"


def test_sweep_drops_expired_uploads(api: Api) -> None:
    store = JobStore(api.jobs_dir)
    stale = new_job_id()
    store.write(_record(stale, "awaiting_upload", upload_expires_at=iso_from_unix(time.time() - 60)))
    fresh = api.create(b"x")["id"]
    api.client.app.state.runner._sweep()  # type: ignore[attr-defined]
    assert not (api.jobs_dir / stale).exists()
    assert (api.jobs_dir / fresh).exists()


def test_sweep_forgets_the_scope_id_of_an_expired_upload(api: Api) -> None:
    store = JobStore(api.jobs_dir)
    job_id = api.create(b"x", scopeId="MATTER-42")["id"]
    job = store.read(job_id)
    assert job is not None
    store.write(job.model_copy(update={"upload_expires_at": iso_from_unix(time.time() - 60)}))
    runner = api.client.app.state.runner  # type: ignore[attr-defined]
    runner._sweep()
    assert not (api.jobs_dir / job_id).exists()
    assert job_id not in runner._scope_ids


def test_sweep_runs_while_a_job_is_running(api_factory: Callable[..., Api], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "SWEEP_EVERY_S", 0.1)
    api = api_factory()
    running = api.submit("hang")
    api.wait(running, until=lambda v: v["job"]["state"] == "running")
    stale = new_job_id()
    JobStore(api.jobs_dir).write(_record(stale, "awaiting_upload", upload_expires_at=iso_from_unix(time.time() - 60)))
    deadline = time.monotonic() + 10
    while (api.jobs_dir / stale).exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not (api.jobs_dir / stale).exists()
    assert api.view(running)["job"]["state"] == "running"
    api.client.post(f"/v1/jobs/{running}/cancel", headers=AUTH)


def test_a_failing_sweep_keeps_sweeping(api_factory: Callable[..., Api], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "SWEEP_EVERY_S", 0.05)
    sweep = Runner._sweep
    calls = itertools.count()

    def flaky_sweep(self: Runner) -> None:
        if next(calls) == 0:
            raise ValueError("bad timestamp")
        sweep(self)

    monkeypatch.setattr(Runner, "_sweep", flaky_sweep)
    api = api_factory()
    stale = new_job_id()
    JobStore(api.jobs_dir).write(_record(stale, "awaiting_upload", upload_expires_at=iso_from_unix(time.time() - 60)))
    deadline = time.monotonic() + 10
    while (api.jobs_dir / stale).exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not (api.jobs_dir / stale).exists()
    assert api.wait(api.submit("released"))["job"]["state"] == "released"


# ----------------------------------------------------------------------------------- system


def test_system_reports_key_presence_only(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLREDACT_SURROGATE_KEY", "f" * 64)
    res = api.client.get("/v1/system", headers=AUTH)
    info = res.json()
    assert info["surrogateKey"] == {"fromEnv": True, "fromConfig": False, "devKeyAllowed": True}
    assert info["profiles"] == ["broad", "claimant_focused", "maximal"]
    assert "f" * 64 not in res.text


@pytest.mark.parametrize(
    ("surrogate", "runtime", "expected"),
    [
        ({"key_file": "surrogate.hex"}, {}, True),
        # A named key file that is missing fails the job, even with the development key allowed.
        ({"key_file": "missing.hex", "allow_development_key": True}, {}, False),
        ({"allow_development_key": True}, {}, True),
        # A pinned production deployment refuses the development key.
        ({"allow_development_key": True}, {"expected_profile": "cpu"}, False),
        ({"font_family": "Serif"}, {}, False),
    ],
)
def test_system_reports_a_key_from_a_deployment_config(
    api_factory: Callable[..., Api], tmp_path: Path, surrogate: dict[str, Any], runtime: dict[str, Any], expected: bool
) -> None:
    (tmp_path / "surrogate.hex").write_text("f" * 64)
    config = tmp_path / "deploy.yaml"
    config.write_text(json.dumps({"surrogate": surrogate, **({"runtime": runtime} if runtime else {})}))
    # A relative key_file resolves in the engine's working directory.
    api = api_factory(config_files=(config,), engine_cwd=tmp_path)
    res = api.client.get("/v1/system", headers=AUTH)
    assert res.json()["surrogateKey"]["fromConfig"] is expected
    assert "surrogate.hex" not in res.text


def test_selfcheck(api: Api) -> None:
    report = api.client.post("/v1/system/selfcheck", json={"profile": "maximal"}, headers=AUTH).json()
    assert report["ok"] is True
    assert report["report"]["runtime_profile"] == "stub-x86_64"
    bad = api.client.post("/v1/system/selfcheck", json={"profile": "nope"}, headers=AUTH)
    assert bad.json() == {"error": "E_API_BAD_PROFILE"}


def test_a_selfcheck_retry_joins_the_run_in_progress(api: Api, monkeypatch: pytest.MonkeyPatch) -> None:
    started = threading.Event()
    release = threading.Event()
    calls = itertools.count()

    def slow_selfcheck(*_args: Any, **_kw: Any) -> EngineResult:
        next(calls)
        started.set()
        release.wait(timeout=10)
        return EngineResult(0, json.dumps({"runtime_profile": "stub"}), None, spawn_failed=False, timed_out=False)

    def post(profile: str) -> Any:
        return api.client.post("/v1/system/selfcheck", json={"profile": profile}, headers=AUTH)

    monkeypatch.setattr(app_module, "run_short", slow_selfcheck)
    replies: list[Any] = []
    first = threading.Thread(target=lambda: replies.append(post("maximal").json()))
    first.start()
    assert started.wait(timeout=10)
    retry = threading.Thread(target=lambda: replies.append(post("maximal").json()))
    retry.start()
    assert post("broad").json() == {"error": "E_API_SELFCHECK_BUSY"}
    time.sleep(0.2)  # let the retry reach the run in progress
    release.set()
    first.join(timeout=10)
    retry.join(timeout=10)
    assert [r["ok"] for r in replies] == [True, True]
    assert next(calls) == 1  # one engine run served both requests
