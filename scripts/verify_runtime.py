#!/usr/bin/env python3
"""Measure local archival HTTP reads and frozen object-model inference separately.

No training, POST requests, external network, model changes or published data.
The script owns its temporary API process/SQLite database and always stops it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
from http.client import HTTPConnection
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def csv_file(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return {"count": 0}
    # Explicit nearest-rank quantiles, also reproducible from the raw CSV.
    percentile = lambda q: ordered[max(0, math.ceil(q * len(ordered)) - 1)]
    return {"count": len(ordered), "min_ms": ordered[0], "mean_ms": statistics.mean(ordered),
            "p50_ms": percentile(.5), "p95_ms": percentile(.95), "p99_ms": percentile(.99),
            "max_ms": ordered[-1]}


def environment():
    versions = {}
    for package in ["fastapi", "pydantic", "uvicorn", "numpy", "pandas", "lightgbm", "pyarrow", "scikit-learn"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    docker = {"client_path": shutil.which("docker"), "container_test_performed": False}
    if docker["client_path"]:
        try:
            result = subprocess.run([docker["client_path"], "version"], capture_output=True, text=True, timeout=5)
            docker.update(returncode=result.returncode, output=result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            docker["error"] = "docker version exceeded 5 seconds; no daemon was started"
    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "executable": sys.executable, "platform": platform.platform(), "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(), "load_average_before": list(os.getloadavg()),
            "package_versions": versions, "docker": docker,
            "isolation": "Shared local machine; other processes are not stopped or controlled"}


def http_json(base, path, connection=None):
    if connection is not None:
        connection.request("GET", path)
        response = connection.getresponse()
        body = response.read()
        return response.status, len(body), json.loads(body)
    with urlopen(base + path, timeout=20) as response:
        body = response.read()
        return response.status, len(body), json.loads(body)


def validate_payload(path, data):
    if path == "/api/health":
        assert data["status"] == "ok" and data["available_modes"]["object"]
    elif path.startswith("/api/risks?"):
        assert data["entity_mode"] == "object" and len(data["cards"]) > 0
        assert not any(any(key.startswith("actual_") or key == "outcome" for key in card) for card in data["cards"])
    elif path.startswith("/api/risks/"):
        assert data["forecast"]["entity_mode"] == "object"
        assert not any(key.startswith("actual_") or key == "outcome" for key in data["card"])
    elif path.startswith("/api/journal"):
        assert data["entries"] == []
    elif path.startswith("/api/tickets"):
        assert data["tickets"] == []
    elif path.startswith("/api/feedback"):
        assert data["feedback"] == []


def benchmark_api(args, output):
    source_paths = [ROOT / "api/main.py", ROOT / "data/app/object_risk_demo.json"]
    before = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    process = None
    shutdown = None
    with tempfile.TemporaryDirectory(prefix="lct-runtime-") as temporary, (output / "uvicorn.log").open("w") as log:
        # Inherited bound socket eliminates the find-free-port/start-server race.
        # This local harness supports macOS/Linux, the two environments documented here.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", args.port))
            listener.listen(128)
            port = listener.getsockname()[1]
            command = [sys.executable, "-m", "uvicorn", "api.main:app", "--fd", str(listener.fileno()),
                       "--workers", "1", "--no-access-log", "--log-level", "warning"]
            env = {**os.environ, "LCT_TICKET_DB": str(Path(temporary) / "tickets.sqlite3")}
            start = time.perf_counter()
            try:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           pass_fds=(listener.fileno(),))
                base = f"http://127.0.0.1:{port}"
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("Owned API process exited; inspect uvicorn.log")
                    try:
                        _, _, health = http_json(base, "/api/health")
                        validate_payload("/api/health", health)
                        break
                    except (OSError, HTTPError):
                        if time.perf_counter() - start > 20:
                            raise RuntimeError("API startup exceeded 20 seconds")
                        time.sleep(.05)
                startup = time.perf_counter() - start
                _, _, archive = http_json(base, "/api/risks?mode=object")
                card_id = quote(archive["cards"][0]["id"], safe="")
                paths = ["/api/health", "/api/risks?mode=object", f"/api/risks/{card_id}?mode=object",
                         "/api/journal?mode=object", "/api/tickets?mode=object", "/api/feedback?mode=object"]
                # Warm each endpoint; SQLite first-use schema creation is outside timed reads.
                for path in paths:
                    _, _, payload = http_json(base, path)
                    validate_payload(path, payload)
                barrier = threading.Barrier(args.users + 1)
                lock = threading.Lock()
                state = {"inflight": 0, "max_inflight": 0, "deadline": 0.0, "start": 0.0}

                def user(user_id):
                    rows = []
                    sequence = 0
                    client = HTTPConnection("127.0.0.1", port, timeout=20) if args.connection_mode == "keep-alive" else None
                    barrier.wait()
                    while time.perf_counter() < state["deadline"]:
                        path = paths[(user_id + sequence) % len(paths)]
                        began = time.perf_counter()
                        with lock:
                            state["inflight"] += 1
                            state["max_inflight"] = max(state["max_inflight"], state["inflight"])
                        status, size, error = None, 0, ""
                        try:
                            status, size, payload = http_json(base, path, client)
                            validate_payload(path, payload)
                        except Exception as exc:
                            status = exc.code if isinstance(exc, HTTPError) else status
                            error = f"{type(exc).__name__}: {exc}"
                            if client is not None:
                                client.close()
                        finally:
                            ended = time.perf_counter()
                            with lock:
                                state["inflight"] -= 1
                        rows.append({"user": user_id, "request": sequence, "endpoint": path,
                                     "start_offset_seconds": began - state["start"], "latency_ms": (ended - began) * 1000,
                                     "status": status, "response_bytes": size, "error": error})
                        sequence += 1
                    if client is not None:
                        client.close()
                    return rows

                with ThreadPoolExecutor(max_workers=args.users) as pool:
                    futures = [pool.submit(user, i) for i in range(args.users)]
                    state["start"] = time.perf_counter()
                    state["deadline"] = state["start"] + args.duration
                    barrier.wait()
                    rows = [row for future in futures for row in future.result()]
                    elapsed = time.perf_counter() - state["start"]
                csv_file(output / "api_requests.csv", sorted(rows, key=lambda item: item["start_offset_seconds"]))
                failures = [row for row in rows if row["error"] or row["status"] != 200]
                result = {"kind": "HTTP reads of precomputed archive; no ML inference", "base_url": base,
                          "owned_pid": process.pid, "server_workers": 1, "users": args.users,
                          "client": "One thread per user, no think time; latency includes body read/JSON decode/payload checks",
                          "connection_mode": args.connection_mode,
                          "duration_requested_seconds": args.duration, "wall_seconds": elapsed,
                          "startup_to_health_seconds": startup, "max_client_inflight": state["max_inflight"],
                          "requests": len(rows), "errors": len(failures), "requests_per_second": len(rows) / elapsed,
                          "all_requests_latency": distribution([row["latency_ms"] for row in rows]),
                          "by_endpoint": {path: distribution([row["latency_ms"] for row in rows if row["endpoint"] == path]) for path in paths},
                          "archive_cards": len(archive["cards"]), "feature_date": archive["feature_date"],
                          "source_sha256": before, "read_only_http_methods": ["GET"],
                          "database": "new temporary empty SQLite; GET startup may initialize schema; no user database used",
                          "limitations": "Loopback client and server share CPU; no browser rendering/TLS/authentication/remote network, no writes and no persistent-user load"}
            finally:
                if process is not None:
                    if process.poll() is None:
                        process.terminate()
                        try:
                            process.wait(timeout=10)
                            shutdown = "owned process terminated and reaped"
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=5)
                            shutdown = "owned process killed after timeout and reaped"
                    else:
                        shutdown = "owned process already exited and reaped"
        # Parent socket is now also closed; no port or API process remains owned by this run.
    after = {str(path.relative_to(ROOT)): sha256(path) for path in source_paths}
    result.update(source_files_unchanged=before == after, owned_process_shutdown=shutdown)
    result["passed"] = not failures and before == after and state["max_inflight"] == args.users
    dump(output / "api_summary.json", result)
    return result


def benchmark_inference(args, output):
    started = time.perf_counter()
    import lightgbm as lgb
    import numpy as np
    sys.path.insert(0, str(ROOT / "src/modeling"))
    from export_object_demo import prediction_frame
    imports_seconds = time.perf_counter() - started
    cfg_path = ROOT / "reports/object_risk_probe/configuration.json"
    cfg = json.loads(cfg_path.read_text())
    model_path = ROOT / cfg["model_path"]
    if sha256(model_path) != cfg["model_sha256"]:
        raise ValueError("Frozen model checksum mismatch")
    input_paths = [cfg_path, model_path, ROOT / "src/modeling/object_risk_probe.py",
                   ROOT / "src/modeling/export_object_demo.py", ROOT / "data/raw/справочник_каналов_датчиков.csv",
                   ROOT / "data/raw/справочник_объектов_диспетчер.csv"]
    input_paths += [ROOT / f"data/interim/daily_{year}.parquet" for year in range(2025, int(args.feature_date[:4]) + 1)]
    hashes = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    for key, path in [("meta_sha256", "data/raw/справочник_каналов_датчиков.csv"),
                      ("objects_sha256", "data/raw/справочник_объектов_диспетчер.csv"),
                      ("daily_2025_sha256", "data/interim/daily_2025.parquet")]:
        if hashes[path] != cfg[key]:
            raise ValueError(f"Frozen training input differs: {path}")
    # This path reads daily aggregates only through D, and asserts missing future labels.
    start = time.perf_counter()
    latest, coverage = prediction_frame(args.feature_date, cfg)
    x = latest[cfg["features"]]
    features_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model = lgb.Booster(model_file=str(model_path))
    load_seconds = time.perf_counter() - start
    if model.feature_name() != cfg["features"]:
        raise ValueError("Frozen model feature ordering mismatch")
    start = time.perf_counter()
    reference = model.predict(x, num_threads=args.ml_threads)
    first_batch_seconds = time.perf_counter() - start
    if not np.isfinite(reference).all() or ((reference < 0) | (reference > 1)).any():
        raise ValueError("Invalid model scores")
    rows = []
    max_difference = 0.0
    for kind in ["batch_scores", "one_object_scores", "batch_shap_contributions"]:
        repeats = args.repeats if kind != "batch_shap_contributions" else max(10, args.repeats // 10)
        for iteration in range(repeats):
            index = iteration % len(x)
            frame = x.iloc[[index]] if kind == "one_object_scores" else x
            start = time.perf_counter()
            prediction = model.predict(frame, pred_contrib=kind == "batch_shap_contributions", num_threads=args.ml_threads)
            elapsed = time.perf_counter() - start
            if kind == "batch_scores":
                difference = np.max(np.abs(prediction - reference))
            elif kind == "one_object_scores":
                difference = abs(prediction[0] - reference[index])
            else:
                reconstructed = 1 / (1 + np.exp(-prediction.sum(axis=1)))
                difference = np.max(np.abs(reconstructed - reference))
            max_difference = max(max_difference, float(difference))
            rows.append({"kind": kind, "repeat": iteration, "objects": len(frame),
                         "latency_ms": elapsed * 1000, "max_score_difference": float(difference)})
    csv_file(output / "inference_samples.csv", rows)
    unchanged = hashes == {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    result = {"kind": "Real CPU LightGBM predict calls, never JSON reads", "feature_date": args.feature_date,
              "objects": len(x), "features": len(cfg["features"]), "trees": model.num_trees(),
              "num_threads": args.ml_threads, "model_sha256": cfg["model_sha256"], "source_sha256": hashes,
              "module_import_seconds": imports_seconds, "feature_rebuild_from_daily_parquet_seconds": features_seconds,
              "model_load_seconds": load_seconds, "first_batch_predict_seconds": first_batch_seconds,
              "features_load_predict_seconds": features_seconds + load_seconds + first_batch_seconds,
              "warm_predict": {kind: distribution([row["latency_ms"] for row in rows if row["kind"] == kind]) for kind in sorted({row["kind"] for row in rows})},
              "repeat_score_max_difference": max_difference, "inputs_unchanged": unchanged,
              "coverage": coverage, "future_label_access_for_scoring": False, "training_performed": False,
              "limitations": "Daily Parquet is precomputed: raw archive extraction, online ingestion, cold filesystem cache and remote serving are not measured. Feature rebuild uses all local history from 2025-01-01 through D; a one-object predict excludes this shared preparation. Warm score timings exclude SHAP, measured separately.",
              "passed": unchanged and max_difference < 1e-12}
    dump(output / "inference_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--users", type=int, default=20)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--port", type=int, default=0, help="0 selects an available loopback port")
    parser.add_argument("--connection-mode", choices=["keep-alive", "new"], default="keep-alive",
                        help="Keep one HTTP/1.1 connection per user, or stress new connections")
    parser.add_argument("--feature-date", default="2026-06-28")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--ml-threads", type=int, default=2)
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/runtime_validation")
    args = parser.parse_args()
    if args.users < 1 or args.duration <= 0 or args.repeats < 1 or args.ml_threads < 1:
        parser.error("users, duration, repeats and ml-threads must be positive")
    if args.skip_api and args.skip_inference:
        parser.error("At least one measurement must run")
    output = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output.mkdir(parents=True, exist_ok=False)
    report = {"environment": environment(), "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
              "script_sha256": sha256(__file__), "artifact_directory": str(output.relative_to(ROOT)) if output.is_relative_to(ROOT) else str(output)}
    dump(output / "environment.json", report)
    try:
        if not args.skip_api:
            print("Measuring local HTTP reads with an owned temporary API process...", flush=True)
            report["api"] = benchmark_api(args, output)
        if not args.skip_inference:
            print("Measuring frozen object-model inference and feature preparation...", flush=True)
            report["inference"] = benchmark_inference(args, output)
        report["load_average_after"] = list(os.getloadavg())
        report["passed"] = all(report[key]["passed"] for key in ["api", "inference"] if key in report)
    except Exception as exc:
        report.update(passed=False, error=f"{type(exc).__name__}: {exc}")
        dump(output / "summary.json", report)
        raise
    dump(output / "summary.json", report)
    print(json.dumps({"passed": report["passed"], "report": str(output / "summary.json")}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
