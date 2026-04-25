# SPDX-License-Identifier: Apache-2.0
"""Video-MME accuracy and speed CI for Qwen3-Omni (Text+Video -> Text, Talker OFF).

Usage:
    pytest tests/test_model/test_qwen3_omni_videomme_ci.py -s -x

Author:
    Qiujiang Chen https://github.com/Jayon02
    Chenyang Zhao https://github.com/zhaochenyang20
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from benchmarks.dataset.prepare import DATASETS
from benchmarks.eval.benchmark_omni_videomme import (
    VideoMMEEvalConfig,
    run_videomme_eval,
)
from sglang_omni.utils import find_available_port
from tests.utils import (
    ServerHandle,
    apply_slack,
    assert_speed_thresholds,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

CONCURRENCY = 4
STARTUP_TIMEOUT = 60

# Note (Chenyang): calibrated on H200 across 5 back-to-back fresh-server
# pytest invocations of this test at concurrency=4 against the
# zhaochenyang20/Video_MME_ci_25 dataset (first-25 prefix of the
# parent Video_MME_ci 50-sample CI subset). Each pytest run starts
# and stops its own server, so every data point sees a pristine GPU.
# Observed per-run:
#
#   run_1: acc=0.60 correct=15/25 failed=0 tput=0.086 toks=2.6 lat=45.64
#   run_2: acc=0.60 correct=15/25 failed=0 tput=0.086 toks=2.8 lat=45.50
#   run_3: acc=0.60 correct=15/25 failed=0 tput=0.085 toks=2.7 lat=45.63
#   run_4: acc=0.60 correct=15/25 failed=0 tput=0.088 toks=2.7 lat=44.49
#   run_5: acc=0.56 correct=14/25 failed=0 tput=0.083 toks=2.6 lat=47.11
#
# Going from videomme-ci-50 to videomme-ci-25 cut wall-clock from
# ~10 min/run to ~5 min/run AND eliminated the ~20% mid-run-OOM flake
# the 50-sample subset showed at the same fixture (the shorter window
# does not accumulate enough encoder-activation fragmentation on the
# thinker GPU to push the pinned mem_fraction_static=0.729 over the
# OOM line). MAX_FAILED is therefore back to 0 (worst-of-5) — any
# regression that causes even one dropped request fails the test.
# _VIDEOMME_P95 below feeds the worst-of-5 speed numbers (min
# tput/toks, max lat); apply_slack(0.75, 1.25) derives the enforced
# thresholds with ±25% machine-variance slack.

VIDEOMME_MIN_ACCURACY = 0.56
VIDEOMME_MAX_FAILED = 0

_VIDEOMME_P95 = {
    4: {
        "throughput_qps": 0.083,
        "tok_per_s_agg": 2.6,
        "latency_mean_s": 47.1,
    },
}
VIDEOMME_THRESHOLDS = apply_slack(_VIDEOMME_P95)


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the text-only Qwen3-Omni server and wait until healthy.

    Note (Chenyang):
    On CI (GITHUB_ACTIONS=true) server stdout/stderr are captured into a
    log file so the main test output stays tidy and the log is attached on
    startup failure. Locally the server inherits the parent's stdout/stderr
    so progress streams live under pytest -s.
    """
    port = find_available_port()
    is_ci = os.environ.get("GITHUB_ACTIONS") == "true"
    log_file: Path | None = (
        tmp_path_factory.mktemp("server_logs") / "server.log" if is_ci else None
    )
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_server.py",
        "--model-path",
        MODEL_PATH,
        "--port",
        str(port),
        "--model-name",
        "qwen3-omni",
        "--thinker-max-seq-len",
        "32768",
        "--encoder-mem-reserve",
        "0.20",
    ]
    proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
    yield ServerHandle(proc=proc, port=port)
    stop_server(proc)


@pytest.mark.benchmark
def test_videomme_accuracy_and_speed(
    server_process: ServerHandle,
    tmp_path: Path,
) -> None:
    """Run videomme-ci-25 at concurrency=4 and assert accuracy + speed thresholds."""
    config = VideoMMEEvalConfig(
        model="qwen3-omni",
        port=server_process.port,
        max_concurrency=CONCURRENCY,
        output_dir=str(tmp_path / "videomme"),
        repo_id=DATASETS["videomme-ci-25"],
        disable_tqdm=True,
    )
    results = asyncio.run(run_videomme_eval(config))

    summary = results["summary"]
    failed = summary.get("failed", 0)
    total = summary.get("total_samples", 0)
    assert failed <= VIDEOMME_MAX_FAILED, (
        f"Video-MME had {failed}/{total} failed requests, "
        f"which exceeds the threshold {VIDEOMME_MAX_FAILED}"
    )

    assert summary["accuracy"] >= VIDEOMME_MIN_ACCURACY, (
        f"Video-MME accuracy {summary['accuracy']:.4f} "
        f"({summary['accuracy'] * 100:.1f}%) < "
        f"threshold {VIDEOMME_MIN_ACCURACY} ({VIDEOMME_MIN_ACCURACY * 100:.0f}%)"
    )

    assert_speed_thresholds(results["speed"], VIDEOMME_THRESHOLDS, CONCURRENCY)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
