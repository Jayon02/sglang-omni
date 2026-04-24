# SPDX-License-Identifier: Apache-2.0
"""Video-MME benchmark for sglang-omni models.

Evaluates video understanding accuracy and performance on the Video-MME
test set via /v1/chat/completions with video input. Each sample is a
multiple-choice question (A-D) grounded in a YouTube video clip, covering
short, medium, and long durations across six domains.

Usage:

    # 1. Download the dataset (Video-MME test split, 2520 questions)
    python -m benchmarks.dataset.prepare --dataset videomme

    # 2. Launch the thinker-only server. The full test split contains
    #    long videos whose prompts approach the 32k-token thinker context;
    #    --thinker-max-seq-len 32768 accommodates the longest ones, and
    #    --encoder-mem-reserve 0.40 holds back ~56 GB of GPU memory for
    #    the co-located video encoder at peak activation (plus inter-stage
    #    relay buffers), outside SGLang's static KV pool. (The 50-sample
    #    CI subset fits with reserve 0.20; the full split needs the larger
    #    reserve because "medium" and "long" clips allocate substantially
    #    more encoder activations on top of the longer prompts. Going
    #    higher than 0.40 on a single H200 leaves SGLang with too little
    #    KV pool to boot — 0.40 is the empirical sweet spot.)
    python examples/run_qwen3_omni_server.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port 8000 \
        --thinker-max-seq-len 32768 \
        --encoder-mem-reserve 0.40

    # 3. Run the benchmark at concurrency 4. The ``--max-samples`` cap
    #    below is tuned to match the reference table — on a single H200
    #    the long-video encoder activations accumulate fragmentation on
    #    the thinker GPU, and the success rate degrades noticeably past
    #    the ~120-sample mark even with ``PYTORCH_ALLOC_CONF=
    #    expandable_segments:True``. 100 samples is the empirical
    #    high-watermark that completes cleanly; the CI subset at stage-7
    #    covers regression, and a full 2520-sample sweep should be
    #    scheduled as a chunked overnight job with a fresh server
    #    restart every ~100 requests.
    python benchmarks/eval/benchmark_omni_videomme.py \
        --model qwen3-omni --port 8000 \
        --max-concurrency 4 --max-tokens 256 --max-samples 100


H200 Reference Results

Reproducibility references for a Video-MME slice — NOT CI thresholds.
CI runs on the curated 50-sample subset ``videomme-ci-50`` and keeps
its own thresholds in
``tests/test_model/test_qwen3_omni_videomme_ci.py``.

Benchmark: Video-MME | Dataset: lmms-lab/Video-MME test split
                       (2520 questions full; first N samples used here)
Hardware:  1 x H200 (default; non-H200 sources are tagged in Source column)
Last verified: 2026-04-24

Accuracy (summary)

| Model      | Config                          | accuracy | correct | failed | mc_fallback | Source                                                              |
| ---------- | ------------------------------- | -------- | ------- | ------ | ----------- | ------------------------------------------------------------------- |
| Qwen3-Omni | thinker-only, encoder-reserve=0.40 | 76.00% | 76/100  | 0      | 2           | PR #327 [H200, first-100 prefix, c=4, max_tokens=256] |

Speed (speed)

| Model      | Config                             | latency_mean_s | latency_p95_s | throughput_qps | tok_per_s_mean | tok_per_s_agg | Source                                                |
| ---------- | ---------------------------------- | -------------- | ------------- | -------------- | -------------- | ------------- | ----------------------------------------------------- |
| Qwen3-Omni | thinker-only, encoder-reserve=0.40 | 42.96          | 76.17         | 0.093          | 2.80           | 2.60          | PR #327 [H200, first-100 prefix, c=4, max_tokens=256] |

Note (Chenyang): full 2520 not run as a single contiguous job — on a
single H200 with c=4 the long-video encoder activations accumulate
fragmentation on the thinker GPU and the success rate degrades past
~100 requests even with ``PYTORCH_ALLOC_CONF=expandable_segments:True``
set on the server (an earlier run observed 54% failures at request 330
of a 500-sample attempt under otherwise-identical flags). 100 samples
is the empirical clean window for a single server, so the reference
above is the largest scale that runs end-to-end without dropped
requests; a full 2520-sample reference should be scheduled as a chunked
overnight job that restarts the server every ~100 requests rather than
as one contiguous run. Prompt length is high (~10k tokens mean) because
each Video-MME prompt carries the dense per-frame vision-placeholder
tokens — that is why the launch above pins
``--thinker-max-seq-len 32768``. The first 100 samples are all in the
``duration=short`` / ``domain=Knowledge`` prefix because the dataset is
not pre-shuffled; the accuracy above is a slice, not a stratified
estimator of full-set accuracy.
"""


from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.benchmarker.runner import BenchmarkRunner, RunConfig
from benchmarks.benchmarker.utils import save_json_results, wait_for_service
from benchmarks.dataset.videomme import load_videomme_samples
from benchmarks.metrics.performance import compute_speed_metrics
from benchmarks.tasks.tts import print_speed_summary
from benchmarks.tasks.video_understanding import (
    compute_videomme_metrics,
    make_videomme_send_fn,
    print_videomme_accuracy_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class VideoMMEEvalConfig:
    model: str
    split: str = "test"
    base_url: str | None = None
    host: str = "localhost"
    port: int = 8000
    max_samples: int | None = None
    max_tokens: int = 256
    temperature: float = 0.0
    output_dir: str | None = None
    max_concurrency: int = 1
    warmup: int = 0
    request_rate: float = float("inf")
    disable_tqdm: bool = False
    repo_id: str | None = None


def _build_base_url(config: VideoMMEEvalConfig) -> str:
    return config.base_url or f"http://{config.host}:{config.port}"


async def run_videomme_eval(config: VideoMMEEvalConfig) -> dict:
    base_url = _build_base_url(config)
    api_url = f"{base_url}/v1/chat/completions"

    samples = load_videomme_samples(
        repo_id=config.repo_id,
        split=config.split,
        max_samples=config.max_samples,
    )
    logger.info("Prepared %d Video-MME samples", len(samples))

    send_fn = make_videomme_send_fn(
        config.model,
        api_url,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    runner = BenchmarkRunner(
        RunConfig(
            max_concurrency=config.max_concurrency,
            request_rate=config.request_rate,
            warmup=config.warmup,
            disable_tqdm=config.disable_tqdm,
        )
    )
    request_results = await runner.run(samples, send_fn)

    summary, per_sample = compute_videomme_metrics(samples, request_results)
    speed = compute_speed_metrics(request_results, wall_clock_s=runner.wall_clock_s)
    results = {
        "summary": summary,
        "speed": speed,
        "config": {
            "model": config.model,
            "base_url": base_url,
            "repo_id": config.repo_id,
            "split": config.split,
            "max_samples": config.max_samples,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "max_concurrency": config.max_concurrency,
            "warmup": config.warmup,
        },
        "per_sample": per_sample,
    }

    if config.output_dir:
        save_json_results(results, config.output_dir, "videomme_results.json")

    return results


async def benchmark(args: argparse.Namespace) -> dict:
    config = VideoMMEEvalConfig(
        model=args.model,
        repo_id=args.repo_id,
        split=args.split,
        base_url=args.base_url,
        host=args.host,
        port=args.port,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        output_dir=args.output_dir,
        max_concurrency=args.max_concurrency,
        warmup=args.warmup,
        request_rate=args.request_rate,
        disable_tqdm=args.disable_tqdm,
    )
    results = await run_videomme_eval(config)
    print_videomme_accuracy_summary(results["summary"], config.model)
    print_speed_summary(
        results["speed"],
        config.model,
        config.max_concurrency,
        title="Video-MME Speed",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Video-MME benchmark for video understanding models."
    )
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="qwen3-omni")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "HuggingFace dataset repo for Video-MME. "
            "Defaults to zhaochenyang20/Video_MME."
        ),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="results/videomme")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--disable-tqdm", action="store_true")
    args = parser.parse_args()

    wait_for_service(args.base_url or f"http://{args.host}:{args.port}")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
