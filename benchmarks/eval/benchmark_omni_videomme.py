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
    #    --encoder-mem-reserve 0.20 holds back ~28 GB of GPU memory for
    #    the co-located video encoder at peak activation, outside SGLang's
    #    static KV pool.
    python examples/run_qwen3_omni_server.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --port 8000 \
        --thinker-max-seq-len 32768 \
        --encoder-mem-reserve 0.20

    # 3. Run the full 2520-sample benchmark at concurrency 4
    python benchmarks/eval/benchmark_omni_videomme.py \
        --model qwen3-omni --port 8000 \
        --max-concurrency 4 --max-tokens 256


H200 Full-Set Reference Results

Reproducibility references for the FULL 2520-sample test split — NOT CI
thresholds. CI runs on the 50-sample subset videomme-ci-50 and keeps its
own thresholds in tests/test_model/test_qwen3_omni_videomme_ci.py.

Benchmark:  Video-MME | Dataset: lmms-lab/Video-MME test split
                       (2520 questions: 900 short + 900 medium + 720 long)
Hardware:   1 x H200 (thinker-only; speech disabled)
Launch:     --thinker-max-seq-len 32768 --encoder-mem-reserve 0.20
Bench:      --max-concurrency 4 --max-tokens 256
Run status: TBD — full-set pending re-run on main after task-5 merge.
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
