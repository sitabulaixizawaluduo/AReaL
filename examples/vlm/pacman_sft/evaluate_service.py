# SPDX-License-Identifier: Apache-2.0

"""Benchmark a served Pacman SFT checkpoint on held-out decision frames."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
import mimetypes
import random
import re
import time
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from pacman_dataset import (
    LETTERS,
    SYSTEM_PROMPT,
    load_pacman_decisions,
    render_question,
)

from areal.utils import logging

logger = logging.getLogger("PacmanSFTServiceEval")


@dataclass(frozen=True, slots=True)
class EvalCase:
    index: int
    frame_path: Path
    frame_relative_path: str
    seed: int
    sample_id: int
    actions: tuple[str, ...]
    expected_letter: str
    expected_action: str


@dataclass(slots=True)
class EvalResult:
    index: int
    frame: str
    seed: int
    sample_id: int
    actions: tuple[str, ...]
    expected_letter: str
    expected_action: str
    raw_output: str | None
    exact_letter: str | None
    parsed_letter: str | None
    predicted_action: str | None
    strict_correct: bool
    parsed_correct: bool
    latency_seconds: float
    attempts: int
    status_code: int | None
    response_id: str | None
    usage: dict[str, Any] | None
    error: str | None


class ServiceResponseError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Pacman dataset root containing */records.jsonl and frame images.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help=(
            "OpenAI-compatible base URL, usually ending in /v1. A full "
            "/chat/completions URL is also accepted."
        ),
    )
    parser.add_argument("--model", required=True, help="Served model name.")
    credentials = parser.add_mutually_exclusive_group()
    credentials.add_argument(
        "--api-key",
        default="EMPTY",
        help="Bearer token. Defaults to EMPTY for trusted internal services.",
    )
    credentials.add_argument(
        "--api-key-file",
        type=Path,
        help="Read the bearer token from this file instead of the command line.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "valid", "test"),
        default="validation",
    )
    parser.add_argument("--split-modulus", type=int, default=10)
    parser.add_argument("--validation-remainder", type=int, default=0)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Number of examples to evaluate; 0 evaluates the whole split.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=1,
        help="Seed used to select a deterministic subset.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=1,
        help="Must match the SFT option-shuffling seed for exact comparability.",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retries after the first attempt for transient HTTP failures.",
    )
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--generation-seed", type=int, default=1)
    parser.add_argument(
        "--prompt-mode",
        choices=("train-like", "system-user"),
        default="train-like",
        help=(
            "train-like keeps the SFT text around the image in one user message; "
            "system-user uses conventional system and user roles."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Ask the service chat template to enable Qwen thinking.",
    )
    parser.add_argument(
        "--omit-chat-template-kwargs",
        action="store_true",
        help="Do not send chat_template_kwargs to services that reject it.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory. Defaults to a timestamped local directory.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if args.concurrency <= 0:
        parser.error("--concurrency must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.retries < 0:
        parser.error("--retries must be non-negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.progress_every <= 0:
        parser.error("--progress-every must be positive")
    return args


def _chat_completions_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def _load_api_key(args: argparse.Namespace) -> str:
    if args.api_key_file is None:
        return str(args.api_key)
    key = args.api_key_file.expanduser().read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file is empty: {args.api_key_file}")
    return key


def _select_cases(args: argparse.Namespace) -> list[EvalCase]:
    data_root = args.data_root.expanduser().resolve()
    decisions = load_pacman_decisions(
        root=data_root,
        split=args.split,
        split_modulus=args.split_modulus,
        validation_remainder=args.validation_remainder,
        shards=None,
    )
    if args.max_samples and args.max_samples < len(decisions):
        random.Random(args.sample_seed).shuffle(decisions)
        decisions = decisions[: args.max_samples]

    cases: list[EvalCase] = []
    for index, decision in enumerate(decisions):
        permutation = list(range(len(decision.actions)))
        random.Random(args.shuffle_seed + decision.sample_id).shuffle(permutation)
        actions = tuple(decision.actions[i] for i in permutation)
        answer_position = permutation.index(decision.teacher_action)
        try:
            relative_path = str(decision.frame_path.relative_to(data_root))
        except ValueError:
            relative_path = str(decision.frame_path)
        cases.append(
            EvalCase(
                index=index,
                frame_path=decision.frame_path,
                frame_relative_path=relative_path,
                seed=decision.seed,
                sample_id=decision.sample_id,
                actions=actions,
                expected_letter=LETTERS[answer_position],
                expected_action=actions[answer_position],
            )
        )
    return cases


def _image_data_uri(frame_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(frame_path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"
    encoded = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _build_messages(
    actions: Sequence[str], image_data_uri: str, prompt_mode: str
) -> list[dict[str, Any]]:
    question = render_question(actions)
    image_part = {
        "type": "image_url",
        "image_url": {"url": image_data_uri, "detail": "auto"},
    }
    if prompt_mode == "train-like":
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{SYSTEM_PROMPT}\n\n<state>\n",
                    },
                    image_part,
                    {
                        "type": "text",
                        "text": f"\n</state>\n\n{question}",
                    },
                ],
            }
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                image_part,
                {"type": "text", "text": question},
            ],
        },
    ]


def _response_text(payload: dict[str, Any]) -> tuple[str, str | None]:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ServiceResponseError(
            "Response does not contain choices[0].message.content"
        ) from exc
    response_id = payload.get("id")
    response_id = response_id if isinstance(response_id, str) else None
    if isinstance(content, str):
        return content, response_id
    if isinstance(content, list):
        text_parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        if text_parts:
            return "".join(text_parts), response_id
    raise ServiceResponseError("Response message content is not text")


def _parse_letters(
    raw_output: str, actions: Sequence[str]
) -> tuple[str | None, str | None]:
    allowed = LETTERS[: len(actions)]
    stripped = raw_output.strip()
    exact = stripped if len(stripped) == 1 and stripped in allowed else None
    matches = re.findall(rf"(?<![A-Z])([{re.escape(allowed)}])(?![A-Z])", stripped)
    parsed = matches[0] if matches else None
    return exact, parsed


def _request_payload(args: argparse.Namespace, case: EvalCase, image_uri: str) -> dict:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": _build_messages(case.actions, image_uri, args.prompt_mode),
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "seed": args.generation_seed,
        "stream": False,
    }
    if not args.omit_chat_template_kwargs:
        payload["chat_template_kwargs"] = {"enable_thinking": args.enable_thinking}
    return payload


def _should_retry(status_code: int) -> bool:
    return status_code in {408, 409, 425, 429} or status_code >= 500


async def _evaluate_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    args: argparse.Namespace,
    case: EvalCase,
) -> EvalResult:
    started = 0.0
    attempts = 0
    status_code: int | None = None
    raw_output: str | None = None
    response_id: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None

    async with semaphore:
        started = time.perf_counter()
        try:
            image_uri = await asyncio.to_thread(_image_data_uri, case.frame_path)
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            payload = _request_payload(args, case, image_uri)
            for attempt in range(args.retries + 1):
                attempts = attempt + 1
                try:
                    response = await client.post(url, json=payload)
                    status_code = response.status_code
                    if _should_retry(status_code) and attempt < args.retries:
                        await asyncio.sleep(min(2**attempt, 8))
                        continue
                    if status_code >= 400:
                        raise ServiceResponseError(
                            f"HTTP {status_code}: {response.text[:500]}",
                            status_code=status_code,
                        )
                    try:
                        response_payload = response.json()
                    except ValueError as exc:
                        raise ServiceResponseError(
                            "Service returned a non-JSON response",
                            status_code=status_code,
                        ) from exc
                    raw_output, response_id = _response_text(response_payload)
                    response_usage = response_payload.get("usage")
                    usage = response_usage if isinstance(response_usage, dict) else None
                    break
                except httpx.RequestError as exc:
                    if attempt < args.retries:
                        await asyncio.sleep(min(2**attempt, 8))
                        continue
                    error = f"{type(exc).__name__}: {exc}"
                except ServiceResponseError as exc:
                    status_code = exc.status_code or status_code
                    error = f"{type(exc).__name__}: {exc}"
                    break

    exact_letter: str | None = None
    parsed_letter: str | None = None
    predicted_action: str | None = None
    if raw_output is not None:
        exact_letter, parsed_letter = _parse_letters(raw_output, case.actions)
        if parsed_letter is not None:
            predicted_action = case.actions[LETTERS.index(parsed_letter)]
    latency = time.perf_counter() - started
    return EvalResult(
        index=case.index,
        frame=case.frame_relative_path,
        seed=case.seed,
        sample_id=case.sample_id,
        actions=case.actions,
        expected_letter=case.expected_letter,
        expected_action=case.expected_action,
        raw_output=raw_output,
        exact_letter=exact_letter,
        parsed_letter=parsed_letter,
        predicted_action=predicted_action,
        strict_correct=exact_letter == case.expected_letter,
        parsed_correct=parsed_letter == case.expected_letter,
        latency_seconds=latency,
        attempts=attempts,
        status_code=status_code,
        response_id=response_id,
        usage=usage,
        error=error,
    )


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(results: Sequence[EvalResult], elapsed: float) -> dict[str, Any]:
    total = len(results)
    succeeded = [result for result in results if result.error is None]
    strict_valid = [result for result in succeeded if result.exact_letter is not None]
    parsed_valid = [result for result in succeeded if result.parsed_letter is not None]
    latencies = [result.latency_seconds for result in succeeded]

    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    predicted_letters: Counter[str] = Counter()
    expected_letters: Counter[str] = Counter()
    for result in results:
        expected_letters[result.expected_letter] += 1
        if result.error is not None:
            predicted = "ERROR"
        elif result.exact_letter is None:
            predicted = "INVALID"
        else:
            predicted = result.exact_letter
        predicted_letters[predicted] += 1
        confusion[result.expected_letter][predicted] += 1

    return {
        "total": total,
        "request_successes": len(succeeded),
        "request_errors": total - len(succeeded),
        "request_success_rate": _ratio(len(succeeded), total),
        "strict_valid_outputs": len(strict_valid),
        "strict_valid_output_rate": _ratio(len(strict_valid), total),
        "strict_teacher_correct": sum(result.strict_correct for result in results),
        "strict_teacher_accuracy": _ratio(
            sum(result.strict_correct for result in results), total
        ),
        "strict_teacher_accuracy_on_valid": _ratio(
            sum(result.strict_correct for result in strict_valid), len(strict_valid)
        ),
        "parsed_valid_outputs": len(parsed_valid),
        "parsed_valid_output_rate": _ratio(len(parsed_valid), total),
        "parsed_teacher_correct": sum(result.parsed_correct for result in results),
        "parsed_teacher_accuracy": _ratio(
            sum(result.parsed_correct for result in results), total
        ),
        "parsed_teacher_accuracy_on_valid": _ratio(
            sum(result.parsed_correct for result in parsed_valid), len(parsed_valid)
        ),
        "random_choice_accuracy": (
            sum(1 / len(result.actions) for result in results) / total
            if total
            else None
        ),
        "elapsed_seconds": elapsed,
        "requests_per_second": _ratio(total, elapsed),
        "latency_seconds": {
            "mean": _ratio(sum(latencies), len(latencies)),
            "p50": _percentile(latencies, 0.50),
            "p90": _percentile(latencies, 0.90),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
        "expected_letters": dict(sorted(expected_letters.items())),
        "predicted_letters": dict(sorted(predicted_letters.items())),
        "confusion_matrix": {
            expected: dict(sorted(row.items()))
            for expected, row in sorted(confusion.items())
        },
    }


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(f"pacman_sft_eval-{timestamp}").resolve()


def _saved_config(
    args: argparse.Namespace, cases: Sequence[EvalCase]
) -> dict[str, Any]:
    return {
        "data_root": str(args.data_root.expanduser().resolve()),
        "base_url": args.base_url,
        "model": args.model,
        "split": args.split,
        "split_modulus": args.split_modulus,
        "validation_remainder": args.validation_remainder,
        "requested_max_samples": args.max_samples,
        "selected_samples": len(cases),
        "sample_seed": args.sample_seed,
        "shuffle_seed": args.shuffle_seed,
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout_seconds,
        "retries": args.retries,
        "max_tokens": args.max_tokens,
        "generation_seed": args.generation_seed,
        "prompt_mode": args.prompt_mode,
        "enable_thinking": args.enable_thinking,
        "omit_chat_template_kwargs": args.omit_chat_template_kwargs,
    }


async def _run(args: argparse.Namespace) -> int:
    cases = _select_cases(args)
    if not cases:
        raise ValueError("The selected Pacman split contains no evaluation cases")
    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=False)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps(_saved_config(args, cases), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    url = _chat_completions_url(args.base_url)
    api_key = _load_api_key(args)
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = httpx.Timeout(args.timeout_seconds)
    limits = httpx.Limits(
        max_connections=args.concurrency,
        max_keepalive_connections=args.concurrency,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    completed = 0
    results: list[EvalResult] = []
    started = time.perf_counter()

    logger.info(
        "Evaluating %d Pacman decisions through %s with concurrency=%d",
        len(cases),
        url,
        args.concurrency,
    )
    with results_path.open("w", encoding="utf-8", buffering=1) as output_stream:

        async def evaluate_and_record(case: EvalCase) -> EvalResult:
            nonlocal completed
            result = await _evaluate_case(
                client=client,
                semaphore=semaphore,
                url=url,
                args=args,
                case=case,
            )
            async with write_lock:
                output_stream.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                completed += 1
                if completed % args.progress_every == 0 or completed == len(cases):
                    logger.info("Completed %d/%d requests", completed, len(cases))
            return result

        async with httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=limits,
            trust_env=False,
        ) as client:
            results = await asyncio.gather(
                *(evaluate_and_record(case) for case in cases)
            )

    elapsed = time.perf_counter() - started
    summary = _summarize(results, elapsed)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.info(
        "Evaluation summary:\n%s", json.dumps(summary, indent=2, sort_keys=True)
    )
    logger.info("Wrote results to %s", output_dir)
    return 0 if summary["request_successes"] else 2


def main() -> int:
    args = _parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
