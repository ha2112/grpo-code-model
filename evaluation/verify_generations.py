#!/usr/bin/env python3
"""Check recovered generations for structural errors and duplicate outputs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


ROUTES = ("venus", "absolute", "probe", "venus-probe")
SUITES = ("codeforces", "venus")
REQUIRED_FIELDS = {
    "route",
    "suite",
    "example_id",
    "objective",
    "completion_index",
    "seed",
    "prompt_tokens",
    "output_tokens",
    "output",
}
FORMAT_PATTERN = re.compile(
    r"\A\s*(?:<thinking>.*?</thinking>\s*)?<solution>.*?</solution>\s*\Z",
    re.DOTALL,
)
CODE_PATTERN = re.compile(
    r"<solution>\s*```(?:python|python3)?\s*(.*?)(?:```|</solution>|\Z)",
    re.DOTALL,
)


def output_digest(text: str) -> str:
    normalized = "\n".join(line.rstrip() for line in text.strip().splitlines())
    return hashlib.sha256(normalized.encode()).hexdigest()


def manifest_ids(path: Path) -> dict[str, set[str]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return {
        suite: {str(item["example_id"]) for item in manifest.get(suite, [])}
        for suite in SUITES
    }


def inspect_generations(path: Path, manifest_path: Path, num_generations: int) -> dict:
    selected = manifest_ids(manifest_path)
    expected = {
        (route, suite, example_id, completion)
        for route in ROUTES
        for suite in SUITES
        for example_id in selected[suite]
        for completion in range(num_generations)
    }

    errors = []
    seen: dict[tuple, int] = {}
    records = []
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(raw_lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: malformed JSON ({exc.msg})")
            continue
        if not isinstance(record, dict):
            errors.append(f"line {line_number}: record is not an object")
            continue
        missing_fields = sorted(REQUIRED_FIELDS - record.keys())
        if missing_fields:
            errors.append(f"line {line_number}: missing fields {missing_fields}")
            continue
        try:
            key = (
                str(record["route"]),
                str(record["suite"]),
                str(record["example_id"]),
                int(record["completion_index"]),
            )
            record["prompt_tokens"] = int(record["prompt_tokens"])
            record["output_tokens"] = int(record["output_tokens"])
        except (TypeError, ValueError) as exc:
            errors.append(f"line {line_number}: invalid numeric field ({exc})")
            continue
        if key in seen:
            errors.append(
                f"line {line_number}: duplicate key {key}; first seen at line {seen[key]}"
            )
            continue
        seen[key] = line_number
        if key[0] not in ROUTES or key[1] not in SUITES:
            errors.append(f"line {line_number}: unknown route or suite in {key}")
        if not 0 <= key[3] < num_generations:
            errors.append(f"line {line_number}: invalid completion index {key[3]}")
        if not isinstance(record["output"], str) or not record["output"].strip():
            errors.append(f"line {line_number}: empty output for {key}")
            continue
        if record["prompt_tokens"] <= 0 or record["output_tokens"] <= 0:
            errors.append(f"line {line_number}: non-positive token count for {key}")
        record["_key"] = key
        record["_digest"] = output_digest(record["output"])
        records.append(record)

    keys = set(seen)
    unexpected = sorted(keys - expected)
    if unexpected:
        errors.append(f"{len(unexpected)} unexpected keys; first: {unexpected[0]}")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    cross_route: dict[tuple, dict[str, str]] = defaultdict(dict)
    for record in records:
        route, suite, example_id, completion = record["_key"]
        groups[(route, suite, example_id)].append(record)
        cross_route[(suite, example_id, completion)][route] = record["_digest"]

    duplicate_groups = []
    duplicate_completions = 0
    for group, items in groups.items():
        counts = Counter(item["_digest"] for item in items)
        extras = sum(count - 1 for count in counts.values() if count > 1)
        if extras:
            duplicate_groups.append((group, extras))
            duplicate_completions += extras

    route_collisions = {}
    for left, right in itertools.combinations(ROUTES, 2):
        comparable = 0
        identical = 0
        for digests in cross_route.values():
            if left in digests and right in digests:
                comparable += 1
                identical += digests[left] == digests[right]
        route_collisions[(left, right)] = (identical, comparable)

    exact_format = 0
    extractable = 0
    valid_python = 0
    max_length = 0
    for record in records:
        text = record["output"]
        exact_format += bool(FORMAT_PATTERN.fullmatch(text))
        match = CODE_PATTERN.search(text)
        if match and match.group(1).strip():
            extractable += 1
            try:
                ast.parse(match.group(1).strip())
                valid_python += 1
            except SyntaxError:
                pass
        max_length += record["output_tokens"] >= 1024

    return {
        "records": records,
        "expected": expected,
        "missing": sorted(expected - keys),
        "errors": errors,
        "duplicate_groups": duplicate_groups,
        "duplicate_completions": duplicate_completions,
        "route_collisions": route_collisions,
        "exact_format": exact_format,
        "extractable": extractable,
        "valid_python": valid_python,
        "max_length": max_length,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generations", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="sample manifest; defaults to sample_manifest.json beside generations",
    )
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.generations.with_name("sample_manifest.json")
    report = inspect_generations(args.generations, manifest, args.num_generations)
    total = len(report["records"])
    counts = Counter(
        (record["route"], record["suite"])
        for record in report["records"]
    )

    print(f"Records: {total}/{len(report['expected'])}")
    for route in ROUTES:
        print(
            f"  {route}: codeforces={counts[(route, 'codeforces')]}, "
            f"venus={counts[(route, 'venus')]}"
        )
    print(f"Missing expected records: {len(report['missing'])}")
    print(f"Exact response format: {report['exact_format']}/{total}")
    print(f"Extractable Python: {report['extractable']}/{total}")
    print(f"Syntactically valid Python: {report['valid_python']}/{total}")
    print(f"Responses at 1024-token limit: {report['max_length']}/{total}")
    print(
        "Within-prompt duplicate completions: "
        f"{report['duplicate_completions']} across {len(report['duplicate_groups'])} groups"
    )
    for group, extras in report["duplicate_groups"][:10]:
        print(f"  duplicate: {group}, extra copies={extras}")

    print("Cross-route identical outputs:")
    for (left, right), (identical, comparable) in report["route_collisions"].items():
        rate = identical / comparable if comparable else 0.0
        print(f"  {left} vs {right}: {identical}/{comparable} ({rate:.1%})")

    for error in report["errors"]:
        print(f"ERROR: {error}")
    if report["missing"]:
        print(f"First missing key: {report['missing'][0]}")

    failed = bool(report["errors"]) or (args.require_complete and bool(report["missing"]))
    print("Integrity: FAILED" if failed else "Integrity: OK")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
