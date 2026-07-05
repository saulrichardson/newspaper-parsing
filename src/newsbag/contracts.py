from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SourceRef:
    source_system: str
    source_id: str
    source_url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseInputPage:
    page_id: str
    image_path: str
    issue_id: str = ""
    page_number: int | None = None
    source: SourceRef | None = None
    checksum_sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PageProfile:
    page_id: str
    width: int
    height: int
    aspect_ratio: float
    mean_luma: float
    dark_pixel_share: float
    estimated_complexity: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeInfo:
    seconds: float
    resource_class: str
    status: str
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NormalizedRegion:
    region_id: str
    bbox_xyxy: list[float]
    label: str
    confidence: float
    source_model: str
    text: str = ""
    reading_order: int | None = None
    provenance: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelOutput:
    page_id: str
    model_id: str
    regions: list[NormalizedRegion]
    runtime: RuntimeInfo
    profile: PageProfile | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FusedPage:
    page_id: str
    regions: list[NormalizedRegion]
    transcript: str
    model_ids: list[str]
    disagreement_score: float
    quality: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunBundle:
    run_id: str
    run_dir: str
    input_manifest: str
    profile: str
    page_count: int
    model_ids: list[str]
    outputs: dict[str, str]
    performance: dict[str, Any]
    provenance: dict[str, Any]


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(raw) for key, raw in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(raw) for raw in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
            count += 1
    return count


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_parse_input_manifest(path: Path) -> list[ParseInputPage]:
    pages: list[ParseInputPage] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            source_raw = row.get("source")
            source = None
            if isinstance(source_raw, dict):
                source = SourceRef(
                    source_system=str(source_raw.get("source_system") or ""),
                    source_id=str(source_raw.get("source_id") or ""),
                    source_url=str(source_raw.get("source_url") or ""),
                    metadata=dict(source_raw.get("metadata") or {}),
                )
            page_id = str(row.get("page_id") or "").strip()
            image_path = str(row.get("image_path") or "").strip()
            if not page_id:
                raise ValueError(f"manifest row {line_number} is missing page_id")
            if not image_path:
                raise ValueError(f"manifest row {line_number} is missing image_path")
            page_number_raw = row.get("page_number")
            pages.append(
                ParseInputPage(
                    page_id=page_id,
                    image_path=image_path,
                    issue_id=str(row.get("issue_id") or ""),
                    page_number=int(page_number_raw) if page_number_raw not in (None, "") else None,
                    source=source,
                    checksum_sha256=str(row.get("checksum_sha256") or ""),
                    metadata=dict(row.get("metadata") or {}),
                )
            )
    return pages
