from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


TEXT_CLASSES = {
    "article",
    "headline",
    "author",
    "image_caption",
    "newspaper_header",
    "table",
}


class BBox(BaseModel):
    id: int
    cls: Literal[
        "article",
        "headline",
        "author",
        "image_caption",
        "newspaper_header",
        "table",
    ] = Field(alias="class")
    bbox: dict[str, int]
    legibility: str

    @field_validator("bbox")
    @classmethod
    def _bbox_keys(cls, v: dict[str, int]) -> dict[str, int]:
        required = {"x0", "y0", "x1", "y1"}
        if set(v) != required:
            raise ValueError(f"bbox must have keys {required}")
        if not (v["x0"] < v["x1"] and v["y0"] < v["y1"]):
            raise ValueError("bbox coordinates must satisfy x0<x1 and y0<y1")
        return v

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return (int(self.bbox["x0"]), int(self.bbox["y0"]), int(self.bbox["x1"]), int(self.bbox["y1"]))


class BoxResult(BaseModel):
    id: int
    cls: Literal[
        "article",
        "headline",
        "author",
        "image_caption",
        "newspaper_header",
        "table",
    ] = Field(alias="class")
    bbox: dict[str, int]
    legibility: str
    status: Literal["ok", "unreadable", "error"]
    transcript: str | None
    model: str
    prompt: str
    attempts: int
    duration_ms: float
    error: dict | None = None

    @field_validator("bbox")
    @classmethod
    def _bbox_keys(cls, v: dict[str, int]) -> dict[str, int]:
        required = {"x0", "y0", "x1", "y1"}
        if set(v) != required:
            raise ValueError(f"bbox must have keys {required}")
        if not (v["x0"] < v["x1"] and v["y0"] < v["y1"]):
            raise ValueError("bbox coordinates must satisfy x0<x1 and y0<y1")
        return v

    model_config = {"populate_by_name": True}

    @field_validator("transcript")
    @classmethod
    def _transcript_rules(cls, v: str | None, info):
        status = info.data.get("status")
        if status == "ok":
            if not v or not isinstance(v, str) or not v.strip():
                raise ValueError("ok status requires non-empty transcript")
        if status == "unreadable":
            if v is None:
                return ""
            if v != "":
                raise ValueError("unreadable status requires empty transcript")
        if status == "error":
            if v is not None:
                # we allow None or empty on error; keep as-is
                return v
        return v


class PageResult(BaseModel):
    page_id: str
    png_path: str
    layout_path: str
    model: str
    prompt: str
    started_at: str
    finished_at: str
    boxes: list[BoxResult]

    @field_validator("boxes")
    @classmethod
    def _box_ids_unique(cls, v: list[BoxResult]) -> list[BoxResult]:
        ids = [b.id for b in v]
        if len(ids) != len(set(ids)):
            raise ValueError("box ids must be unique within a page")
        if not v:
            raise ValueError("page must contain at least one box result")
        return v

    model_config = {"populate_by_name": True}
