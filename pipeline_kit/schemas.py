"""Typed Pydantic input/output schemas for the four pipeline steps.

There are no real images or meshes here — just lightweight typed stand-ins. Two
properties make them useful as a substrate:

* **Provenance.** Each output records the id(s) of the input it came from, so a
  completed chain is a readable data-flow graph.
* **Content-addressed ids.** An output's id is derived from its inputs (see
  :func:`pipeline_kit.determinism.content_id`), so re-running a step on the same
  input yields an output with an identical id — a natural idempotency key.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

NUM_VIEWS = 8  # generate_multiview always emits exactly this many views


class Image(BaseModel):
    """An input photograph to run through the pipeline."""

    id: str
    width: int = 1024
    height: int = 1024


class Mask(BaseModel):
    """A segmentation mask produced by ``segment``."""

    id: str
    image_id: str


class Cutout(BaseModel):
    """A background-removed cutout produced by ``remove_bg``."""

    id: str
    image_id: str
    mask_id: str


class View(BaseModel):
    """A single rendered viewpoint within a :class:`MultiviewResult`."""

    id: str
    index: int
    azimuth: float


class MultiviewResult(BaseModel):
    """Exactly eight generated views produced by ``generate_multiview``."""

    id: str
    cutout_id: str
    views: list[View]

    @field_validator("views")
    @classmethod
    def _exactly_eight(cls, value: list[View]) -> list[View]:
        if len(value) != NUM_VIEWS:
            raise ValueError(f"expected exactly {NUM_VIEWS} views, got {len(value)}")
        return value


class Mesh(BaseModel):
    """A 3D mesh fitted to a last, produced by ``fit_to_last``."""

    id: str
    views_id: str
    vertex_count: int
    face_count: int
