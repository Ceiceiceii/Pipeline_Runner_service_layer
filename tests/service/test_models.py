"""Submit-time chain validation and idempotency keys."""

from __future__ import annotations

import pytest

from pipeline_kit.schemas import Image
from service.models import (
    PRESETS,
    ChainValidationError,
    derived_idempotency_key,
    resolve_chain,
    validate_chain,
)


def test_presets_are_valid_chains():
    for chain in PRESETS.values():
        validate_chain(chain)


def test_resolve_preset_and_unknown_preset():
    assert resolve_chain("cutout") == ("segment", "remove_bg")
    with pytest.raises(ChainValidationError):
        resolve_chain("does-not-exist")


def test_unknown_step_rejected_by_name():
    with pytest.raises(ChainValidationError) as excinfo:
        validate_chain(("segment", "sharpen"))
    assert "sharpen" in str(excinfo.value)


def test_incompatible_order_names_the_exact_step():
    # generate_multiview needs a Cutout, which only remove_bg produces.
    with pytest.raises(ChainValidationError) as excinfo:
        validate_chain(("segment", "generate_multiview"))
    problems = excinfo.value.problems
    assert len(problems) == 1
    assert "generate_multiview" in problems[0]
    assert "Cutout" in problems[0]


def test_step_missing_earlier_dependency():
    # remove_bg needs the Mask from segment.
    with pytest.raises(ChainValidationError):
        validate_chain(("remove_bg",))


def test_derived_idempotency_key_is_stable_and_input_sensitive():
    image = Image(id="img-1")
    chain = PRESETS["full"]
    assert derived_idempotency_key(chain, image) == derived_idempotency_key(
        chain, image
    )
    assert derived_idempotency_key(chain, image) != derived_idempotency_key(
        PRESETS["cutout"], image
    )
    assert derived_idempotency_key(chain, image) != derived_idempotency_key(
        chain, Image(id="img-2")
    )
