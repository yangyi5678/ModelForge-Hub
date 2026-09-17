from __future__ import annotations

import pytest
from pydantic import ValidationError

from training_agent.config import SearchParam


def test_search_space_validation() -> None:
    SearchParam(type="float", low=1e-5, high=1e-3, log=True)
    SearchParam(type="categorical", choices=[8, 16])
    with pytest.raises(ValidationError):
        SearchParam(type="float", low=1, high=1)
    with pytest.raises(ValidationError):
        SearchParam(type="float", low=0, high=1, log=True)
    with pytest.raises(ValidationError):
        SearchParam(type="categorical", choices=[])
