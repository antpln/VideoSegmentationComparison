"""Registry utilities for model/prompt runners to enable modular expansion.

Each model family (e.g., sam2, edgetam, future models) exposes a dict mapping
prompt type strings (e.g., "points", "bbox") to a callable runner with the
standard signature used throughout the benchmark. New models can register
themselves by calling ``register_model_family(name, runners_dict)`` from their
module's import side-effect or an initialization block.

The benchmark core will consult this registry to resolve a (model_name,prompt)
pair to a concrete runner. This removes the need to special‑case model names
in the main pipeline when adding future methods.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

# Runner type kept loose to avoid circular typing with numpy / Path imports.
RunnerType = Callable[..., dict]

_MODEL_FAMILIES: Dict[str, Dict[str, RunnerType]] = {}


def register_model_family(family: str, runners: Dict[str, RunnerType]) -> None:
    """Register or overwrite a model family.

    Parameters
    ----------
    family: Normalized family key (e.g., "sam2", "edgetam").
    runners: Mapping of prompt type ("points", "bbox", etc.) to runner callables.
    """
    _MODEL_FAMILIES[family] = runners


def get_runner(family: str, prompt: str) -> Optional[RunnerType]:
    fam = _MODEL_FAMILIES.get(family)
    if fam is None:
        return None
    return fam.get(prompt)


def list_families() -> Dict[str, Dict[str, RunnerType]]:
    return dict(_MODEL_FAMILIES)
