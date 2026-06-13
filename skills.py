"""
================================================================================
 Physics ToT — Skill Registry and ToT Integration Layer
================================================================================
 This module holds the ToT-specific layer: hard-rule checking, domain plugin
 templates, stage prompt contracts, and the public ``SKILL_REGISTRY`` with
 ``invoke_skill``/``search_skills`` lookup.

 The generic SymPy computation skills (mechanics, EM, quantum, thermo,
 relativity, optics, waves, fluids) are implemented in ``physics_skills.py``
 and re-exported here so registry entries and direct imports keep working.

 Benchmark fixtures live in ``benchmarks.py``, outside this registry, so the
 system under test cannot discover benchmark answers via skill search.

================================================================================
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sp
from sympy import Matrix, simplify, solve

from physics_skills import (
    _dimension_powers,
    aberrations,
    angular_momentum_eigenstates,
    bernoulli_equation,
    commutator,
    continuity_equation,
    dimensional_analysis,
    doppler_classical,
    effective_potential_analysis,
    em_wave_dispersion,
    error_propagation,
    euler_fluid_equation,
    euler_rigid_body_equations,
    fields_from_potentials,
    four_vector_inner_product,
    grating_equation,
    hamiltonian_equations,
    inertia_tensor,
    jones_calculus,
    lagrangian_equations,
    lorentz_boost_matrix,
    lorentz_transform_event,
    maxwell_equations_check,
    mirror_matrix,
    multi_slit_intensity,
    navier_stokes_check,
    noether_conservation,
    optical_system,
    partition_function,
    pauli_algebra,
    pauli_matrices,
    perturbation_first_order,
    poiseuille_flow,
    poynting_vector,
    ray_refraction_matrix,
    ray_translation_matrix,
    relativistic_energy_momentum,
    reynolds_number,
    scalar_laplacian,
    schrodinger_1d,
    single_slit_diffraction,
    sound_speed,
    special_functions,
    standing_wave_modes,
    statistical_distributions,
    stokes_drag,
    stokes_mueller,
    surface_tension,
    t,
    thermodynamic_partial,
    thermodynamic_potentials,
    thick_lens,
    thin_lens_matrix,
    vector_curl,
    vector_divergence,
    vector_gradient,
    velocity_addition,
    vorticity_and_stream,
)



def _dimension_maps_match(left: Dict[Any, sp.Expr], right: Dict[Any, sp.Expr]) -> bool:
    keys = set(left) | set(right)
    return all(sp.simplify(left.get(key, 0) - right.get(key, 0)) == 0 for key in keys)


def _resolve_rule_value(value: Any, known_vars: Dict[str, Any]) -> Any:
    if isinstance(value, str) and value in known_vars:
        return known_vars[value]
    return value


def _contains_pattern(items: Sequence[str], pattern: str) -> bool:
    return any(pattern in item for item in items)


def _boundary_condition_text(boundary_conditions: Dict[str, Any]) -> List[str]:
    return [f"{key}: {value}" for key, value in boundary_conditions.items()]


_META_TASK_SCOPE_STOPWORDS = {
    "a",
    "an",
    "and",
    "apply",
    "as",
    "at",
    "by",
    "compare",
    "compute",
    "define",
    "derive",
    "do",
    "each",
    "exactly",
    "for",
    "from",
    "identify",
    "in",
    "into",
    "is",
    "it",
    "local",
    "method",
    "methodology",
    "model",
    "next",
    "of",
    "on",
    "only",
    "or",
    "perform",
    "problem",
    "quantity",
    "approximation",
    "correction",
    "refine",
    "relation",
    "segment",
    "sequential",
    "single",
    "solve",
    "step",
    "target",
    "task",
    "the",
    "then",
    "to",
    "use",
    "using",
    "verify",
    "via",
    "with",
}


def _normalize_meta_scope_token(token: str) -> str:
    normalized = token.lower()
    if len(normalized) > 4 and normalized.endswith("ies"):
        return normalized[:-3] + "y"
    if len(normalized) > 5 and normalized.endswith("ing"):
        return normalized[:-3]
    if len(normalized) > 4 and normalized.endswith("ed"):
        return normalized[:-2]
    if len(normalized) > 4 and normalized.endswith("es"):
        return normalized[:-2]
    if len(normalized) > 3 and normalized.endswith("s"):
        return normalized[:-1]
    return normalized


def _meta_scope_keywords(text: str) -> List[str]:
    keywords: List[str] = []
    seen = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_\-]*", str(text)):
        normalized = _normalize_meta_scope_token(token)
        if len(normalized) < 3 or normalized in _META_TASK_SCOPE_STOPWORDS:
            continue
        if normalized in seen:
            continue
        keywords.append(normalized)
        seen.add(normalized)
    return keywords


def _meta_scope_step_score(candidate_tokens: set[str], step_text: str) -> float:
    step_tokens = _meta_scope_keywords(step_text)
    if not step_tokens:
        return 0.0
    overlap = sum(1 for token in step_tokens if token in candidate_tokens)
    denominator = max(1, min(4, len(step_tokens)))
    return overlap / denominator


def _contains_deferral_language(text: str) -> bool:
    lowered = str(text).lower()
    markers = (
        "defer",
        "deferred",
        "later",
        "next step",
        "next refinement",
        "pending",
        "leave",
        "future",
    )
    return any(marker in lowered for marker in markers)


def _contains_route_comparison_language(text: str) -> bool:
    lowered = str(text).lower()
    markers = (
        "compare",
        "comparison",
        "route a",
        "route b",
        "routes",
        "versus",
        " vs ",
        "alternative",
        "primary solution route",
        "proposed",
    )
    return any(marker in lowered for marker in markers)


def _meta_task_step_scope_diagnostics(
    *,
    thought_step: str,
    equations: Sequence[str],
    used_models: Sequence[str],
    boundary_conditions: Dict[str, Any],
    meta_task: Dict[str, Any],
    meta_task_progress: Dict[str, Any],
    enforce_scope: bool,
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "enforced": bool(enforce_scope and meta_task),
        "phase": "",
        "current_step": "",
        "current_step_guidance": "",
        "current_step_index": 0,
        "current_step_score": 0.0,
        "future_step_matches": [],
        "violations": [],
    }
    if not diagnostics["enforced"]:
        return diagnostics

    step_ordering = [str(item) for item in meta_task.get("step_ordering", []) if str(item).strip()]
    first_step = str(meta_task.get("first_step", "")).strip()
    fallback_step = step_ordering[0] if step_ordering else first_step
    total_steps = len(step_ordering)
    current_step_index = 0
    raw_step_index = meta_task_progress.get("current_step_index", 0)
    try:
        current_step_index = int(raw_step_index)
    except (TypeError, ValueError):
        current_step_index = 0
    if total_steps:
        current_step_index = max(0, min(current_step_index, total_steps - 1))

    current_step = str(meta_task_progress.get("current_step", "")).strip()
    if not current_step:
        current_step = step_ordering[current_step_index] if step_ordering else fallback_step
    phase = str(meta_task_progress.get("phase", "")).strip() or (
        "strategy_scan" if current_step_index == 0 else "incremental_refinement"
    )
    current_step_guidance = str(meta_task_progress.get("current_step_guidance", "")).strip() or current_step

    remaining_steps = meta_task_progress.get("remaining_steps")
    if not isinstance(remaining_steps, list):
        remaining_steps = step_ordering[current_step_index + 1 :] if step_ordering else []
    remaining_steps = [str(item) for item in remaining_steps if str(item).strip()]

    candidate_parts = [str(thought_step)]
    candidate_parts.extend(str(item) for item in equations)
    candidate_parts.extend(str(item) for item in used_models)
    candidate_parts.extend(_boundary_condition_text(boundary_conditions))
    candidate_tokens = set(_meta_scope_keywords(" ".join(candidate_parts)))

    comparative_strategy_scan = phase == "strategy_scan" and _contains_route_comparison_language(
        " ".join([str(thought_step), *(str(item) for item in equations), *(str(item) for item in used_models)])
    )

    future_match_parts: List[str] = []
    if not comparative_strategy_scan:
        if phase != "strategy_scan" or not _contains_deferral_language(thought_step):
            future_match_parts.append(str(thought_step))
        future_match_parts.extend(str(item) for item in equations)
        future_match_parts.extend(str(item) for item in used_models)
        future_match_parts.extend(_boundary_condition_text(boundary_conditions))
    future_match_tokens = set(_meta_scope_keywords(" ".join(future_match_parts)))

    current_step_score = _meta_scope_step_score(candidate_tokens, current_step_guidance)
    future_step_matches: List[Dict[str, Any]] = []
    future_step_threshold = 0.5 if phase == "strategy_scan" else 0.75
    for offset, step_text in enumerate(remaining_steps, start=1):
        score = _meta_scope_step_score(future_match_tokens, step_text)
        if score >= future_step_threshold:
            future_step_matches.append(
                {
                    "step_index": current_step_index + offset,
                    "step": step_text,
                    "score": round(score, 3),
                }
            )

    violations: List[str] = []
    min_current_step_score = 0.2 if phase == "strategy_scan" else 0.35
    if current_step_guidance and current_step_score < min_current_step_score:
        violations.append(
            f"Candidate does not stay on the current meta step: {current_step_guidance}"
        )
    if future_step_matches:
        future_labels = "; ".join(match["step"] for match in future_step_matches)
        violations.append(
            f"Candidate jumps ahead beyond the current meta step into future steps: {future_labels}"
        )

    diagnostics.update(
        {
            "phase": phase,
            "current_step": current_step,
            "current_step_guidance": current_step_guidance,
            "current_step_index": current_step_index,
            "current_step_score": round(current_step_score, 3),
            "comparative_strategy_scan": comparative_strategy_scan,
            "future_step_matches": future_step_matches,
            "violations": violations,
        }
    )
    return diagnostics


def _text_symbols(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)
    expanded = set(tokens)
    for token in tokens:
        parts = [part for part in re.split(r"_+", token) if part]
        expanded.update(parts)
        for part in parts:
            if len(part) > 1 and len(set(part)) == 1:
                expanded.add(part[0])
    return sorted(expanded)


def _boundary_condition_value_text(value: Any) -> str:
    if isinstance(value, dict) and "value" in value:
        return str(value["value"])
    return str(value)


def _boundary_condition_allowed_dependencies(value: Any) -> Optional[set[str]]:
    if not isinstance(value, dict) or "allowed_dependencies" not in value:
        return None
    allowed = value["allowed_dependencies"]
    if not isinstance(allowed, (list, tuple, set)):
        raise TypeError("boundary condition 'allowed_dependencies' must be a list, tuple, or set.")
    return {str(item) for item in allowed}


def _normalize_boundary_condition_key_text(key_text: str) -> tuple[str, bool]:
    if ":" not in key_text:
        return key_text, False
    label, remainder = key_text.split(":", 1)
    normalized_label = label.strip()
    normalized_remainder = remainder.strip()
    if not normalized_label or not normalized_remainder:
        return key_text, False
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9 ()_\-/]*", normalized_label):
        return normalized_remainder, True
    return key_text, False


def _semantic_boundary_condition_violations(
    equations: Sequence[str],
    boundary_conditions: Dict[str, Any],
    known_vars: Dict[str, Any],
) -> List[str]:
    equation_symbols = set()
    for equation in equations:
        equation_symbols.update(_text_symbols(str(equation)))
    equation_symbols.update(str(name) for name in known_vars)

    violations: List[str] = []
    for key, raw_value in boundary_conditions.items():
        key_text = str(key)
        normalized_key_text, has_descriptive_label = _normalize_boundary_condition_key_text(key_text)
        value_text = _boundary_condition_value_text(raw_value)
        value_symbols = set(_text_symbols(value_text))
        allowed_dependencies = _boundary_condition_allowed_dependencies(raw_value)

        if "=" in normalized_key_text:
            axis = normalized_key_text.split("=", 1)[0].strip()
            if not has_descriptive_label and axis and axis not in equation_symbols:
                violations.append(
                    f"Boundary condition axis is not grounded in equations or known variables: {axis}"
                )
            if allowed_dependencies is None and axis and axis in value_symbols:
                violations.append(
                    f"Boundary condition value depends on constrained axis: {key_text}"
                )
            if allowed_dependencies is not None:
                disallowed = sorted(symbol for symbol in value_symbols if symbol not in allowed_dependencies)
                if disallowed:
                    violations.append(
                        f"Boundary condition value uses disallowed dependencies for {key_text}: {', '.join(disallowed)}"
                    )
            continue

        key_symbols = set(_text_symbols(normalized_key_text))
        if key_symbols and not has_descriptive_label and not any(symbol in equation_symbols for symbol in key_symbols):
            violations.append(
                f"Boundary condition key is not grounded in equations or known variables: {key_text}"
            )

    return violations


def tot_hard_rule_check(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Generic hard-rule validation skill for ToT node candidates.

    Parameters
    ----------
    equations
        Candidate equations to validate.
    known_vars
        Known or derived variables available for rule checks.
    require_equations
        If ``True`` and no equations are present, the branch is vetoed.
    required_known_vars
        Variable names that must exist in ``known_vars``.
    required_equation_patterns
        String fragments that must appear in at least one equation.
    required_any_equation_patterns
        String fragments where at least one must appear in the candidate equations.
    forbidden_equation_patterns
        String fragments that must not appear in any equation.
    used_models
        Physical models or approximations currently attached to the node.
    required_models
        Exact model names that must appear in ``used_models``.
    forbidden_models
        Exact model names that must not appear in ``used_models``.
    required_model_patterns
        String fragments that must appear in at least one model name.
    required_any_model_patterns
        String fragments where at least one must appear in the active model list.
    forbidden_model_patterns
        String fragments that must not appear in any model name.
    boundary_conditions
        Initial or boundary conditions currently attached to the node.
    required_boundary_condition_keys
        Boundary-condition keys that must exist.
    forbidden_boundary_condition_keys
        Boundary-condition keys that must not exist.
    required_boundary_condition_patterns
        String fragments that must appear in at least one rendered boundary condition.
    required_any_boundary_condition_patterns
        String fragments where at least one must appear in the rendered boundary conditions.
    forbidden_boundary_condition_patterns
        String fragments that must not appear in any rendered boundary condition.
    required_boundary_conditions
        Exact boundary-condition key/value pairs that must match.
    forbidden_boundary_conditions
        Exact boundary-condition key/value pairs that must not match.
    semantic_boundary_checks
        If ``True`` (default), run semantic consistency checks on boundary-condition
        axes and value dependencies instead of relying only on exact or pattern matching.
    required_all_context_patterns
        String fragments that must all appear somewhere across equations, thought_step,
        models, known variables, meta-task text, or boundary-condition text.
    required_any_context_patterns
        String fragments where at least one must appear somewhere across equations,
        models, known variables, or boundary-condition text.
    forbidden_any_context_patterns
        String fragments that must not appear anywhere across equations, models,
        known variables, or boundary-condition text.
    dimension_equalities
        List of checks of the form ``{"left": ..., "right": ..., "label": ...}``.
        Each side may be a SymPy unit expression, a dimension expression, or the
        name of an entry in ``known_vars``.
    positive_var_names
        Variables that must be positive when their sign is decidable.
    nonzero_var_names
        Variables that must be nonzero when their value is decidable.
    finite_var_names
        Variables that must be finite when their value is decidable.
    custom_violations
        Extra violations supplied by the caller.

    Returns
    -------
    dict
        ``passed`` indicates whether the branch survives hard-rule checking.
        ``violations`` contains veto reasons. ``checked`` records the applied
        checks for later auditing.
    """

    validation_bundle = tot_validation_plugin_bundle(params)
    effective_params = _merge_validation_rule_params(
        validation_bundle.get("hard_rule_params", {}),
        _normalize_validation_rule_params(params),
    )
    for key, value in params.items():
        if key not in effective_params:
            effective_params[key] = value

    equations = [str(item) for item in effective_params.get("equations", [])]
    known_vars = dict(effective_params.get("known_vars", {}))
    used_models = [str(item) for item in effective_params.get("used_models", [])]
    boundary_conditions = {
        str(key): value for key, value in dict(effective_params.get("boundary_conditions", {})).items()
    }
    boundary_text = _boundary_condition_text(boundary_conditions)
    violations: List[str] = []
    dimension_results: List[Dict[str, Any]] = []
    meta_task = (
        dict(effective_params.get("meta_task", {}))
        if isinstance(effective_params.get("meta_task"), dict)
        else {}
    )
    meta_task_progress = (
        dict(effective_params.get("meta_task_progress", {}))
        if isinstance(effective_params.get("meta_task_progress"), dict)
        else {}
    )
    thought_step = str(effective_params.get("thought_step", ""))
    problem_context = (
        dict(effective_params.get("problem_context", {}))
        if isinstance(effective_params.get("problem_context"), dict)
        else {}
    )
    context_text = " ".join(
        [
            *equations,
            *used_models,
            *boundary_text,
            thought_step,
            str(problem_context.get("problem_statement", "")),
            str(problem_context.get("task", "")),
            str(problem_context.get("skill_query", "")),
            str(meta_task.get("objective", "")),
            str(meta_task.get("first_step", "")),
            *[str(item) for item in meta_task.get("step_ordering", [])],
            str(meta_task_progress.get("current_step", "")),
            str(meta_task_progress.get("current_step_guidance", "")),
            *[str(item) for item in meta_task_progress.get("previous_steps", [])],
            *[str(item) for item in meta_task_progress.get("remaining_steps", [])],
            *[str(key) for key in known_vars.keys()],
            *[str(value) for value in known_vars.values()],
        ]
    ).lower()

    if effective_params.get("require_equations", True) and not equations:
        violations.append("No candidate equations were provided for hard-rule checking.")

    required_known_vars = [str(name) for name in effective_params.get("required_known_vars", [])]
    for name in required_known_vars:
        if name not in known_vars:
            violations.append(f"Missing required variable: {name}")

    required_patterns = [str(pattern) for pattern in effective_params.get("required_equation_patterns", [])]
    for pattern in required_patterns:
        if not any(pattern in equation for equation in equations):
            violations.append(f"No equation matches required pattern: {pattern}")

    required_any_equation_patterns = [
        str(pattern) for pattern in effective_params.get("required_any_equation_patterns", [])
    ]
    if required_any_equation_patterns and not any(
        any(pattern in equation for equation in equations)
        for pattern in required_any_equation_patterns
    ):
        violations.append(
            "No equation matches any required pattern: " + " | ".join(required_any_equation_patterns)
        )

    forbidden_patterns = [str(pattern) for pattern in effective_params.get("forbidden_equation_patterns", [])]
    for pattern in forbidden_patterns:
        if any(pattern in equation for equation in equations):
            violations.append(f"Equation matches forbidden pattern: {pattern}")

    required_models = [str(model) for model in effective_params.get("required_models", [])]
    for model in required_models:
        if model not in used_models:
            violations.append(f"Missing required model: {model}")

    forbidden_models = [str(model) for model in effective_params.get("forbidden_models", [])]
    for model in forbidden_models:
        if model in used_models:
            violations.append(f"Forbidden model used: {model}")

    required_model_patterns = [str(pattern) for pattern in effective_params.get("required_model_patterns", [])]
    for pattern in required_model_patterns:
        if not _contains_pattern(used_models, pattern):
            violations.append(f"No model matches required pattern: {pattern}")

    required_any_model_patterns = [
        str(pattern) for pattern in effective_params.get("required_any_model_patterns", [])
    ]
    if required_any_model_patterns and not any(
        _contains_pattern(used_models, pattern)
        for pattern in required_any_model_patterns
    ):
        violations.append(
            "No model matches any required pattern: " + " | ".join(required_any_model_patterns)
        )

    forbidden_model_patterns = [str(pattern) for pattern in effective_params.get("forbidden_model_patterns", [])]
    for pattern in forbidden_model_patterns:
        if _contains_pattern(used_models, pattern):
            violations.append(f"Model matches forbidden pattern: {pattern}")

    required_boundary_condition_keys = [
        str(key) for key in effective_params.get("required_boundary_condition_keys", [])
    ]
    for key in required_boundary_condition_keys:
        if key not in boundary_conditions:
            violations.append(f"Missing required boundary condition key: {key}")

    forbidden_boundary_condition_keys = [
        str(key) for key in effective_params.get("forbidden_boundary_condition_keys", [])
    ]
    for key in forbidden_boundary_condition_keys:
        if key in boundary_conditions:
            violations.append(f"Forbidden boundary condition key present: {key}")

    required_boundary_condition_patterns = [
        str(pattern) for pattern in effective_params.get("required_boundary_condition_patterns", [])
    ]
    for pattern in required_boundary_condition_patterns:
        if not _contains_pattern(boundary_text, pattern):
            violations.append(f"No boundary condition matches required pattern: {pattern}")

    required_any_boundary_condition_patterns = [
        str(pattern)
        for pattern in effective_params.get("required_any_boundary_condition_patterns", [])
    ]
    if required_any_boundary_condition_patterns and not any(
        _contains_pattern(boundary_text, pattern)
        for pattern in required_any_boundary_condition_patterns
    ):
        violations.append(
            "No boundary condition matches any required pattern: "
            + " | ".join(required_any_boundary_condition_patterns)
        )

    forbidden_boundary_condition_patterns = [
        str(pattern) for pattern in effective_params.get("forbidden_boundary_condition_patterns", [])
    ]
    for pattern in forbidden_boundary_condition_patterns:
        if _contains_pattern(boundary_text, pattern):
            violations.append(f"Boundary condition matches forbidden pattern: {pattern}")

    required_all_context_patterns = [
        str(pattern) for pattern in effective_params.get("required_all_context_patterns", [])
    ]
    for pattern in required_all_context_patterns:
        if pattern.lower() not in context_text:
            violations.append(f"No context matches required pattern: {pattern}")

    required_any_context_patterns = [
        str(pattern) for pattern in effective_params.get("required_any_context_patterns", [])
    ]
    if required_any_context_patterns and not any(
        pattern.lower() in context_text for pattern in required_any_context_patterns
    ):
        violations.append(
            "No context matches any required pattern: " + " | ".join(required_any_context_patterns)
        )

    forbidden_any_context_patterns = [
        str(pattern) for pattern in effective_params.get("forbidden_any_context_patterns", [])
    ]
    for pattern in forbidden_any_context_patterns:
        if pattern.lower() in context_text:
            violations.append(f"Context matches forbidden pattern: {pattern}")

    required_boundary_conditions = {
        str(key): value for key, value in dict(effective_params.get("required_boundary_conditions", {})).items()
    }
    for key, value in required_boundary_conditions.items():
        if key not in boundary_conditions:
            violations.append(f"Missing required boundary condition: {key}")
            continue
        if boundary_conditions[key] != value:
            violations.append(
                f"Boundary condition mismatch for {key}: expected {value!r}, got {boundary_conditions[key]!r}"
            )

    forbidden_boundary_conditions = {
        str(key): value for key, value in dict(effective_params.get("forbidden_boundary_conditions", {})).items()
    }
    for key, value in forbidden_boundary_conditions.items():
        if key in boundary_conditions and boundary_conditions[key] == value:
            violations.append(f"Forbidden boundary condition present: {key} = {value!r}")

    semantic_boundary_checks = bool(effective_params.get("semantic_boundary_checks", True))
    semantic_boundary_violations: List[str] = []
    if semantic_boundary_checks and boundary_conditions:
        semantic_boundary_violations = _semantic_boundary_condition_violations(
            equations=equations,
            boundary_conditions=boundary_conditions,
            known_vars=known_vars,
        )
        violations.extend(semantic_boundary_violations)

    meta_task_scope = _meta_task_step_scope_diagnostics(
        thought_step=thought_step,
        equations=equations,
        used_models=used_models,
        boundary_conditions=boundary_conditions,
        meta_task=meta_task,
        meta_task_progress=meta_task_progress,
        enforce_scope=bool(effective_params.get("enforce_meta_task_step_scope", False)),
    )

    for item in effective_params.get("dimension_equalities", []):
        if not isinstance(item, dict):
            raise TypeError("Each dimension equality check must be a dictionary.")
        left = _resolve_rule_value(item["left"], known_vars)
        right = _resolve_rule_value(item["right"], known_vars)
        label = str(item.get("label", f"{item['left']} ~ {item['right']}"))
        left_map = _dimension_powers(left)
        right_map = _dimension_powers(right)
        passed = _dimension_maps_match(left_map, right_map)
        dimension_results.append(
            {
                "label": label,
                "left_dimensions": left_map,
                "right_dimensions": right_map,
                "passed": passed,
            }
        )
        if not passed:
            violations.append(f"Dimension mismatch: {label}")

    for name in effective_params.get("positive_var_names", []):
        if name not in known_vars:
            continue
        value = sp.sympify(known_vars[name])
        if value.is_positive is False:
            violations.append(f"Variable must be positive: {name}")

    for name in effective_params.get("nonzero_var_names", []):
        if name not in known_vars:
            continue
        value = sp.sympify(known_vars[name])
        if value.is_zero is True:
            violations.append(f"Variable must be nonzero: {name}")

    for name in effective_params.get("finite_var_names", []):
        if name not in known_vars:
            continue
        value = sp.sympify(known_vars[name])
        if value.is_finite is False:
            violations.append(f"Variable must be finite: {name}")

    custom_violations = [str(item) for item in effective_params.get("custom_violations", [])]
    violations.extend(custom_violations)

    return {
        "passed": not violations,
        "violations": violations,
        "checked": {
            "equation_count": len(equations),
            "known_var_count": len(known_vars),
            "used_models": used_models,
            "boundary_conditions": boundary_conditions,
            "required_known_vars": required_known_vars,
            "required_equation_patterns": required_patterns,
            "required_any_equation_patterns": required_any_equation_patterns,
            "forbidden_equation_patterns": forbidden_patterns,
            "required_models": required_models,
            "forbidden_models": forbidden_models,
            "required_model_patterns": required_model_patterns,
            "required_any_model_patterns": required_any_model_patterns,
            "forbidden_model_patterns": forbidden_model_patterns,
            "required_boundary_condition_keys": required_boundary_condition_keys,
            "forbidden_boundary_condition_keys": forbidden_boundary_condition_keys,
            "required_boundary_condition_patterns": required_boundary_condition_patterns,
            "required_any_boundary_condition_patterns": required_any_boundary_condition_patterns,
            "forbidden_boundary_condition_patterns": forbidden_boundary_condition_patterns,
            "required_all_context_patterns": required_all_context_patterns,
            "required_any_context_patterns": required_any_context_patterns,
            "forbidden_any_context_patterns": forbidden_any_context_patterns,
            "required_boundary_conditions": required_boundary_conditions,
            "forbidden_boundary_conditions": forbidden_boundary_conditions,
            "semantic_boundary_checks": semantic_boundary_checks,
            "semantic_boundary_violations": semantic_boundary_violations,
            "meta_task_step_scope": meta_task_scope,
            "dimension_equalities": dimension_results,
            "positive_var_names": [str(name) for name in effective_params.get("positive_var_names", [])],
            "nonzero_var_names": [str(name) for name in effective_params.get("nonzero_var_names", [])],
            "finite_var_names": [str(name) for name in effective_params.get("finite_var_names", [])],
            "validation_plugin_selection_mode": validation_bundle.get("selection_mode", "fallback"),
            "validation_plugins": [
                {
                    "name": validator.get("name", ""),
                    "label": validator.get("label", ""),
                    "skill_names": list(validator.get("skill_names", [])),
                    "summary": validator.get("summary", ""),
                    "hard_rule_params": dict(validator.get("hard_rule_params", {})),
                }
                for validator in validation_bundle.get("selected_validators", [])
            ],
        },
    }


TOT_DOMAIN_PLUGIN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "general-scientific": {
        "name": "general-scientific",
        "label": "General Scientific Reasoning",
        "summary": "Domain-agnostic planning bundle used when no stronger subject plugin is selected.",
        "knowledge_scope": [
            "governing relations and objectives",
            "constraints and admissibility conditions",
            "invariants, symmetries, or conserved structure",
            "boundary or initial conditions",
            "approximations, closures, and limiting regimes",
        ],
        "representative_formulas": [
            {
                "latex": r"f(x) = 0",
                "meaning": "Implicit governing relation or residual form.",
            },
            {
                "latex": r"\mathcal{C}(x) \le 0",
                "meaning": "Constraint or admissibility condition.",
            },
            {
                "latex": r"x = x_0 + \varepsilon \, \delta x",
                "meaning": "One-step correction or perturbative refinement.",
            },
        ],
        "route_seed_options": [
            {
                "label": "governing-relation route",
                "route_family": "governing-relation",
                "governing_models": ["Core governing relation"],
                "guidance": "Name one governing relation or dominant dependency only.",
                "correction_mode": "governing-law scan",
                "correction_target": "active governing relation",
            },
            {
                "label": "constraint route",
                "route_family": "constraint",
                "governing_models": ["Constraint relation"],
                "guidance": "State one structural constraint or admissibility rule only.",
                "correction_mode": "constraint-first scan",
                "correction_target": "active constraint",
            },
            {
                "label": "invariant route",
                "route_family": "invariant",
                "governing_models": ["Invariant structure"],
                "guidance": "Name one symmetry, invariant, or conserved quantity only.",
                "correction_mode": "invariant-first scan",
                "correction_target": "active invariant",
            },
            {
                "label": "closure route",
                "route_family": "closure",
                "governing_models": ["Closure assumption"],
                "guidance": "Choose one approximation, constitutive closure, or modeling assumption only.",
                "correction_mode": "closure-family scan",
                "correction_target": "active closure",
            },
        ],
        "match_terms": ["general", "cross-disciplinary", "interdisciplinary", "unknown domain"],
    },
    "theoretical-mechanics": {
        "name": "theoretical-mechanics",
        "label": "Analytical Mechanics",
        "module": "Theoretical Mechanics",
        "summary": "Mechanics bundle focused on force, energy, momentum, kinematics, and constraints.",
        "knowledge_scope": [
            "force balance and free-body structure",
            "energy conservation and dissipation",
            "momentum transfer and impulses",
            "kinematics, coordinates, and constraints",
        ],
        "representative_formulas": [
            {
                "latex": r"\sum F = m a",
                "meaning": "Force-balance route for translational dynamics.",
                "related_skills": ["lagrangian_equations", "hamiltonian_equations"],
            },
            {
                "latex": r"L = T - V",
                "meaning": "Variational route in generalized coordinates.",
                "related_skills": ["lagrangian_equations", "noether_conservation"],
            },
        ],
        "route_seed_options": [
            {
                "label": "force-balance route",
                "route_family": "force-balance",
                "governing_models": ["Newton's Second Law"],
                "guidance": "Name one dominant force balance or one decisive force component only.",
                "correction_mode": "direct force inventory",
                "correction_target": "active force term",
            },
            {
                "label": "energy route",
                "route_family": "energy",
                "governing_models": ["Work-Energy Theorem"],
                "guidance": "State one governing energy relation or deferred loss term only.",
                "correction_mode": "lossless baseline first",
                "correction_target": "dissipation term",
            },
            {
                "label": "momentum route",
                "route_family": "momentum",
                "governing_models": ["Momentum balance"],
                "guidance": "State one momentum-transfer relation or impulse approximation only.",
                "correction_mode": "impulse-balance scan",
                "correction_target": "momentum exchange",
            },
        ],
        "match_terms": [
            "force",
            "energy",
            "momentum",
            "mass",
            "acceleration",
            "friction",
            "motion",
            "lagrangian",
            "hamiltonian",
            "torque",
            "equilibrium",
        ],
    },
    "electrodynamics": {
        "name": "electrodynamics",
        "label": "Electrodynamics",
        "module": "Electrodynamics",
        "summary": "Field-theory bundle for Maxwell equations, potentials, media, and energy flux.",
        "knowledge_scope": [
            "field equations and source terms",
            "potentials and gauge structure",
            "boundary or interface conditions",
            "energy flux and wave propagation",
        ],
        "representative_formulas": [
            {
                "latex": r"\nabla \cdot \mathbf{E} = \rho / \varepsilon_0",
                "meaning": "Gauss-law route from source to field structure.",
                "related_skills": ["maxwell_equations_check", "fields_from_potentials"],
            },
            {
                "latex": r"\mathbf{S} = \frac{1}{\mu_0} \, \mathbf{E} \times \mathbf{B}",
                "meaning": "Energy-flux route via the Poynting vector.",
                "related_skills": ["poynting_vector"],
            },
        ],
        "route_seed_options": [
            {
                "label": "field-equation route",
                "route_family": "field-equation",
                "governing_models": ["Maxwell equations"],
                "guidance": "Choose one governing field equation or source-field relation only.",
                "correction_mode": "field-equation scan",
                "correction_target": "active field relation",
            },
            {
                "label": "potential route",
                "route_family": "potential",
                "governing_models": ["Scalar/vector potentials"],
                "guidance": "Pick one potential representation or gauge condition only.",
                "correction_mode": "representation-choice scan",
                "correction_target": "potential or gauge",
            },
            {
                "label": "boundary-matching route",
                "route_family": "boundary-matching",
                "governing_models": ["Boundary conditions"],
                "guidance": "Name one interface condition or boundary family only.",
                "correction_mode": "boundary-family scan",
                "correction_target": "active boundary condition",
            },
        ],
        "match_terms": [
            "electric",
            "magnetic",
            "field",
            "charge",
            "current",
            "maxwell",
            "electromagnetic",
            "potential",
            "gauss",
            "faraday",
        ],
    },
    "quantum-mechanics": {
        "name": "quantum-mechanics",
        "label": "Quantum Mechanics",
        "module": "Quantum Mechanics",
        "summary": "Quantum bundle for eigenvalue problems, operators, state structure, and perturbations.",
        "knowledge_scope": [
            "hamiltonian and eigenvalue structure",
            "operator algebra and commutators",
            "boundary conditions and basis choice",
            "perturbative corrections",
        ],
        "representative_formulas": [
            {
                "latex": r"\hat{H} \psi = E \psi",
                "meaning": "Stationary eigenvalue route.",
                "related_skills": ["schrodinger_1d", "angular_momentum_eigenstates"],
            },
            {
                "latex": r"[\hat{A}, \hat{B}] = \hat{A}\hat{B} - \hat{B}\hat{A}",
                "meaning": "Operator-compatibility and symmetry route.",
                "related_skills": ["commutator", "pauli_algebra"],
            },
        ],
        "route_seed_options": [
            {
                "label": "eigenvalue route",
                "route_family": "eigenvalue",
                "governing_models": ["Stationary Schroedinger equation"],
                "guidance": "Name one eigenvalue relation, basis, or separable structure only.",
                "correction_mode": "eigenbasis scan",
                "correction_target": "state representation",
            },
            {
                "label": "operator route",
                "route_family": "operator",
                "governing_models": ["Operator algebra"],
                "guidance": "Choose one operator identity, commutator, or symmetry relation only.",
                "correction_mode": "operator-identity scan",
                "correction_target": "active operator relation",
            },
            {
                "label": "perturbation route",
                "route_family": "perturbation",
                "governing_models": ["Perturbation theory"],
                "guidance": "State one perturbative split or correction Hamiltonian only.",
                "correction_mode": "perturbation-order scan",
                "correction_target": "active correction term",
            },
        ],
        "match_terms": [
            "quantum",
            "wavefunction",
            "schrodinger",
            "hamiltonian",
            "eigenvalue",
            "spin",
            "operator",
            "commutator",
            "perturbation",
        ],
    },
    "thermodynamics": {
        "name": "thermodynamics",
        "label": "Thermodynamics and Statistical Physics",
        "module": "Thermodynamics and Statistical Physics",
        "summary": "Thermo/stat-mech bundle for state functions, ensembles, equations of state, and responses.",
        "knowledge_scope": [
            "state variables and equations of state",
            "thermodynamic potentials and Maxwell relations",
            "partition functions and ensembles",
            "response coefficients and constrained derivatives",
        ],
        "representative_formulas": [
            {
                "latex": r"dU = T \, dS - P \, dV + \mu \, dN",
                "meaning": "Fundamental differential route for state relations.",
                "related_skills": ["thermodynamic_potentials", "thermodynamic_partial"],
            },
            {
                "latex": r"Z = \sum_i e^{-\beta E_i}",
                "meaning": "Ensemble route from the canonical partition function.",
                "related_skills": ["partition_function", "statistical_distributions"],
            },
        ],
        "route_seed_options": [
            {
                "label": "state-function route",
                "route_family": "state-function",
                "governing_models": ["Thermodynamic differential"],
                "guidance": "Name one state differential or balance relation only.",
                "correction_mode": "state-differential scan",
                "correction_target": "active state relation",
            },
            {
                "label": "equation-of-state route",
                "route_family": "equation-of-state",
                "governing_models": ["Equation of state"],
                "guidance": "Choose one constitutive state relation only.",
                "correction_mode": "constitutive-state scan",
                "correction_target": "equation of state",
            },
            {
                "label": "ensemble route",
                "route_family": "ensemble",
                "governing_models": ["Partition function"],
                "guidance": "Pick one ensemble or partition-function representation only.",
                "correction_mode": "ensemble-choice scan",
                "correction_target": "ensemble choice",
            },
        ],
        "match_terms": [
            "thermo",
            "temperature",
            "entropy",
            "enthalpy",
            "free energy",
            "partition",
            "ensemble",
            "equation of state",
            "heat",
            "pressure",
        ],
    },
    "special-relativity": {
        "name": "special-relativity",
        "label": "Special Relativity",
        "module": "Special Relativity",
        "summary": "Relativity bundle for invariants, boosts, frame transforms, and relativistic conservation.",
        "knowledge_scope": [
            "Lorentz transformations and frame changes",
            "spacetime intervals and invariants",
            "energy-momentum structure",
            "relativistic limits and composition laws",
        ],
        "representative_formulas": [
            {
                "latex": r"s^2 = c^2 t^2 - x^2 - y^2 - z^2",
                "meaning": "Invariant-interval route.",
                "related_skills": ["lorentz_transform_event", "four_vector_inner_product"],
            },
            {
                "latex": r"E^2 = p^2 c^2 + m^2 c^4",
                "meaning": "Relativistic energy-momentum route.",
                "related_skills": ["relativistic_energy_momentum"],
            },
        ],
        "route_seed_options": [
            {
                "label": "invariant route",
                "route_family": "invariant",
                "governing_models": ["Minkowski invariant"],
                "guidance": "State one frame-invariant interval or four-vector relation only.",
                "correction_mode": "invariant-first scan",
                "correction_target": "active invariant",
            },
            {
                "label": "frame-transform route",
                "route_family": "frame-transform",
                "governing_models": ["Lorentz transformation"],
                "guidance": "Choose one frame transform or boost direction only.",
                "correction_mode": "frame-choice scan",
                "correction_target": "active frame transform",
            },
            {
                "label": "energy-momentum route",
                "route_family": "energy-momentum",
                "governing_models": ["Relativistic energy-momentum relation"],
                "guidance": "State one conserved four-momentum relation only.",
                "correction_mode": "four-momentum scan",
                "correction_target": "active conservation law",
            },
        ],
        "match_terms": [
            "relativity",
            "lorentz",
            "boost",
            "spacetime",
            "four-vector",
            "proper time",
            "relativistic",
            "gamma",
        ],
    },
    "optics-and-waves": {
        "name": "optics-and-waves",
        "label": "Optics and Waves",
        "module": "Optics and Waves",
        "summary": "Optics bundle for interference, diffraction, ray-transfer systems, and wave matching.",
        "knowledge_scope": [
            "phase accumulation and interference",
            "diffraction and aperture structure",
            "ray-transfer and imaging systems",
            "wave boundary matching and polarization",
        ],
        "representative_formulas": [
            {
                "latex": r"I(\theta) \propto \mathrm{sinc}^2(\beta)",
                "meaning": "Aperture-diffraction route.",
                "related_skills": ["single_slit_diffraction", "multi_slit_intensity"],
            },
            {
                "latex": r"m \lambda = d \sin \theta",
                "meaning": "Grating or phase-matching route.",
                "related_skills": ["grating_equation"],
            },
        ],
        "route_seed_options": [
            {
                "label": "phase route",
                "route_family": "phase",
                "governing_models": ["Phase accumulation"],
                "guidance": "Name one phase relation or path-difference condition only.",
                "correction_mode": "phase-difference scan",
                "correction_target": "active phase term",
            },
            {
                "label": "diffraction route",
                "route_family": "diffraction",
                "governing_models": ["Diffraction envelope"],
                "guidance": "Choose one aperture or diffraction relation only.",
                "correction_mode": "aperture-choice scan",
                "correction_target": "active aperture effect",
            },
            {
                "label": "boundary-matching route",
                "route_family": "boundary-matching",
                "governing_models": ["Wave boundary conditions"],
                "guidance": "Fix one interface, polarization, or boundary condition only.",
                "correction_mode": "interface-choice scan",
                "correction_target": "active boundary condition",
            },
        ],
        "match_terms": [
            "optics",
            "wave",
            "diffraction",
            "interference",
            "lens",
            "mirror",
            "polarization",
            "grating",
        ],
    },
    "fluid-mechanics": {
        "name": "fluid-mechanics",
        "label": "Fluid Mechanics",
        "module": "Fluid Mechanics",
        "summary": "Continuum-flow bundle for conservation laws, constitutive closures, transport, and regime maps.",
        "knowledge_scope": [
            "mass and momentum conservation",
            "pressure, viscosity, and constitutive closures",
            "regime maps and dimensionless groups",
            "transport, drag, and surface effects",
        ],
        "representative_formulas": [
            {
                "latex": r"\partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0",
                "meaning": "Continuity route from mass conservation.",
                "related_skills": ["continuity_equation"],
            },
            {
                "latex": r"\mathrm{Re} = \frac{\rho U L}{\mu}",
                "meaning": "Regime-selection route via Reynolds number.",
                "related_skills": ["reynolds_number"],
            },
        ],
        "route_seed_options": [
            {
                "label": "continuity route",
                "route_family": "continuity",
                "governing_models": ["Continuity equation"],
                "guidance": "State one mass-conservation or incompressibility relation only.",
                "correction_mode": "conservation-first scan",
                "correction_target": "mass balance",
            },
            {
                "label": "momentum-balance route",
                "route_family": "momentum-balance",
                "governing_models": ["Momentum equation"],
                "guidance": "Name one dominant momentum balance or pressure-viscous competition only.",
                "correction_mode": "dominant-term scan",
                "correction_target": "active momentum balance",
            },
            {
                "label": "regime-map route",
                "route_family": "regime-map",
                "governing_models": ["Dimensionless regime comparison"],
                "guidance": "Identify one regime split or applicability test before deriving later details.",
                "correction_mode": "regime-selection scan",
                "correction_target": "applicable flow regime",
            },
        ],
        "match_terms": [
            "fluid",
            "flow",
            "viscosity",
            "drag",
            "terminal velocity",
            "reynolds",
            "navier",
            "bernoulli",
            "pressure",
        ],
    },
}


TOT_SKILL_TEMPLATE_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "lagrangian_equations": {
        "knowledge_scope": [
            "generalized coordinates and kinetic/potential energy",
            "Euler-Lagrange structure and coordinate constraints",
            "local dynamical updates in variational form",
        ],
        "representative_formulas": [
            {
                "latex": r"\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = 0",
                "meaning": "Euler-Lagrange route for equations of motion.",
                "related_skills": ["lagrangian_equations"],
            }
        ],
        "route_seed_options": [
            {
                "label": "Euler-Lagrange route",
                "route_family": "euler-lagrange",
                "governing_models": ["Lagrangian mechanics"],
                "guidance": "Use the lagrangian_equations skill only: state one generalized-coordinate relation or one energy term only.",
                "correction_mode": "variational-structure scan",
                "correction_target": "active generalized coordinate",
            }
        ],
    },
    "maxwell_equations_check": {
        "knowledge_scope": [
            "field-source consistency",
            "divergence and curl constraints",
            "electromagnetic boundary and propagation structure",
        ],
        "representative_formulas": [
            {
                "latex": r"\nabla \cdot \mathbf{E} = \rho / \varepsilon_0",
                "meaning": "Gauss-law consistency route.",
                "related_skills": ["maxwell_equations_check"],
            },
            {
                "latex": r"\nabla \times \mathbf{B} = \mu_0 \mathbf{J} + \mu_0 \varepsilon_0 \partial_t \mathbf{E}",
                "meaning": "Ampere-Maxwell consistency route.",
                "related_skills": ["maxwell_equations_check"],
            },
        ],
        "route_seed_options": [
            {
                "label": "Maxwell-consistency route",
                "route_family": "maxwell-consistency",
                "governing_models": ["Maxwell equations"],
                "guidance": "Use the maxwell_equations_check skill only: verify one divergence/curl relation or one source-field consistency condition.",
                "correction_mode": "field-consistency scan",
                "correction_target": "active Maxwell relation",
            }
        ],
    },
    "schrodinger_1d": {
        "knowledge_scope": [
            "stationary-state eigenvalue structure",
            "potential-well or barrier modeling",
            "boundary-condition-driven quantization",
        ],
        "representative_formulas": [
            {
                "latex": r"-\frac{\hbar^2}{2m}\frac{d^2\psi}{dx^2} + V(x)\psi = E\psi",
                "meaning": "One-dimensional stationary Schroedinger route.",
                "related_skills": ["schrodinger_1d"],
            }
        ],
        "route_seed_options": [
            {
                "label": "Schroedinger-eigenvalue route",
                "route_family": "schroedinger-eigenvalue",
                "governing_models": ["Stationary Schroedinger equation"],
                "guidance": "Use the schrodinger_1d skill only: state one eigenvalue relation, potential regime, or boundary condition only.",
                "correction_mode": "eigenproblem scan",
                "correction_target": "active potential or boundary regime",
            }
        ],
    },
    "partition_function": {
        "knowledge_scope": [
            "canonical ensemble state sums",
            "thermodynamic observables derived from Z",
            "temperature dependence and energy-level structure",
        ],
        "representative_formulas": [
            {
                "latex": r"Z = \sum_i e^{-\beta E_i}",
                "meaning": "Canonical partition-function route.",
                "related_skills": ["partition_function"],
            },
            {
                "latex": r"U = -\frac{\partial}{\partial \beta} \ln Z",
                "meaning": "Observable extraction route from the partition function.",
                "related_skills": ["partition_function"],
            },
        ],
        "route_seed_options": [
            {
                "label": "partition-function route",
                "route_family": "partition-function",
                "governing_models": ["Canonical ensemble"],
                "guidance": "Use the partition_function skill only: choose one state-sum representation or one derived observable from Z.",
                "correction_mode": "ensemble-state-sum scan",
                "correction_target": "active observable from Z",
            }
        ],
    },
    "continuity_equation": {
        "knowledge_scope": [
            "mass conservation and transport balance",
            "compressibility or incompressibility assumptions",
            "flux structure across control surfaces",
        ],
        "representative_formulas": [
            {
                "latex": r"\partial_t \rho + \nabla \cdot (\rho \mathbf{u}) = 0",
                "meaning": "Continuity route from mass conservation.",
                "related_skills": ["continuity_equation"],
            }
        ],
        "route_seed_options": [
            {
                "label": "continuity route",
                "route_family": "continuity",
                "governing_models": ["Mass conservation"],
                "guidance": "Use the continuity_equation skill only: state one conservation law or one flux assumption only.",
                "correction_mode": "transport-balance scan",
                "correction_target": "active flux term",
            }
        ],
    },
    "lorentz_transform_event": {
        "knowledge_scope": [
            "frame changes and spacetime coordinates",
            "Lorentz invariance and event mapping",
            "kinematic interpretation across inertial frames",
        ],
        "representative_formulas": [
            {
                "latex": r"x'^\mu = \Lambda^\mu_{\ \nu} x^\nu",
                "meaning": "Lorentz-transform route between inertial frames.",
                "related_skills": ["lorentz_transform_event"],
            }
        ],
        "route_seed_options": [
            {
                "label": "Lorentz-transform route",
                "route_family": "lorentz-transform",
                "governing_models": ["Lorentz transformation"],
                "guidance": "Use the lorentz_transform_event skill only: choose one frame mapping or invariant-event interpretation.",
                "correction_mode": "frame-mapping scan",
                "correction_target": "active frame transform",
            }
        ],
    },
    "thin_lens_matrix": {
        "knowledge_scope": [
            "paraxial ray-transfer structure",
            "optical element composition",
            "focal constraints and imaging conditions",
        ],
        "representative_formulas": [
            {
                "latex": r"M_{\mathrm{lens}} = \begin{pmatrix} 1 & 0 \\ -1/f & 1 \end{pmatrix}",
                "meaning": "Thin-lens ABCD-matrix route.",
                "related_skills": ["thin_lens_matrix"],
            }
        ],
        "route_seed_options": [
            {
                "label": "thin-lens matrix route",
                "route_family": "thin-lens-matrix",
                "governing_models": ["Paraxial optics"],
                "guidance": "Use the thin_lens_matrix skill only: state one lens-transfer relation or one focal condition only.",
                "correction_mode": "ray-transfer scan",
                "correction_target": "active optical element",
            }
        ],
    },
}


TOT_SKILL_VALIDATION_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "lagrangian_equations": {
        "summary": "Validate variational mechanics steps by checking for Lagrangian or Euler-Lagrange structure.",
        "hard_rule_params": {
            "required_any_context_patterns": [
                "lagrangian",
                "euler-lagrange",
                "generalized coordinate",
                "\\dot{q}",
                "\\partial l",
            ],
        },
    },
    "maxwell_equations_check": {
        "summary": "Validate electromagnetic field steps by checking for Maxwell-style field/source structure.",
        "hard_rule_params": {
            "required_any_context_patterns": [
                "maxwell",
                "\\nabla",
                "\\mathbf{e}",
                "\\mathbf{b}",
                "electric",
                "magnetic",
            ],
        },
    },
    "schrodinger_1d": {
        "summary": "Validate one-dimensional quantum steps by checking for wavefunction, Hamiltonian, or potential structure.",
        "hard_rule_params": {
            "required_any_context_patterns": [
                "schrodinger",
                "\\psi",
                "\\hbar",
                "potential",
                "eigenvalue",
            ],
        },
    },
    "partition_function": {
        "summary": "Validate canonical-ensemble steps by checking for partition-function or Boltzmann-weight structure.",
        "hard_rule_params": {
            "required_any_context_patterns": [
                "partition",
                "beta",
                "ln z",
                "z =",
                "boltzmann",
            ],
            "positive_var_names": ["Z"],
        },
    },
    "continuity_equation": {
        "summary": "Validate transport steps by checking for continuity or mass-conservation structure.",
        "hard_rule_params": {
            "required_any_context_patterns": [
                "continuity",
                "mass conservation",
                "\\nabla \\cdot",
                "\\rho",
                "\\mathbf{u}",
            ],
        },
    },
    "lorentz_transform_event": {
        "summary": "Validate relativistic frame-change steps by checking for Lorentz-transform or invariant structure.",
        "hard_rule_params": {
            "required_any_context_patterns": [
                "lorentz",
                "boost",
                "gamma",
                "interval",
                "four-vector",
            ],
        },
    },
    "thin_lens_matrix": {
        "summary": "Validate paraxial-optics steps by checking for thin-lens or ABCD-matrix structure.",
        "hard_rule_params": {
            "required_any_context_patterns": [
                "lens",
                "abcd",
                "1/f",
                "focal",
                "matrix",
            ],
            "nonzero_var_names": ["f"],
        },
    },
}


def _ordered_unique_strings(items: Sequence[Any]) -> List[str]:
    values: List[str] = []
    for item in items:
        text = str(item).strip()
        if text and text not in values:
            values.append(text)
    return values


def _default_skill_template(skill_name: str) -> Dict[str, Any]:
    entry = get_skill_entry(skill_name)
    module = str(entry.get("module", "")).strip()
    summary = str(entry.get("summary", "")).strip()
    keywords = _ordered_unique_strings(entry.get("keywords", []))
    primary_focus = keywords[0] if keywords else skill_name.replace("_", "-")
    return {
        "name": skill_name,
        "label": skill_name,
        "skill_name": skill_name,
        "module": module,
        "summary": summary,
        "knowledge_scope": _ordered_unique_strings([summary, *keywords[:3]]),
        "representative_formulas": [
            {
                "latex": rf"\mathrm{{{skill_name}}}(\cdots)",
                "meaning": summary or f"Use the {skill_name} skill as the active modeling operator.",
                "related_skills": [skill_name],
            }
        ],
        "route_seed_options": [
            {
                "label": f"{skill_name} route",
                "route_family": skill_name.replace("_", "-"),
                "governing_models": [module or skill_name],
                "guidance": f"Use the {skill_name} skill only: state one {primary_focus} relation, assumption, or structure that matches this skill.",
                "correction_mode": "skill-specific scan",
                "correction_target": primary_focus,
            }
        ],
        "skill_names": [skill_name],
        "keywords": keywords,
    }


def _build_skill_template(skill_name: str) -> Dict[str, Any]:
    template = _default_skill_template(skill_name)
    override = TOT_SKILL_TEMPLATE_OVERRIDES.get(skill_name, {})
    if override:
        if "label" in override:
            template["label"] = str(override["label"]).strip() or template["label"]
        if "module" in override:
            template["module"] = str(override["module"]).strip() or template["module"]
        if "summary" in override:
            template["summary"] = str(override["summary"]).strip() or template["summary"]
        if "knowledge_scope" in override:
            template["knowledge_scope"] = _ordered_unique_strings(override.get("knowledge_scope", []))
        if "representative_formulas" in override:
            template["representative_formulas"] = _normalize_domain_plugin(
                {
                    "name": skill_name,
                    "representative_formulas": override.get("representative_formulas", []),
                },
                fallback_name=skill_name,
            )["representative_formulas"]
        if "route_seed_options" in override:
            template["route_seed_options"] = _normalize_domain_plugin(
                {
                    "name": skill_name,
                    "route_seed_options": override.get("route_seed_options", []),
                },
                fallback_name=skill_name,
            )["route_seed_options"]
    return template


def _skill_names_for_module(module: str) -> List[str]:
    return sorted(
        skill_name
        for skill_name, entry in SKILL_REGISTRY.items()
        if entry.get("module") == module and not skill_name.startswith("tot_")
    )


_VALIDATION_RULE_STRING_LIST_KEYS: Tuple[str, ...] = (
    "required_known_vars",
    "required_equation_patterns",
    "required_any_equation_patterns",
    "forbidden_equation_patterns",
    "required_models",
    "forbidden_models",
    "required_model_patterns",
    "required_any_model_patterns",
    "forbidden_model_patterns",
    "required_boundary_condition_keys",
    "forbidden_boundary_condition_keys",
    "required_boundary_condition_patterns",
    "required_any_boundary_condition_patterns",
    "forbidden_boundary_condition_patterns",
    "required_all_context_patterns",
    "required_any_context_patterns",
    "forbidden_any_context_patterns",
    "positive_var_names",
    "nonzero_var_names",
    "finite_var_names",
    "custom_violations",
)
_VALIDATION_RULE_DICT_KEYS: Tuple[str, ...] = (
    "required_boundary_conditions",
    "forbidden_boundary_conditions",
)
_VALIDATION_RULE_STRUCTURED_LIST_KEYS: Tuple[str, ...] = ("dimension_equalities",)
_VALIDATION_RULE_BOOL_KEYS: Tuple[str, ...] = (
    "require_equations",
    "semantic_boundary_checks",
)


def _normalize_validation_rule_params(params: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in _VALIDATION_RULE_STRING_LIST_KEYS:
        if key in params:
            normalized[key] = _ordered_unique_strings(params.get(key, []))
    for key in _VALIDATION_RULE_DICT_KEYS:
        if key in params:
            raw_value = params.get(key, {})
            normalized[key] = (
                {str(item_key): item_value for item_key, item_value in dict(raw_value).items()}
                if isinstance(raw_value, dict)
                else {}
            )
    for key in _VALIDATION_RULE_STRUCTURED_LIST_KEYS:
        if key in params:
            raw_items = params.get(key, [])
            normalized[key] = [item for item in raw_items if isinstance(item, dict)]
    for key in _VALIDATION_RULE_BOOL_KEYS:
        if key in params:
            normalized[key] = bool(params.get(key))

    section_aliases: Dict[str, Any] = {}

    equations_rules = params.get("equations")
    if isinstance(equations_rules, dict):
        section_aliases["required_equation_patterns"] = _ordered_unique_strings(
            [
                *equations_rules.get("require_all_patterns", []),
                *equations_rules.get("require_patterns", []),
            ]
        )
        section_aliases["required_any_equation_patterns"] = _ordered_unique_strings(
            equations_rules.get("require_any_patterns", [])
        )
        section_aliases["forbidden_equation_patterns"] = _ordered_unique_strings(
            equations_rules.get("forbid_patterns", [])
        )

    models_rules = params.get("models")
    if isinstance(models_rules, dict):
        section_aliases["required_models"] = _ordered_unique_strings(models_rules.get("require_exact", []))
        section_aliases["forbidden_models"] = _ordered_unique_strings(models_rules.get("forbid_exact", []))
        section_aliases["required_model_patterns"] = _ordered_unique_strings(
            [
                *models_rules.get("require_all_patterns", []),
                *models_rules.get("require_patterns", []),
            ]
        )
        section_aliases["required_any_model_patterns"] = _ordered_unique_strings(
            models_rules.get("require_any_patterns", [])
        )
        section_aliases["forbidden_model_patterns"] = _ordered_unique_strings(
            models_rules.get("forbid_patterns", [])
        )

    boundary_rules = params.get("boundary_conditions")
    if isinstance(boundary_rules, dict):
        section_aliases["required_boundary_condition_keys"] = _ordered_unique_strings(
            boundary_rules.get("require_keys", [])
        )
        section_aliases["forbidden_boundary_condition_keys"] = _ordered_unique_strings(
            boundary_rules.get("forbid_keys", [])
        )
        section_aliases["required_boundary_condition_patterns"] = _ordered_unique_strings(
            [
                *boundary_rules.get("require_all_patterns", []),
                *boundary_rules.get("require_patterns", []),
            ]
        )
        section_aliases["required_any_boundary_condition_patterns"] = _ordered_unique_strings(
            boundary_rules.get("require_any_patterns", [])
        )
        section_aliases["forbidden_boundary_condition_patterns"] = _ordered_unique_strings(
            boundary_rules.get("forbid_patterns", [])
        )
        require_matches = boundary_rules.get("require_matches", {})
        if isinstance(require_matches, dict):
            section_aliases["required_boundary_conditions"] = require_matches
        forbid_matches = boundary_rules.get("forbid_matches", {})
        if isinstance(forbid_matches, dict):
            section_aliases["forbidden_boundary_conditions"] = forbid_matches

    context_rules = params.get("context")
    if isinstance(context_rules, dict):
        section_aliases["required_all_context_patterns"] = _ordered_unique_strings(
            [
                *context_rules.get("require_all_patterns", []),
                *context_rules.get("require_patterns", []),
            ]
        )
        section_aliases["required_any_context_patterns"] = _ordered_unique_strings(
            context_rules.get("require_any_patterns", [])
        )
        section_aliases["forbidden_any_context_patterns"] = _ordered_unique_strings(
            context_rules.get("forbid_patterns", [])
        )

    variables_rules = params.get("variables")
    if isinstance(variables_rules, dict):
        section_aliases["required_known_vars"] = _ordered_unique_strings(
            variables_rules.get("require_known", [])
        )
        section_aliases["positive_var_names"] = _ordered_unique_strings(variables_rules.get("positive", []))
        section_aliases["nonzero_var_names"] = _ordered_unique_strings(variables_rules.get("nonzero", []))
        section_aliases["finite_var_names"] = _ordered_unique_strings(variables_rules.get("finite", []))

    dimensions_rules = params.get("dimensions")
    if isinstance(dimensions_rules, dict):
        equalities = dimensions_rules.get("equalities", [])
        if isinstance(equalities, list):
            section_aliases["dimension_equalities"] = [item for item in equalities if isinstance(item, dict)]

    flags_rules = params.get("flags")
    if isinstance(flags_rules, dict):
        if "require_equations" in flags_rules:
            section_aliases["require_equations"] = bool(flags_rules.get("require_equations"))
        if "semantic_boundary_checks" in flags_rules:
            section_aliases["semantic_boundary_checks"] = bool(flags_rules.get("semantic_boundary_checks"))

    violations_rules = params.get("violations")
    if isinstance(violations_rules, dict):
        section_aliases["custom_violations"] = _ordered_unique_strings(violations_rules.get("append", []))

    return _merge_validation_rule_params(normalized, section_aliases)


def _merge_validation_rule_params(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for key in _VALIDATION_RULE_BOOL_KEYS:
        if key in overrides:
            merged[key] = bool(overrides[key])
        elif key in defaults:
            merged[key] = bool(defaults[key])
    for key in _VALIDATION_RULE_STRING_LIST_KEYS:
        values = _ordered_unique_strings([*defaults.get(key, []), *overrides.get(key, [])])
        if values:
            merged[key] = values
    for key in _VALIDATION_RULE_DICT_KEYS:
        values = {
            **dict(defaults.get(key, {})),
            **dict(overrides.get(key, {})),
        }
        if values:
            merged[key] = values
    for key in _VALIDATION_RULE_STRUCTURED_LIST_KEYS:
        values = list(defaults.get(key, []))
        for item in overrides.get(key, []):
            if item not in values:
                values.append(item)
        if values:
            merged[key] = values
    return merged


def _combine_validation_rule_sets(rule_sets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for rule_set in rule_sets:
        merged = _merge_validation_rule_params(merged, _normalize_validation_rule_params(rule_set))
    return merged


def _default_skill_validation_template(skill_name: str) -> Dict[str, Any]:
    entry = get_skill_entry(skill_name)
    module = str(entry.get("module", "")).strip()
    summary = str(entry.get("summary", "")).strip()
    return {
        "name": skill_name,
        "label": skill_name,
        "skill_name": skill_name,
        "module": module,
        "summary": summary,
        "skill_names": [skill_name],
        "hard_rule_params": {"require_equations": True},
    }


def _build_skill_validation_template(skill_name: str) -> Dict[str, Any]:
    template = _default_skill_validation_template(skill_name)
    override = TOT_SKILL_VALIDATION_OVERRIDES.get(skill_name, {})
    if override:
        if "label" in override:
            template["label"] = str(override["label"]).strip() or template["label"]
        if "module" in override:
            template["module"] = str(override["module"]).strip() or template["module"]
        if "summary" in override:
            template["summary"] = str(override["summary"]).strip() or template["summary"]
        template["hard_rule_params"] = _merge_validation_rule_params(
            template["hard_rule_params"],
            _normalize_validation_rule_params(dict(override.get("hard_rule_params", {}))),
        )
    return template


def _normalize_validation_plugin(plugin: Dict[str, Any], *, fallback_name: str) -> Dict[str, Any]:
    name = str(plugin.get("name", fallback_name)).strip() or fallback_name
    label = str(plugin.get("label", name)).strip() or name
    module = str(plugin.get("module", "")).strip()
    summary = str(plugin.get("summary", "")).strip()
    skill_names = _ordered_unique_strings(plugin.get("skill_names", []))
    if not skill_names and module:
        skill_names = _skill_names_for_module(module)
    hard_rule_params = _normalize_validation_rule_params(dict(plugin.get("validation_rules", {})))
    return {
        "name": name,
        "label": label,
        "module": module,
        "summary": summary,
        "skill_names": skill_names,
        "hard_rule_params": hard_rule_params,
    }


def tot_validation_plugin_bundle(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""Build a pluggable validation bundle that injects skill-specific hard-rule parameters.

    ``validation_rules`` may be supplied either as flat ``tot_hard_rule_check`` keys
    or as nested sections:

    - ``equations``: ``require_patterns`` / ``require_all_patterns``, ``require_any_patterns``, ``forbid_patterns``
    - ``models``: ``require_exact``, ``require_patterns`` / ``require_all_patterns``, ``require_any_patterns``, ``forbid_exact``, ``forbid_patterns``
    - ``boundary_conditions``: ``require_keys``, ``forbid_keys``, ``require_patterns`` / ``require_all_patterns``, ``require_any_patterns``, ``forbid_patterns``, ``require_matches``, ``forbid_matches``
    - ``context``: ``require_patterns`` / ``require_all_patterns``, ``require_any_patterns``, ``forbid_patterns``
    - ``variables``: ``require_known``, ``positive``, ``nonzero``, ``finite``
    - ``dimensions``: ``equalities``
    - ``flags``: ``require_equations``, ``semantic_boundary_checks``
    - ``violations``: ``append``

    The bundle normalizes these forms into the flat hard-rule parameter surface
    consumed by ``tot_hard_rule_check``.
    """

    problem_context = dict(params.get("problem_context", {})) if isinstance(params.get("problem_context"), dict) else {}
    explicit_skill_names = params.get("skill_names", problem_context.get("skill_names", []))
    if isinstance(explicit_skill_names, (str, bytes)):
        explicit_skill_names = [str(explicit_skill_names)]
    elif explicit_skill_names in (None, ""):
        explicit_skill_names = []
    else:
        explicit_skill_names = [str(item) for item in explicit_skill_names]
    explicit_skill_names = _ordered_unique_strings(explicit_skill_names)

    selected_validators: List[Dict[str, Any]] = []
    selection_mode = "fallback"
    if explicit_skill_names:
        selected_validators = [
            _build_skill_validation_template(skill_name)
            for skill_name in explicit_skill_names
            if skill_name in SKILL_REGISTRY
        ]
        if selected_validators:
            selection_mode = "explicit"
    else:
        custom_plugins = params.get("domain_plugins", problem_context.get("domain_plugins", []))
        if custom_plugins in (None, ""):
            custom_plugins = []
        if custom_plugins and not isinstance(custom_plugins, list):
            raise TypeError("domain_plugins must be a list of plugin dictionaries.")
        selected_validators = [
            _normalize_validation_plugin(item, fallback_name=f"custom-validator-{index + 1}")
            for index, item in enumerate(custom_plugins)
            if isinstance(item, dict)
            and isinstance(item.get("validation_rules"), dict)
            and item.get("validation_rules")
        ]
        if selected_validators:
            selection_mode = "custom"

    hard_rule_params = _combine_validation_rule_sets(
        [validator.get("hard_rule_params", {}) for validator in selected_validators]
    )
    return {
        "selection_mode": selection_mode,
        "selected_validators": selected_validators,
        "recommended_skills": _ordered_unique_strings(
            skill_name
            for validator in selected_validators
            for skill_name in validator.get("skill_names", [])
        ),
        "hard_rule_params": hard_rule_params,
    }


def _normalize_domain_plugin(plugin: Dict[str, Any], *, fallback_name: str) -> Dict[str, Any]:
    name = str(plugin.get("name", fallback_name)).strip() or fallback_name
    label = str(plugin.get("label", name)).strip() or name
    module = str(plugin.get("module", "")).strip()
    skill_names = _ordered_unique_strings(plugin.get("skill_names", []))
    if not skill_names and module:
        skill_names = _skill_names_for_module(module)

    representative_formulas = []
    for item in plugin.get("representative_formulas", []):
        if not isinstance(item, dict):
            continue
        latex = str(item.get("latex", "")).strip()
        if not latex:
            continue
        representative_formulas.append(
            {
                "latex": latex,
                "meaning": str(item.get("meaning", "")).strip(),
                "related_skills": _ordered_unique_strings(item.get("related_skills", [])),
            }
        )

    route_seed_options = []
    for item in plugin.get("route_seed_options", []):
        if not isinstance(item, dict):
            continue
        route_family = str(item.get("route_family", "")).strip()
        if not route_family:
            continue
        route_seed_options.append(
            {
                "label": str(item.get("label", route_family)).strip() or route_family,
                "route_family": route_family,
                "governing_models": _ordered_unique_strings(item.get("governing_models", [])),
                "guidance": str(item.get("guidance", "")).strip(),
                "correction_mode": str(item.get("correction_mode", "")).strip(),
                "correction_target": str(item.get("correction_target", "")).strip(),
            }
        )

    return {
        "name": name,
        "label": label,
        "module": module,
        "summary": str(plugin.get("summary", "")).strip(),
        "knowledge_scope": _ordered_unique_strings(plugin.get("knowledge_scope", [])),
        "representative_formulas": representative_formulas,
        "route_seed_options": route_seed_options,
        "skill_names": skill_names,
        "match_terms": _ordered_unique_strings(plugin.get("match_terms", [])),
    }


def _domain_plugin_aliases(plugin: Dict[str, Any]) -> List[str]:
    aliases = [plugin.get("name", ""), plugin.get("label", ""), plugin.get("module", "")]
    aliases.extend(plugin.get("match_terms", []))
    normalized: List[str] = []
    for alias in aliases:
        text = str(alias).strip().lower()
        if not text:
            continue
        if text not in normalized:
            normalized.append(text)
        spaced = text.replace("-", " ")
        if spaced not in normalized:
            normalized.append(spaced)
    return normalized


def _render_domain_plugin_prompt_fragment(plugins: Sequence[Dict[str, Any]]) -> str:
    if not plugins:
        return (
            "If no domain plugin is selected, use a generic route scan over governing relations, constraints, invariants, "
            "boundary conditions, limiting cases, approximations, and closures."
        )

    sentences: List[str] = []
    for plugin in plugins[:3]:
        parts = [f"Domain plugin {plugin['label']}." ]
        scope_text = ", ".join(plugin.get("knowledge_scope", [])[:4])
        if scope_text:
            parts.append(f"Knowledge scope: {scope_text}.")
        formulas = plugin.get("representative_formulas", [])[:2]
        if formulas:
            formula_text = "; ".join(
                f"{item['latex']} ({item['meaning']})" if item.get("meaning") else item["latex"]
                for item in formulas
            )
            parts.append(f"Representative LaTeX formulas: {formula_text}.")
        skills_text = ", ".join(plugin.get("skill_names", [])[:5])
        if skills_text:
            parts.append(f"Relevant skills: {skills_text}.")
        sentences.append(" ".join(parts))
    sentences.append(
        "Use the selected plugin scopes, formulas, and skills to seed route_options and step_blueprints; each route should isolate exactly one governing law/model, structural constraint, boundary-condition family, approximation, or closure choice."
    )
    return " ".join(sentences)


def _render_skill_template_prompt_fragment(skill_templates: Sequence[Dict[str, Any]]) -> str:
    if not skill_templates:
        return ""

    sentences: List[str] = []
    for template in skill_templates[:4]:
        parts = [f"Selected skill {template['skill_name']} ({template.get('module', '') or 'unknown module'})."]
        if template.get("summary"):
            parts.append(f"Role: {template['summary']}")
        scope_text = ", ".join(template.get("knowledge_scope", [])[:3])
        if scope_text:
            parts.append(f"Knowledge scope: {scope_text}.")
        formulas = template.get("representative_formulas", [])[:2]
        if formulas:
            formula_text = "; ".join(
                f"{item['latex']} ({item['meaning']})" if item.get("meaning") else item["latex"]
                for item in formulas
            )
            parts.append(f"Formula templates: {formula_text}.")
        routes = ", ".join(item.get("route_family", "") for item in template.get("route_seed_options", [])[:2] if item.get("route_family"))
        if routes:
            parts.append(f"Preferred route families: {routes}.")
        sentences.append(" ".join(parts))
    sentences.append(
        "Use the selected skills as the primary theme switch: route_options and step_blueprints should stay centered on those skill-specific formulas, quantities, and assumptions."
    )
    return " ".join(sentences)


def tot_domain_plugin_bundle(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""Build a pluggable domain bundle for planning prompts and route seeding."""

    problem_context = dict(params.get("problem_context", {})) if isinstance(params.get("problem_context"), dict) else {}
    explicit_skill_names = params.get("skill_names", problem_context.get("skill_names", []))
    if isinstance(explicit_skill_names, (str, bytes)):
        explicit_skill_names = [str(explicit_skill_names)]
    elif explicit_skill_names in (None, ""):
        explicit_skill_names = []
    else:
        explicit_skill_names = [str(item) for item in explicit_skill_names]
    explicit_skill_names = _ordered_unique_strings(explicit_skill_names)

    selected_skill_templates: List[Dict[str, Any]] = []
    if explicit_skill_names:
        for skill_name in explicit_skill_names:
            if skill_name not in SKILL_REGISTRY:
                continue
            selected_skill_templates.append(_build_skill_template(skill_name))

    custom_plugins = params.get("domain_plugins", problem_context.get("domain_plugins", []))
    if custom_plugins in (None, ""):
        custom_plugins = []
    if custom_plugins and not isinstance(custom_plugins, list):
        raise TypeError("domain_plugins must be a list of plugin dictionaries.")

    normalized_custom_plugins = [
        _normalize_domain_plugin(item, fallback_name=f"custom-plugin-{index + 1}")
        for index, item in enumerate(custom_plugins)
        if isinstance(item, dict)
    ]
    if selected_skill_templates:
        selected_plugins = [
            {
                "name": template["name"],
                "label": template["label"],
                "module": template["module"],
                "summary": template["summary"],
                "knowledge_scope": list(template.get("knowledge_scope", [])),
                "representative_formulas": [dict(item) for item in template.get("representative_formulas", [])],
                "route_seed_options": [dict(item) for item in template.get("route_seed_options", [])],
                "skill_names": list(template.get("skill_names", [])),
            }
            for template in selected_skill_templates
        ]
        selection_mode = "explicit"
    elif normalized_custom_plugins:
        selected_plugins = normalized_custom_plugins
        selection_mode = "custom"
    else:
        explicit_domains = params.get("domain", problem_context.get("domain", problem_context.get("discipline", [])))
        if isinstance(explicit_domains, (str, bytes)):
            explicit_domain_list = [str(explicit_domains)]
        elif explicit_domains in (None, ""):
            explicit_domain_list = []
        else:
            explicit_domain_list = [str(item) for item in explicit_domains]
        explicit_domain_list = _ordered_unique_strings(explicit_domain_list)

        query_text = " ".join(
            part
            for part in [
                str(params.get("query", "")).strip(),
                str(problem_context.get("skill_query", "")).strip(),
                str(params.get("problem_statement", problem_context.get("problem_statement", ""))).strip(),
                str(problem_context.get("task", "")).strip(),
            ]
            if part
        ).lower()

        selected_plugins = []
        selection_mode = "fallback"
        for key, template in TOT_DOMAIN_PLUGIN_TEMPLATES.items():
            plugin = _normalize_domain_plugin(template, fallback_name=key)
            if explicit_domain_list and any(
                domain.lower() in _domain_plugin_aliases(plugin)
                for domain in explicit_domain_list
            ):
                selected_plugins.append(plugin)
        if selected_plugins:
            selection_mode = "explicit"
        else:
            for key, template in TOT_DOMAIN_PLUGIN_TEMPLATES.items():
                if key == "general-scientific":
                    continue
                plugin = _normalize_domain_plugin(template, fallback_name=key)
                if any(alias in query_text for alias in _domain_plugin_aliases(plugin)):
                    selected_plugins.append(plugin)
            if selected_plugins:
                selection_mode = "inferred"
            else:
                selected_plugins = [
                    _normalize_domain_plugin(
                        TOT_DOMAIN_PLUGIN_TEMPLATES["general-scientific"],
                        fallback_name="general-scientific",
                    )
                ]

    knowledge_scope = _ordered_unique_strings(
        item
        for plugin in selected_plugins
        for item in plugin.get("knowledge_scope", [])
    )
    representative_formulas = [
        {
            "plugin_name": plugin["name"],
            "plugin_label": plugin["label"],
            "latex": item["latex"],
            "meaning": item["meaning"],
            "related_skills": list(item.get("related_skills", [])),
        }
        for plugin in selected_plugins
        for item in plugin.get("representative_formulas", [])
    ]
    route_seed_options = [
        dict(item)
        for plugin in selected_plugins
        for item in plugin.get("route_seed_options", [])
    ]
    recommended_skills = _ordered_unique_strings(
        skill_name
        for plugin in selected_plugins
        for skill_name in plugin.get("skill_names", [])
    )

    if selected_skill_templates:
        prompt_fragment = _render_skill_template_prompt_fragment(selected_skill_templates)
    else:
        prompt_fragment = _render_domain_plugin_prompt_fragment(selected_plugins)

    return {
        "selection_mode": selection_mode,
        "selected_plugins": [
            {
                key: value
                for key, value in plugin.items()
                if key != "match_terms"
            }
            for plugin in selected_plugins
        ],
        "selected_skills": [
            {
                key: value
                for key, value in template.items()
                if key != "keywords"
            }
            for template in selected_skill_templates
        ],
        "knowledge_scope": knowledge_scope,
        "representative_formulas": representative_formulas,
        "recommended_skills": recommended_skills,
        "route_seed_options": route_seed_options,
        "prompt_fragment": prompt_fragment,
    }


def tot_stage_prompt_contract(params: Dict[str, Any]) -> Dict[str, Any]:
    r"""Return stage-specific JSON format contracts and prompt fragments for ToT chat stages."""

    stage = str(params.get("stage", "")).strip().lower()
    if not stage:
        raise ValueError("tot_stage_prompt_contract requires a non-empty 'stage'.")

    domain_bundle = tot_domain_plugin_bundle(params)
    domain_plugin_prompt = str(domain_bundle.get("prompt_fragment", "")).strip()

    contracts: Dict[str, Dict[str, Any]] = {
        "meta-analysis": {
            "required_keys": [
                "objective",
                "givens",
                "unknowns",
                "minimal_subproblems",
                "step_ordering",
                "first_step",
                "completion_signals",
            ],
            "optional_keys": ["route_options", "step_blueprints"],
            "single_step": False,
            "prompt_fragment": (
                "You are the ToT planning model. Analyze the problem once at session creation time and return only a JSON object "
                "with keys objective, givens, unknowns, minimal_subproblems, step_ordering, first_step, completion_signals. "
                "The first item in minimal_subproblems and step_ordering must be a route-splitting checkpoint only: preserve many plausible modeling routes by scanning the active domain plugin guidance rather than relying on one fixed subject-specific route list, while still noting governing laws/models, hidden assumptions, deferred corrections, and alternative correction quantities or closure choices without solving the target. "
                f"{domain_plugin_prompt} "
                "Keep each route option and step blueprint short and atomic: each one should represent the simplest route-local first move, such as naming one governing law/model, one decisive assumption, or one active correction quantity or closure. Do not let a single step both compare many routes and refine them. "
                "Every later item must refine exactly one quantity, relation, assumption, approximation, or correction term. Prefer breadth before commitment: if several routes or correction styles look viable, keep them visible in the plan so later orchestration can choose among them. If useful, also include optional route_options and step_blueprints objects that preserve route_family, governing_models, assumptions, deferred_terms, target quantities, correction_mode, and correction_target for distributed downstream reasoning. For non-trivial problems, prefer roughly five to seven coarse checkpoints instead of collapsing the full plan into only two or three steps. Keep the plan coarse and action-oriented; later per-step orchestration will strictly split each checkpoint into one executable micro task. Do not solve the full problem. Do not use markdown."
            ),
        },
        "orchestrator": {
            "required_keys": [
                "step_focus",
                "current_step_guidance",
                "task_breakdown",
                "selected_task",
                "deferred_tasks",
                "completion_signals",
            ],
            "optional_keys": ["selected_route_family", "candidate_tasks"],
            "single_step": False,
            "prompt_fragment": (
                "You are the ToT orchestrator. Return only a JSON object with keys step_focus, current_step_guidance, task_breakdown, selected_task, deferred_tasks, completion_signals. "
                "You do not receive the full problem statement; operate only on the local checkpoint metadata already provided in the request. "
                "Use the current node state, parent state, meta-task progress, and latest review feedback to strictly decompose the active checkpoint into the smallest executable micro tasks, then choose exactly one selected_task for the modeling model to execute now. "
                "During strategy_scan, selected_task must isolate one route family only and do exactly one thing: name one governing law/model, state one decisive assumption, or choose one active correction quantity or closure. Put all other work into deferred_tasks. "
                "When multiple route families or correction modes remain viable, also include optional selected_route_family and candidate_tasks objects so downstream nodes can preserve distributed reasoning across alternatives without losing structure; candidate_tasks should preserve route_family, correction_mode, and correction_target whenever they matter. Do not derive equations or solve the task yourself. Do not use markdown."
            ),
        },
        "proposal": {
            "required_keys": [
                "thought_step",
                "equations",
                "known_vars",
                "used_models",
                "quantities",
                "boundary_conditions",
            ],
            "single_step": True,
            "prompt_fragment": (
                "You are the ToT modeling model. Return only a JSON object with keys thought_step, equations, known_vars, used_models, quantities, boundary_conditions. "
                "Produce exactly one minimal next-step candidate for the current tree node. Do not solve the whole problem, do not include multiple alternatives, and do not jump ahead to later subproblems. "
                "Use request.problem_context.orchestrator_task.selected_task and request.problem_context.meta_task_progress.current_step_guidance as the only allowed subproblem for this node. "
                "The orchestrator task is authoritative: execute only that selected task and defer every item listed in request.problem_context.orchestrator_task.deferred_tasks. "
                "If request.problem_context.meta_task_progress.phase is strategy_scan, stay at planning level and stay route-local and atomic: do not compare many routes inside one node. State only one short planning claim for the selected route, such as one governing law/model, one decisive assumption, or one active correction quantity or closure. "
                "If request.problem_context.meta_task_progress.phase is incremental_refinement, add or correct exactly one quantity, relation, approximation, or correction term, and keep the step short. If the refinement is non-terminal and request.parent_node is present, the child must add exactly one explicit local delta beyond the parent: one correction, one boundary condition, or one control parameter. The thought_step itself must name that new local delta and must not paraphrase the parent claim. Surface the same delta in equations, quantities, boundary_conditions, or known_vars using a short marker such as active_correction, active_boundary_condition, or active_control_parameter. "
                "Advance only that step and leave later steps untouched. Do not use markdown."
            ),
        },
        "reflection": {
            "required_keys": [
                "thought_step",
                "equations",
                "known_vars",
                "used_models",
                "quantities",
                "boundary_conditions",
            ],
            "single_step": True,
            "prompt_fragment": (
                "You are the ToT modeling model refining an existing branch. Return only a JSON object with keys thought_step, equations, known_vars, used_models, quantities, boundary_conditions. "
                "Make exactly one local revision step for the current branch. Do not restart the full solution, do not emit multiple revisions, and do not skip ahead beyond request.problem_context.orchestrator_task.selected_task and request.problem_context.meta_task_progress.current_step_guidance. "
                "Address the selected orchestrator task only; everything else stays deferred. "
                "If request.problem_context.meta_task_progress.phase is strategy_scan, keep the branch at planning level, stay route-local and atomic, and revise only one planning claim for one route. "
                "If request.problem_context.meta_task_progress.phase is incremental_refinement, only fix one quantity, relation, approximation, or correction term, and keep the fix short. If the latest critique says the child repeated its parent, repair that by adding exactly one explicit local delta: one correction, one boundary condition, or one control parameter. The revised thought_step itself must name that delta instead of paraphrasing the parent claim. Surface the same delta in equations, quantities, boundary_conditions, or known_vars using a short marker such as active_correction, active_boundary_condition, or active_control_parameter. "
                "Do not use markdown."
            ),
        },
        "evaluation": {
            "required_keys": [
                "domain_consistency",
                "variable_grounding",
                "contextual_relevance",
                "simplicity_hint",
                "reason",
                "hard_rule_violations",
            ],
            "single_step": False,
            "prompt_fragment": (
                "You are the ToT review model. Return only a JSON object with keys domain_consistency, variable_grounding, contextual_relevance, simplicity_hint, reason, hard_rule_violations. "
                "You do not receive the full problem statement; score only against the local node state and the currently selected subtask. "
                "Use numeric values in [0,1] and an array for hard_rule_violations. Do not use markdown."
            ),
        },
        "delete-review": {
            "required_keys": ["approved", "reason", "risk_level"],
            "single_step": False,
            "prompt_fragment": (
                "You are the ToT audit model reviewing a node deletion request. Return only a JSON object with keys approved, reason, risk_level. Do not use markdown."
            ),
        },
    }

    try:
        return dict(contracts[stage])
    except KeyError as exc:
        available = ", ".join(sorted(contracts))
        raise ValueError(f"Unsupported ToT prompt-contract stage: {stage}. Available: {available}") from exc



def _skill_entry(
    func: Any,
    *,
    module: str,
    section: str,
    call_style: str,
    signature: str,
    returns: str,
    summary: str,
    keywords: Sequence[str],
) -> Dict[str, Any]:
    return {
        "callable": func,
        "module": module,
        "section": section,
        "call_style": call_style,
        "signature": signature,
        "returns": returns,
        "summary": summary,
        "keywords": list(keywords),
    }


SKILL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "lagrangian_equations": _skill_entry(
        lagrangian_equations,
        module="Theoretical Mechanics",
        section="1.1",
        call_style="params_dict",
        signature="lagrangian_equations(params: dict) -> dict",
        returns="dict",
        summary="Derive Euler-Lagrange equations from kinetic and potential energy expressions.",
        keywords=("lagrangian", "euler-lagrange", "mechanics", "equations of motion"),
    ),
    "hamiltonian_equations": _skill_entry(
        hamiltonian_equations,
        module="Theoretical Mechanics",
        section="1.2",
        call_style="params_dict",
        signature="hamiltonian_equations(params: dict) -> dict",
        returns="dict",
        summary="Perform a Legendre transform and build canonical Hamilton equations.",
        keywords=("hamiltonian", "legendre transform", "canonical equations", "momenta"),
    ),
    "inertia_tensor": _skill_entry(
        inertia_tensor,
        module="Theoretical Mechanics",
        section="1.3",
        call_style="params_dict",
        signature="inertia_tensor(params: dict) -> dict",
        returns="dict",
        summary="Compute the inertia tensor and its principal moments and axes.",
        keywords=("inertia tensor", "rigid body", "principal axes", "diagonalization"),
    ),
    "euler_rigid_body_equations": _skill_entry(
        euler_rigid_body_equations,
        module="Theoretical Mechanics",
        section="1.3",
        call_style="params_dict",
        signature="euler_rigid_body_equations(params: dict) -> dict",
        returns="dict",
        summary="Construct Euler rigid-body rotation equations in the principal-axis frame.",
        keywords=("rigid body", "euler equations", "torque", "angular velocity"),
    ),
    "vector_divergence": _skill_entry(
        vector_divergence,
        module="Electrodynamics",
        section="2.1",
        call_style="direct_args",
        signature="vector_divergence(F, coord_system='cartesian', coords=None) -> Expr",
        returns="Expr",
        summary="Compute divergence in Cartesian, cylindrical, or spherical coordinates.",
        keywords=("vector calculus", "divergence", "field operator", "coordinates"),
    ),
    "vector_curl": _skill_entry(
        vector_curl,
        module="Electrodynamics",
        section="2.1",
        call_style="direct_args",
        signature="vector_curl(F, coord_system='cartesian', coords=None) -> Matrix",
        returns="Matrix",
        summary="Compute curl in orthogonal coordinate systems.",
        keywords=("vector calculus", "curl", "rotation", "coordinates"),
    ),
    "vector_gradient": _skill_entry(
        vector_gradient,
        module="Electrodynamics",
        section="2.1",
        call_style="direct_args",
        signature="vector_gradient(phi, coord_system='cartesian', coords=None) -> Matrix",
        returns="Matrix",
        summary="Compute the gradient of a scalar field in an orthogonal coordinate system.",
        keywords=("vector calculus", "gradient", "scalar field", "coordinates"),
    ),
    "scalar_laplacian": _skill_entry(
        scalar_laplacian,
        module="Electrodynamics",
        section="2.1",
        call_style="direct_args",
        signature="scalar_laplacian(phi, coord_system='cartesian', coords=None) -> Expr",
        returns="Expr",
        summary="Compute the scalar Laplacian in orthogonal coordinates.",
        keywords=("laplacian", "vector calculus", "scalar field", "coordinates"),
    ),
    "maxwell_equations_check": _skill_entry(
        maxwell_equations_check,
        module="Electrodynamics",
        section="2.2",
        call_style="params_dict",
        signature="maxwell_equations_check(params: dict) -> dict",
        returns="dict",
        summary="Check whether E, B, rho, and J satisfy Maxwell equations.",
        keywords=("maxwell", "electromagnetism", "field check", "gauss", "ampere", "faraday"),
    ),
    "fields_from_potentials": _skill_entry(
        fields_from_potentials,
        module="Electrodynamics",
        section="2.3",
        call_style="params_dict",
        signature="fields_from_potentials(params: dict) -> dict",
        returns="dict",
        summary="Construct E and B from scalar and vector potentials and test gauge conditions.",
        keywords=("potential", "gauge", "lorenz gauge", "coulomb gauge", "electromagnetism"),
    ),
    "poynting_vector": _skill_entry(
        poynting_vector,
        module="Electrodynamics",
        section="2.4",
        call_style="params_dict",
        signature="poynting_vector(params: dict) -> dict",
        returns="dict",
        summary="Compute the Poynting vector and electromagnetic energy density.",
        keywords=("poynting", "energy flux", "electromagnetism", "energy density"),
    ),
    "em_wave_dispersion": _skill_entry(
        em_wave_dispersion,
        module="Electrodynamics",
        section="2.5",
        call_style="params_dict",
        signature="em_wave_dispersion(params: dict) -> dict",
        returns="dict",
        summary="Build electromagnetic wave dispersion relations in vacuum, dielectrics, or conductors.",
        keywords=("dispersion", "electromagnetic wave", "dielectric", "conductor", "refractive index"),
    ),
    "commutator": _skill_entry(
        commutator,
        module="Quantum Mechanics",
        section="3.1",
        call_style="direct_args",
        signature="commutator(A, B, simplify_result=True) -> Any",
        returns="Any",
        summary="Compute a commutator for matrices, symbolic expressions, or quantum operators.",
        keywords=("commutator", "operator algebra", "quantum", "matrix"),
    ),
    "schrodinger_1d": _skill_entry(
        schrodinger_1d,
        module="Quantum Mechanics",
        section="3.2",
        call_style="params_dict",
        signature="schrodinger_1d(params: dict) -> dict",
        returns="dict",
        summary="Solve or build the 1D stationary Schrodinger equation for preset or custom potentials.",
        keywords=("schrodinger", "1D quantum", "potential well", "harmonic oscillator", "eigenstates"),
    ),
    "pauli_matrices": _skill_entry(
        pauli_matrices,
        module="Quantum Mechanics",
        section="3.3",
        call_style="zero_arg",
        signature="pauli_matrices() -> dict",
        returns="dict",
        summary="Return the Pauli matrices and the 2x2 identity matrix.",
        keywords=("pauli", "spin", "sigma matrices", "quantum"),
    ),
    "pauli_algebra": _skill_entry(
        pauli_algebra,
        module="Quantum Mechanics",
        section="3.3",
        call_style="params_dict",
        signature="pauli_algebra(params: dict) -> dict",
        returns="dict",
        summary="Apply common Pauli-matrix algebra operations and eigenanalysis.",
        keywords=("pauli", "spin algebra", "eigenvectors", "commutator", "anticommutator"),
    ),
    "angular_momentum_eigenstates": _skill_entry(
        angular_momentum_eigenstates,
        module="Quantum Mechanics",
        section="3.4",
        call_style="params_dict",
        signature="angular_momentum_eigenstates(params: dict) -> dict",
        returns="dict",
        summary="Build spherical-harmonic angular-momentum eigenstates and operator checks.",
        keywords=("angular momentum", "spherical harmonics", "L2", "Lz", "quantum"),
    ),
    "perturbation_first_order": _skill_entry(
        perturbation_first_order,
        module="Quantum Mechanics",
        section="3.5",
        call_style="params_dict",
        signature="perturbation_first_order(params: dict) -> dict",
        returns="dict",
        summary="Compute first-order non-degenerate stationary perturbation energy shifts.",
        keywords=("perturbation", "first order", "quantum", "energy correction"),
    ),
    "thermodynamic_potentials": _skill_entry(
        thermodynamic_potentials,
        module="Thermodynamics and Statistical Physics",
        section="4.1",
        call_style="params_dict",
        signature="thermodynamic_potentials(params: dict) -> dict",
        returns="dict",
        summary="Relate thermodynamic potentials, natural variables, and Maxwell relations.",
        keywords=("thermodynamics", "maxwell relations", "helmholtz", "gibbs", "enthalpy"),
    ),
    "thermodynamic_partial": _skill_entry(
        thermodynamic_partial,
        module="Thermodynamics and Statistical Physics",
        section="4.1",
        call_style="params_expr",
        signature="thermodynamic_partial(params: dict) -> Expr",
        returns="Expr",
        summary="Evaluate constrained thermodynamic partial derivatives by Jacobians.",
        keywords=("partial derivative", "jacobian", "thermodynamics", "constrained derivative"),
    ),
    "partition_function": _skill_entry(
        partition_function,
        module="Thermodynamics and Statistical Physics",
        section="4.2-4.3",
        call_style="params_dict",
        signature="partition_function(params: dict) -> dict",
        returns="dict",
        summary="Build a canonical partition function and derived thermodynamic observables.",
        keywords=("partition function", "canonical ensemble", "free energy", "entropy", "heat capacity"),
    ),
    "statistical_distributions": _skill_entry(
        statistical_distributions,
        module="Thermodynamics and Statistical Physics",
        section="4.4",
        call_style="params_dict",
        signature="statistical_distributions(params: dict) -> dict",
        returns="dict",
        summary="Return Maxwell-Boltzmann, Fermi-Dirac, or Bose-Einstein occupation formulas.",
        keywords=("maxwell-boltzmann", "fermi-dirac", "bose-einstein", "statistics"),
    ),
    "lorentz_boost_matrix": _skill_entry(
        lorentz_boost_matrix,
        module="Special Relativity",
        section="5.1",
        call_style="params_dict",
        signature="lorentz_boost_matrix(params: dict) -> dict",
        returns="dict",
        summary="Construct Lorentz boost matrices along x or in arbitrary directions.",
        keywords=("lorentz", "boost", "special relativity", "matrix"),
    ),
    "lorentz_transform_event": _skill_entry(
        lorentz_transform_event,
        module="Special Relativity",
        section="5.2",
        call_style="params_dict",
        signature="lorentz_transform_event(params: dict) -> dict",
        returns="dict",
        summary="Transform a spacetime event with a Lorentz boost.",
        keywords=("lorentz transform", "event", "spacetime", "special relativity"),
    ),
    "four_vector_inner_product": _skill_entry(
        four_vector_inner_product,
        module="Special Relativity",
        section="5.3",
        call_style="direct_args",
        signature="four_vector_inner_product(A, B) -> Expr",
        returns="Expr",
        summary="Evaluate the Minkowski inner product with signature (+,-,-,-).",
        keywords=("four-vector", "minkowski", "inner product", "invariant"),
    ),
    "relativistic_energy_momentum": _skill_entry(
        relativistic_energy_momentum,
        module="Special Relativity",
        section="5.4",
        call_style="params_dict",
        signature="relativistic_energy_momentum(params: dict) -> dict",
        returns="dict",
        summary="Solve relativistic energy-momentum relations and construct four-momentum.",
        keywords=("energy-momentum", "four-momentum", "relativity", "gamma factor"),
    ),
    "velocity_addition": _skill_entry(
        velocity_addition,
        module="Special Relativity",
        section="5.5",
        call_style="params_dict",
        signature="velocity_addition(params: dict) -> dict",
        returns="dict",
        summary="Apply 1D relativistic velocity addition or its inverse relation.",
        keywords=("velocity addition", "relativity", "boost", "kinematics"),
    ),
    "multi_slit_intensity": _skill_entry(
        multi_slit_intensity,
        module="Optics and Waves",
        section="6.1",
        call_style="params_dict",
        signature="multi_slit_intensity(params: dict) -> dict",
        returns="dict",
        summary="Compute multi-slit Fraunhofer interference with optional diffraction envelope.",
        keywords=("interference", "multi-slit", "fraunhofer", "diffraction"),
    ),
    "grating_equation": _skill_entry(
        grating_equation,
        module="Optics and Waves",
        section="6.2",
        call_style="params_dict",
        signature="grating_equation(params: dict) -> dict",
        returns="dict",
        summary="Solve the diffraction grating equation for one unknown quantity.",
        keywords=("grating", "diffraction", "wavelength", "angle"),
    ),
    "single_slit_diffraction": _skill_entry(
        single_slit_diffraction,
        module="Optics and Waves",
        section="6.3",
        call_style="params_dict",
        signature="single_slit_diffraction(params: dict) -> dict",
        returns="dict",
        summary="Compute single-slit Fraunhofer diffraction intensity and minima conditions.",
        keywords=("single slit", "diffraction", "fraunhofer", "optics"),
    ),
    "ray_translation_matrix": _skill_entry(
        ray_translation_matrix,
        module="Optics and Waves",
        section="6.4",
        call_style="direct_args",
        signature="ray_translation_matrix(d) -> Matrix",
        returns="Matrix",
        summary="Return the ABCD matrix for free-space ray translation.",
        keywords=("ray optics", "ABCD", "translation", "matrix optics"),
    ),
    "ray_refraction_matrix": _skill_entry(
        ray_refraction_matrix,
        module="Optics and Waves",
        section="6.4",
        call_style="direct_args",
        signature="ray_refraction_matrix(n1, n2, R=None) -> Matrix",
        returns="Matrix",
        summary="Return the ABCD matrix for planar or spherical refraction.",
        keywords=("ray optics", "refraction", "ABCD", "interface"),
    ),
    "thin_lens_matrix": _skill_entry(
        thin_lens_matrix,
        module="Optics and Waves",
        section="6.4",
        call_style="direct_args",
        signature="thin_lens_matrix(f) -> Matrix",
        returns="Matrix",
        summary="Return the ABCD matrix for a thin lens.",
        keywords=("thin lens", "ABCD", "matrix optics", "focus"),
    ),
    "mirror_matrix": _skill_entry(
        mirror_matrix,
        module="Optics and Waves",
        section="6.4",
        call_style="direct_args",
        signature="mirror_matrix(R) -> Matrix",
        returns="Matrix",
        summary="Return the ABCD matrix for a spherical mirror.",
        keywords=("mirror", "ABCD", "matrix optics", "reflection"),
    ),
    "optical_system": _skill_entry(
        optical_system,
        module="Optics and Waves",
        section="6.4",
        call_style="params_dict",
        signature="optical_system(params: dict) -> dict",
        returns="dict",
        summary="Compose multi-element ABCD systems and derive effective optical parameters.",
        keywords=("optical system", "ABCD", "equivalent focal length", "principal plane"),
    ),
    "noether_conservation": _skill_entry(
        noether_conservation,
        module="Extended Utilities",
        section="7.1",
        call_style="params_dict",
        signature="noether_conservation(params: dict) -> dict",
        returns="dict",
        summary="Detect cyclic coordinates, energy conservation, and user-specified symmetry charges.",
        keywords=("noether", "conservation law", "symmetry", "cyclic coordinate"),
    ),
    "effective_potential_analysis": _skill_entry(
        effective_potential_analysis,
        module="Extended Utilities",
        section="7.2",
        call_style="params_dict",
        signature="effective_potential_analysis(params: dict) -> dict",
        returns="dict",
        summary="Analyze equilibrium points, stability, and small oscillations of an effective potential.",
        keywords=("effective potential", "stability", "equilibrium", "small oscillation"),
    ),
    "special_functions": _skill_entry(
        special_functions,
        module="Extended Utilities",
        section="7.3",
        call_style="params_dict",
        signature="special_functions(params: dict) -> dict",
        returns="dict",
        summary="Construct common special functions such as Legendre, Bessel, Hermite, and Ylm.",
        keywords=("special functions", "legendre", "bessel", "hermite", "spherical harmonics"),
    ),
    "error_propagation": _skill_entry(
        error_propagation,
        module="Extended Utilities",
        section="7.4",
        call_style="params_dict",
        signature="error_propagation(params: dict) -> dict",
        returns="dict",
        summary="Apply first-order Gaussian error propagation to symbolic formulas.",
        keywords=("error propagation", "uncertainty", "gaussian errors", "measurement"),
    ),
    "dimensional_analysis": _skill_entry(
        dimensional_analysis,
        module="Extended Utilities",
        section="7.5",
        call_style="params_dict",
        signature="dimensional_analysis(params: dict) -> dict",
        returns="dict",
        summary="Perform Buckingham Pi dimensional analysis from quantity dimensions.",
        keywords=("dimensional analysis", "buckingham pi", "units", "dimension"),
    ),
    "thick_lens": _skill_entry(
        thick_lens,
        module="Extended Utilities",
        section="7.6",
        call_style="params_dict",
        signature="thick_lens(params: dict) -> dict",
        returns="dict",
        summary="Construct a thick-lens ABCD matrix and effective focal properties.",
        keywords=("thick lens", "ABCD", "optical power", "principal plane"),
    ),
    "aberrations": _skill_entry(
        aberrations,
        module="Extended Utilities",
        section="7.6",
        call_style="params_dict",
        signature="aberrations(params: dict) -> dict",
        returns="dict",
        summary="Estimate paraxial spherical and chromatic aberration measures.",
        keywords=("aberration", "chromatic", "spherical aberration", "optics"),
    ),
    "jones_calculus": _skill_entry(
        jones_calculus,
        module="Extended Utilities",
        section="7.7",
        call_style="params_dict",
        signature="jones_calculus(params: dict) -> dict",
        returns="dict",
        summary="Build Jones vectors and matrices and propagate polarization states.",
        keywords=("jones", "polarization", "waveplate", "polarizer"),
    ),
    "stokes_mueller": _skill_entry(
        stokes_mueller,
        module="Extended Utilities",
        section="7.7",
        call_style="params_dict",
        signature="stokes_mueller(params: dict) -> dict",
        returns="dict",
        summary="Convert Jones to Stokes form and apply Mueller-matrix optics.",
        keywords=("stokes", "mueller", "polarization", "optics"),
    ),
    "doppler_classical": _skill_entry(
        doppler_classical,
        module="Extended Utilities",
        section="7.8",
        call_style="params_dict",
        signature="doppler_classical(params: dict) -> dict",
        returns="dict",
        summary="Compute the classical Doppler shift for moving source and observer.",
        keywords=("doppler", "classical waves", "frequency shift", "acoustics"),
    ),
    "standing_wave_modes": _skill_entry(
        standing_wave_modes,
        module="Extended Utilities",
        section="7.8",
        call_style="params_dict",
        signature="standing_wave_modes(params: dict) -> dict",
        returns="dict",
        summary="Enumerate 1D standing-wave mode frequencies for common boundary types.",
        keywords=("standing wave", "boundary condition", "mode", "resonance"),
    ),
    "continuity_equation": _skill_entry(
        continuity_equation,
        module="Fluid Mechanics",
        section="8.1",
        call_style="params_dict",
        signature="continuity_equation(params: dict) -> dict",
        returns="dict",
        summary="Check compressible or incompressible mass conservation.",
        keywords=("continuity", "fluid", "mass conservation", "incompressible"),
    ),
    "bernoulli_equation": _skill_entry(
        bernoulli_equation,
        module="Fluid Mechanics",
        section="8.2",
        call_style="params_dict",
        signature="bernoulli_equation(params: dict) -> dict",
        returns="dict",
        summary="Build and solve Bernoulli relations between two flow points.",
        keywords=("bernoulli", "fluid", "pressure", "flow speed", "height"),
    ),
    "euler_fluid_equation": _skill_entry(
        euler_fluid_equation,
        module="Fluid Mechanics",
        section="8.3",
        call_style="params_dict",
        signature="euler_fluid_equation(params: dict) -> dict",
        returns="dict",
        summary="Compute residuals of the inviscid Euler fluid equations in Cartesian coordinates.",
        keywords=("euler fluid", "inviscid", "fluid equations", "residual"),
    ),
    "navier_stokes_check": _skill_entry(
        navier_stokes_check,
        module="Fluid Mechanics",
        section="8.3",
        call_style="params_dict",
        signature="navier_stokes_check(params: dict) -> dict",
        returns="dict",
        summary="Compute incompressible Navier-Stokes residuals in Cartesian coordinates.",
        keywords=("navier-stokes", "fluid", "viscosity", "residual", "incompressible"),
    ),
    "vorticity_and_stream": _skill_entry(
        vorticity_and_stream,
        module="Fluid Mechanics",
        section="8.4",
        call_style="params_dict",
        signature="vorticity_and_stream(params: dict) -> dict",
        returns="dict",
        summary="Convert stream functions to velocity fields or compute vorticity from a velocity field.",
        keywords=("vorticity", "stream function", "fluid", "incompressible"),
    ),
    "reynolds_number": _skill_entry(
        reynolds_number,
        module="Fluid Mechanics",
        section="8.5",
        call_style="params_dict",
        signature="reynolds_number(params: dict) -> dict",
        returns="dict",
        summary="Compute Reynolds number and classify a laminar or turbulent regime when possible.",
        keywords=("reynolds", "fluid", "laminar", "turbulent", "regime"),
    ),
    "poiseuille_flow": _skill_entry(
        poiseuille_flow,
        module="Fluid Mechanics",
        section="8.6",
        call_style="params_dict",
        signature="poiseuille_flow(params: dict) -> dict",
        returns="dict",
        summary="Return pipe-flow velocity profile, volumetric flow, and related quantities.",
        keywords=("poiseuille", "pipe flow", "laminar", "fluid", "velocity profile"),
    ),
    "stokes_drag": _skill_entry(
        stokes_drag,
        module="Fluid Mechanics",
        section="8.7",
        call_style="params_dict",
        signature="stokes_drag(params: dict) -> dict",
        returns="dict",
        summary="Compute Stokes drag and terminal settling speed for a sphere.",
        keywords=("stokes drag", "terminal velocity", "viscous drag", "sphere"),
    ),
    "sound_speed": _skill_entry(
        sound_speed,
        module="Fluid Mechanics",
        section="8.8",
        call_style="params_dict",
        signature="sound_speed(params: dict) -> dict",
        returns="dict",
        summary="Compute sound speed in an ideal gas or a compressible fluid medium.",
        keywords=("sound speed", "acoustics", "ideal gas", "bulk modulus"),
    ),
    "surface_tension": _skill_entry(
        surface_tension,
        module="Fluid Mechanics",
        section="8.9",
        call_style="params_dict",
        signature="surface_tension(params: dict) -> dict",
        returns="dict",
        summary="Evaluate Young-Laplace pressure jumps and capillary-rise relations.",
        keywords=("surface tension", "capillary rise", "young-laplace", "fluid interface"),
    ),
    "tot_hard_rule_check": _skill_entry(
        tot_hard_rule_check,
        module="Extended Utilities",
        section="7.x",
        call_style="params_dict",
        signature="tot_hard_rule_check(params: dict) -> dict",
        returns="dict",
        summary="Apply deterministic veto rules such as missing variables, forbidden patterns, and dimension mismatches.",
        keywords=("hard rule", "validation", "dimension check", "veto", "tot"),
    ),
    "tot_validation_plugin_bundle": _skill_entry(
        tot_validation_plugin_bundle,
        module="Extended Utilities",
        section="7.x",
        call_style="params_dict",
        signature="tot_validation_plugin_bundle(params: dict) -> dict",
        returns="dict",
        summary="Build a pluggable validation bundle with skill-specific deterministic hard-rule parameters.",
        keywords=("plugin", "validation", "hard rule", "deterministic", "tot"),
    ),
    "tot_domain_plugin_bundle": _skill_entry(
        tot_domain_plugin_bundle,
        module="Extended Utilities",
        section="7.x",
        call_style="params_dict",
        signature="tot_domain_plugin_bundle(params: dict) -> dict",
        returns="dict",
        summary="Build a pluggable domain bundle with knowledge scope, representative LaTeX formulas, and route seeds.",
        keywords=("plugin", "domain", "latex", "meta-analysis", "route planning", "knowledge scope"),
    ),
    "tot_stage_prompt_contract": _skill_entry(
        tot_stage_prompt_contract,
        module="Extended Utilities",
        section="7.x",
        call_style="params_dict",
        signature="tot_stage_prompt_contract(params: dict) -> dict",
        returns="dict",
        summary="Return stage-specific JSON format contracts and prompt fragments for ToT chat stages.",
        keywords=("tot", "prompt", "json", "contract", "single-step", "format"),
    ),
}


def get_skill_entry(skill_name: str) -> Dict[str, Any]:
    """Return the registry entry for a public skill name."""

    try:
        return SKILL_REGISTRY[skill_name]
    except KeyError as exc:
        available = ", ".join(sorted(SKILL_REGISTRY))
        raise KeyError(f"Unknown skill: {skill_name}. Available skills: {available}") from exc


def search_skills(
    query: str,
    *,
    module: Optional[str] = None,
    call_style: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Search public skills by name, module, summary, signature, or keywords."""

    query_text = query.strip().lower()
    if not query_text:
        raise ValueError("search_skills requires a non-empty query.")

    matches: List[Dict[str, Any]] = []
    for skill_name, entry in SKILL_REGISTRY.items():
        if module is not None and entry["module"] != module:
            continue
        if call_style is not None and entry["call_style"] != call_style:
            continue

        haystack = " ".join(
            [
                skill_name,
                entry["module"],
                entry["section"],
                entry["signature"],
                entry["returns"],
                entry["summary"],
                *entry["keywords"],
            ]
        ).lower()
        if query_text not in haystack:
            continue

        match = {key: value for key, value in entry.items() if key != "callable"}
        match["name"] = skill_name
        matches.append(match)

    matches.sort(key=lambda item: (item["module"], item["section"], item["name"]))
    if limit is None:
        return matches
    return matches[:limit]


def invoke_skill(
    skill_name: str,
    payload: Optional[Dict[str, Any]] = None,
    *,
    include_trace: bool = False,
) -> Any:
    """Invoke a registered public skill with call-style-aware dispatch.

    Parameters
    ----------
    skill_name
        Public skill name registered in ``SKILL_REGISTRY``.
    payload
        Unified invocation payload.

        - ``params_dict`` / ``params_expr`` skills expect a plain parameter dict.
        - ``zero_arg`` skills ignore an omitted payload and reject non-empty input.
        - ``direct_args`` skills expect either:
          ``{"args": [...], "kwargs": {...}}`` or an empty payload for a no-arg call.
    include_trace
        If ``True``, return both the result and a structured invocation trace.
    """

    entry = get_skill_entry(skill_name)
    call_style = entry["call_style"]
    skill_callable = entry["callable"]
    normalized_payload = {} if payload is None else dict(payload)

    if call_style in ("params_dict", "params_expr"):
        result = skill_callable(normalized_payload)
        trace = {
            "skill_name": skill_name,
            "call_style": call_style,
            "payload": normalized_payload,
        }
    elif call_style == "zero_arg":
        if normalized_payload:
            raise ValueError(f"Skill '{skill_name}' does not accept a payload.")
        result = skill_callable()
        trace = {
            "skill_name": skill_name,
            "call_style": call_style,
            "payload": {},
        }
    elif call_style == "direct_args":
        args = normalized_payload.get("args", [])
        kwargs = normalized_payload.get("kwargs", {})
        extra_keys = set(normalized_payload) - {"args", "kwargs"}
        if extra_keys:
            extra_list = ", ".join(sorted(extra_keys))
            raise ValueError(
                f"Direct-argument skill '{skill_name}' only accepts 'args' and 'kwargs'; got: {extra_list}"
            )
        if not isinstance(args, (list, tuple)):
            raise TypeError(f"Skill '{skill_name}' expects payload['args'] to be a list or tuple.")
        if not isinstance(kwargs, dict):
            raise TypeError(f"Skill '{skill_name}' expects payload['kwargs'] to be a dictionary.")
        result = skill_callable(*args, **kwargs)
        trace = {
            "skill_name": skill_name,
            "call_style": call_style,
            "args": list(args),
            "kwargs": dict(kwargs),
        }
    else:
        raise ValueError(f"Unsupported call style in registry for skill '{skill_name}': {call_style}")

    if not include_trace:
        return result
    return {
        "result": result,
        "trace": trace,
    }


PUBLIC_SKILL_NAMES: Tuple[str, ...] = tuple(SKILL_REGISTRY.keys())