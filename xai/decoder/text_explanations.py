from __future__ import annotations

import argparse
import json
import math
import sys

from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from domain.constants import (
    FEATURE_LABELS_FR as FEATURE_LABELS,
    CONSTRAINT_LABELS_FR as CONSTRAINT_LABELS,
    COUNTERFACTUAL_FEATURE_LABELS_FR as COUNTERFACTUAL_FEATURE_LABELS,
)


def _parse_step_list(raw: str | None, max_steps: int) -> List[int]:
    if max_steps <= 0:
        return []
    if raw is None:
        # Default shortlist spanning early/mid/late behavior.
        candidates = [0, 1, 2, 5, 10, max_steps - 1]
        return sorted({s for s in candidates if 0 <= s < max_steps})
    if raw.strip().lower() == "all":
        return list(range(max_steps))
    out: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        step = int(token)
        if 0 <= step < max_steps:
            out.append(step)
    return sorted(set(out))


def _parse_instance_selection(raw: str | None, max_instances: int) -> Tuple[List[int], bool]:
    if max_instances <= 0:
        return [], False
    if raw is None:
        return [0], False
    token = raw.strip().lower()
    if token == "all":
        return list(range(max_instances)), True
    idx = int(raw)
    if not (0 <= idx < max_instances):
        raise ValueError(
            f"Invalid instance index {idx}. Available range: [0, {max_instances-1}]"
        )
    return [idx], False


def _node_before_step(actions: List[int], step: int) -> int:
    if step <= 0:
        return 0
    return int(actions[step - 1])


def _visited_customers(actions: List[int], step: int) -> set[int]:
    visited = set()
    for a in actions[:step]:
        a = int(a)
        if a > 0:
            visited.add(a)
    return visited


def _rank_in_top(node: int, top_nodes: List[int]) -> int | None:
    for i, n in enumerate(top_nodes, start=1):
        if int(n) == int(node):
            return i
    return None


def _dist(locs: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(locs[a] - locs[b]))


def _percentile_position(value: float, reference: np.ndarray) -> float:
    if reference.size == 0:
        return 1.0
    return float((reference <= value).mean())


def _format_reasons(reasons: List[str]) -> str:
    if not reasons:
        return "Raison principale difficile a isoler avec ce niveau de trace."
    if len(reasons) == 1:
        return reasons[0] + "."
    return ", ".join(reasons[:-1]) + " et " + reasons[-1] + "."


def _feature_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def _constraint_label(name: str) -> str:
    return CONSTRAINT_LABELS.get(name, name)


def _constraint_item_name(item: dict) -> str:
    return str(
        item.get("constraint_state", item.get("constraint", ""))
    ).strip()


def _constraint_item_family(item: dict) -> str:
    return str(item.get("constraint_family", "")).strip()


def _constraint_item_label(item: dict) -> str:
    state_name = _constraint_item_name(item)
    family_name = _constraint_item_family(item)
    state_label = _constraint_label(state_name)
    if family_name and family_name != state_name:
        family_label = _constraint_label(family_name)
        return f"{family_label} / {state_label}"
    return state_label


def _format_top_features(top_features: List[dict]) -> str:
    parts = []
    for item in top_features:
        name = _feature_label(str(item.get("feature", "")))
        share = float(item.get("share", 0.0)) * 100.0
        parts.append(f"{name} ({share:.1f}%)")
    return ", ".join(parts)


def _format_top_constraints(top_constraints: List[dict]) -> str:
    parts = []
    for idx, item in enumerate(top_constraints, start=1):
        name = _constraint_item_label(item)
        share = float(item.get("share", 0.0)) * 100.0
        if idx <= 3:
            parts.append(f"{name} ({share:.1f}%)")
        else:
            parts.append(f"{name} ({share:.1f}%, hors top 3)")
    return ", ".join(parts)


def _format_structural_constraints(items: List[dict]) -> str:
    labels = [
        _constraint_item_label(item)
        for item in items or []
        if isinstance(item, dict)
        and str(item.get("constraint_state", item.get("constraint", ""))).strip()
    ]
    return ", ".join(labels)


def _trace_decoder_dynamic_states(trace: dict) -> List[object]:
    return (
        trace.get("decoder_dynamic_states", []) or []
    ) or (
        trace.get("decoder_state_constraint_states", []) or []
    )


def _trace_instance_structural_constraints(trace: dict) -> List[dict]:
    return trace.get("instance_structural_constraints", []) or []


def _constraint_payload_reading_lines(
    top_constraints: List[dict],
    source_label: str,
) -> List[str]:
    lines: List[str] = []
    constraint_shares: List[Tuple[str, float]] = []
    for item in top_constraints or []:
        try:
            share = max(float(item.get("share", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
        constraint_shares.append((_constraint_item_label(item), share))

    if not constraint_shares:
        return lines

    top_name, top_share = constraint_shares[0]
    second_share = constraint_shares[1][1] if len(constraint_shares) > 1 else 0.0
    third_share = constraint_shares[2][1] if len(constraint_shares) > 2 else 0.0

    if top_share >= 0.6 and top_share >= second_share + 0.15:
        lines.append(
            f"Lecture ({source_label}): la décision paraît surtout dominée par {top_name.lower()}"
        )
    elif len(constraint_shares) >= 2 and abs(top_share - second_share) <= 0.1:
        second_name = constraint_shares[1][0]
        lines.append(
            f"Lecture ({source_label}): la décision semble surtout arbitrer entre {top_name.lower()} et {second_name.lower()}"
        )
    elif len(constraint_shares) >= 3 and (top_share - third_share) <= 0.15:
        lines.append(
            f"Lecture ({source_label}): plusieurs familles de contraintes contribuent de façon comparable"
        )
    return lines


def _abductive_reading_lines(
    top_constraints: List[dict], top_features: List[dict]
) -> List[str]:
    lines: List[str] = []
    lines.extend(_constraint_payload_reading_lines(top_constraints, "attribution d'entrée"))

    feature_shares: List[Tuple[str, float]] = []
    for item in top_features or []:
        try:
            share = max(float(item.get("share", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
        feature_shares.append((_feature_label(str(item.get("feature", ""))), share))

    if feature_shares:
        top_feature, top_share = feature_shares[0]
        second_share = feature_shares[1][1] if len(feature_shares) > 1 else 0.0
        if top_share >= 0.5 and top_share >= second_share + 0.15:
            lines.append(
                f"Lecture: côté variables d'entrée, {top_feature.lower()} porte l'essentiel de l'explication"
            )
        elif len(feature_shares) >= 2 and abs(top_share - second_share) <= 0.1:
            second_feature = feature_shares[1][0]
            lines.append(
                f"Lecture: côté variables d'entrée, {top_feature.lower()} et {second_feature.lower()} pèsent de façon proche"
            )

    return lines


def _contrastive_source_label(source: str) -> str:
    if source == "full_feasible":
        return "alternative faisable"
    if source == "policy_masked":
        return "alternative masque politique"
    if source == "policy_next":
        return "alternative de politique"
    if source == "strictly_feasible":
        return "alternative strictement faisable"
    return "aucune alternative"


def _format_contrastive_alternative(
    title: str,
    alt_action: int,
    alt_feasible: bool,
    alt_recourse: bool,
    logit_gap: float,
    logprob_gap: float,
    constraints_suffix: str | None = None,
) -> List[str]:
    if alt_action < 0:
        return []
    alt_label = "depot" if alt_action == 0 else f"noeud {alt_action}"
    chunks = [f"{title}: {alt_label}"]
    if np.isfinite(logit_gap):
        chunks.append(f"marge logit {logit_gap:+.3f}")
    if np.isfinite(logprob_gap):
        chunks.append(f"marge logprob {logprob_gap:+.3f}")
    if not alt_feasible and alt_action > 0:
        chunks.append("non faisable sous contraintes complètes")
    if alt_recourse:
        chunks.append("impliquerait recours")
    if constraints_suffix:
        chunks.append(constraints_suffix)
    return [", ".join(chunks)]


def _format_variant_constraints(active_constraints: List[str]) -> str:
    if not active_constraints:
        return "base_vrp"
    return ", ".join(str(x) for x in active_constraints)


def _format_counterfactual(counterfactual: dict) -> str:
    feature = COUNTERFACTUAL_FEATURE_LABELS.get(
        str(counterfactual.get("feature", "")), str(counterfactual.get("feature", ""))
    )
    direction = str(counterfactual.get("direction", "increase"))
    delta = float(counterfactual.get("delta", 0.0))
    target_action = int(counterfactual.get("target_action", -1))
    node_idx = counterfactual.get("node", None)
    status = str(counterfactual.get("status", "approximate"))
    target_label = "depot" if target_action == 0 else f"noeud {target_action}"

    if node_idx is not None:
        subject = f"{feature} du noeud {int(node_idx)}"
    else:
        subject = feature

    verb = "augmentait" if direction == "increase" else "diminuait"
    prefix = "+" if direction == "increase" else "-"

    if status == "switch":
        return (
            f"Contrefactuel actionnable: si {subject} {verb} de {prefix}{delta:.3f}, "
            f"le modèle choisirait {target_label}"
        )
    if status == "make_feasible":
        return (
            f"Contrefactuel faisabilité: si {subject} {verb} de {prefix}{delta:.3f}, "
            f"{target_label} deviendrait faisable sans encore devenir le choix principal"
        )
    return (
        f"Contrefactuel local (approximation): si {subject} {verb} de {prefix}{delta:.3f}, "
        f"l'écart avec {target_label} se refermerait sans bascule vérifiée"
    )


def _contrastive_reading_lines(
    alt_action: int,
    alt_source: str,
    alt_feasible: bool,
    alt_recourse: bool,
    logit_gap: float,
) -> List[str]:
    if alt_action < 0 or alt_source == "none":
        return []

    lines: List[str] = []
    if math.isfinite(logit_gap):
        abs_gap = abs(float(logit_gap))
        if abs_gap >= 5.0:
            lines.append(
                "Lecture: marge forte; le modèle préfère nettement l'action choisie à la meilleure alternative"
            )
        elif abs_gap >= 2.0:
            lines.append(
                "Lecture: marge intermédiaire; l'action choisie domine l'alternative mais sans écrasement"
            )
        else:
            lines.append(
                "Lecture: marge faible; le choix est relativement serré face à la meilleure alternative"
            )
    if not alt_feasible and alt_action > 0:
        lines.append(
            "Lecture: l'alternative perd aussi parce qu'elle n'est pas pleinement faisable"
        )
    if alt_recourse:
        lines.append(
            "Lecture: l'alternative resterait possible, mais au prix d'un recours"
        )
    return lines


def _counterfactual_reading_lines(counterfactual: dict) -> List[str]:
    if not isinstance(counterfactual, dict) or not counterfactual:
        return []

    status = str(counterfactual.get("status", "approximate"))
    rel_raw = counterfactual.get("relative_delta", None)
    try:
        rel = abs(float(rel_raw)) if rel_raw is not None else float("nan")
    except (TypeError, ValueError):
        rel = float("nan")
    try:
        delta = abs(float(counterfactual.get("delta", float("nan"))))
    except (TypeError, ValueError):
        delta = float("nan")

    if math.isfinite(rel):
        strength_value = rel
        unit = "relative"
    else:
        strength_value = delta
        unit = "absolute"

    if not math.isfinite(strength_value):
        return []

    if unit == "relative":
        if strength_value <= 0.25:
            strength = "levier fort"
        elif strength_value <= 0.75:
            strength = "levier intermédiaire"
        else:
            strength = "levier faible"
    else:
        if strength_value <= 0.5:
            strength = "levier fort"
        elif strength_value <= 1.5:
            strength = "levier intermédiaire"
        else:
            strength = "levier faible"

    if status == "switch":
        return [
            f"Lecture: {strength}; un changement de taille "
            f"{'modeste' if strength == 'levier fort' else 'plus marquée'} suffit ici à faire basculer la décision"
        ]
    if status == "make_feasible":
        return [
            f"Lecture: {strength}; ce levier agit d'abord sur la faisabilité de l'alternative, pas encore sur la décision finale"
        ]
    return [
        f"Lecture: {strength}; c'est un signal local plausible, mais pas une bascule confirmée"
    ]


def _format_step_deletion(step_payload: dict) -> List[str]:
    deletion = step_payload.get("deletion", {}) or {}
    if not isinstance(deletion, dict) or not deletion:
        return []

    entries: List[dict] = []
    for raw_k, payload in sorted(
        deletion.items(),
        key=lambda kv: int(str(kv[0])) if str(kv[0]).isdigit() else 10**9,
    ):
        if not isinstance(payload, dict):
            continue
        try:
            k = int(raw_k)
        except (TypeError, ValueError):
            continue
        metrics: List[str] = []
        try:
            flip = float(payload.get("mean_action_flip_rate", float("nan")))
        except (TypeError, ValueError):
            flip = float("nan")
        try:
            dlogp = float(payload.get("mean_logprob_drop", float("nan")))
        except (TypeError, ValueError):
            dlogp = float("nan")
        try:
            dlogit = float(payload.get("mean_logit_drop", float("nan")))
        except (TypeError, ValueError):
            dlogit = float("nan")
        if math.isfinite(flip) or math.isfinite(dlogp) or math.isfinite(dlogit):
            entries.append(
                {
                    "k": k,
                    "flip": flip,
                    "dlogp": dlogp,
                    "dlogit": dlogit,
                }
            )
    if not entries:
        return []

    lines: List[str] = []
    best_entry = max(
        entries,
        key=lambda item: (
            item["flip"] if math.isfinite(item["flip"]) else float("-inf"),
            -int(item["k"]),
        ),
    )
    best_flip = best_entry["flip"]
    best_k = int(best_entry["k"])
    if math.isfinite(best_flip):
        if best_flip >= 0.7:
            lines.append(
                f"Lecture: les clients les plus attribués paraissent très fidèles ici; "
                f"perturber le top-{best_k} clients change la décision dans {best_flip * 100.0:.1f}% des cas"
            )
        elif best_flip >= 0.4:
            lines.append(
                f"Lecture: fidélité intermédiaire; perturber le top-{best_k} clients "
                f"change la décision dans {best_flip * 100.0:.1f}% des cas"
            )
        else:
            lines.append(
                f"Lecture: fidélité faible à modérée; même en perturbant le top-{best_k} clients, "
                f"la décision ne change que dans {best_flip * 100.0:.1f}% des cas"
            )

    flip_values = [item["flip"] for item in entries if math.isfinite(item["flip"])]
    if len(flip_values) >= 2:
        monotone = all(
            later + 1e-9 >= earlier
            for earlier, later in zip(flip_values, flip_values[1:])
        )
        if monotone:
            lines.append(
                "Lecture: comme attendu, perturber davantage de clients attribués déstabilise au moins autant la décision"
            )

    for item in entries:
        k = int(item["k"])
        metrics: List[str] = []
        if math.isfinite(item["flip"]):
            metrics.append(
                f"la décision change dans {item['flip'] * 100.0:.1f}% des cas"
            )
        if math.isfinite(item["dlogp"]):
            metrics.append(f"la log-probabilité baisse de {item['dlogp']:.3f}")
        if math.isfinite(item["dlogit"]):
            metrics.append(f"le logit baisse de {item['dlogit']:.3f}")
        if metrics:
            lines.append(f"Top-{k}: " + ", ".join(metrics))
    return lines


def _mean_constraint_share(payloads: List[object]) -> dict[str, float]:
    if not payloads:
        return {}
    groups = list(CONSTRAINT_LABELS.keys())
    values = {group: [] for group in groups}
    for payload in payloads:
        step_map = {group: 0.0 for group in groups}
        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("constraint", "")).strip() or "other"
                try:
                    share = max(float(item.get("share", 0.0)), 0.0)
                except (TypeError, ValueError):
                    continue
                target = name if name in step_map else "other"
                step_map[target] += share
        for group in groups:
            values[group].append(step_map[group])
    return {
        group: float(np.mean(entries))
        for group, entries in values.items()
        if entries
    }


def _dominant_constraint_name(shares: dict[str, float]) -> str:
    if not shares:
        return "none"
    name, value = max(shares.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
    return str(name) if float(value) > 0 else "none"


def _finite_series(values: List[object]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            cast = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cast):
            out.append(cast)
    return out


def _masked_finite_series(values: List[object], mask: List[bool]) -> List[float]:
    out: List[float] = []
    for value, keep in zip(values, mask):
        if not bool(keep):
            continue
        try:
            cast = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(cast):
            out.append(cast)
    return out


def _share_leq(values: List[float], threshold: float) -> float:
    finite = _finite_series([float(v) for v in values])
    if not finite:
        return float("nan")
    return float(sum(1 for value in finite if value <= threshold) / len(finite))


def _share_gt(values: List[float], threshold: float = 0.0) -> float:
    finite = _finite_series([float(v) for v in values])
    if not finite:
        return float("nan")
    return float(sum(1 for value in finite if value > threshold) / len(finite))


def _trajectory_reading_lines(
    depot_share: float | None,
    customer_hop: float | None,
    recourse_events: float | None,
    early_top: str | None,
    late_top: str | None,
    late_capacity_share: float | None = None,
    tw_tight_share: float | None = None,
    late_tw_tight_share: float | None = None,
    recourse_under_tw_tight_share: float | None = None,
) -> List[str]:
    lines: List[str] = []

    if depot_share is not None and math.isfinite(float(depot_share)):
        share = float(depot_share)
        if share >= 0.35:
            lines.append("Lecture: tournée très fragmentée; les retours dépôt sont fréquents")
        elif share >= 0.15:
            lines.append("Lecture: tournée partiellement fragmentée; le dépôt coupe régulièrement la séquence")
        else:
            lines.append("Lecture: tournée plutôt continue; peu de retours dépôt")

    if customer_hop is not None and math.isfinite(float(customer_hop)):
        hop = float(customer_hop)
        if hop <= 0.14:
            lines.append("Lecture: tournée compacte spatialement; les clients successifs restent proches")
        elif hop >= 0.20:
            lines.append("Lecture: tournée plus étalée; les sauts entre clients sont relativement longs")

    if recourse_events is not None and math.isfinite(float(recourse_events)):
        rec = float(recourse_events)
        if rec >= 3.0:
            lines.append("Lecture: le recours devient un mécanisme récurrent sur la trajectoire")
        elif rec >= 0.5:
            lines.append("Lecture: le recours reste ponctuel mais visible")

    if tw_tight_share is not None and math.isfinite(float(tw_tight_share)):
        share = float(tw_tight_share)
        if share >= 0.45:
            lines.append(
                "Lecture: une large part des décisions clients se fait sous forte pression de fenêtre de temps"
            )
        elif share >= 0.20:
            lines.append(
                "Lecture: la pression temporelle est présente sur une part non négligeable des décisions"
            )
    if (
        late_tw_tight_share is not None
        and math.isfinite(float(late_tw_tight_share))
        and float(late_tw_tight_share) >= 0.40
    ):
        lines.append(
            "Lecture: la fin de tournée devient nettement plus contrainte par les fenêtres de temps"
        )
    if (
        recourse_under_tw_tight_share is not None
        and math.isfinite(float(recourse_under_tw_tight_share))
        and float(recourse_under_tw_tight_share) >= 0.50
    ):
        lines.append(
            "Lecture: les recours observés apparaissent majoritairement quand la marge TW est déjà serrée"
        )

    if (
        late_capacity_share is not None
        and math.isfinite(float(late_capacity_share))
        and float(late_capacity_share) >= 0.45
    ):
        lines.append(
            "Lecture: la fin de tournée paraît fortement dominée par la capacité et les demandes"
        )
    elif (
        early_top
        and late_top
        and early_top != "none"
        and late_top != "none"
        and early_top != late_top
    ):
        lines.append(
            "Lecture: le régime de décision change entre le début et la fin de la tournée"
        )

    return lines


def _summarize_trace_trajectory(trace: dict, include_reading: bool = True) -> List[str]:
    actions = [int(a) for a in trace.get("actions", [])]
    if not actions:
        return ["Aucune trajectoire stockée pour cette instance"]

    locs = np.array(trace.get("locs", []), dtype=float)
    recourse_flags = [bool(v) for v in trace.get("recourse_triggered", [])]
    top_constraints = trace.get("top_constraints", []) or []
    decoder_state_constraints = trace.get("decoder_state_constraints", []) or []
    decoder_dynamic_states = _trace_decoder_dynamic_states(trace)
    instance_structural_constraints = _trace_instance_structural_constraints(trace)
    solution_features = trace.get("solution_features", {}) or {}

    depot_returns = sum(1 for action in actions if action == 0)
    customer_actions = sum(1 for action in actions if action > 0)
    first_depot = next((idx for idx, action in enumerate(actions) if action == 0), None)
    first_recourse = next((idx for idx, flag in enumerate(recourse_flags) if flag), None)

    hop_distances: List[float] = []
    customer_hops: List[float] = []
    current = 0
    for action in actions:
        if 0 <= current < len(locs) and 0 <= action < len(locs):
            hop = _dist(locs, current, action)
            if math.isfinite(hop):
                hop_distances.append(hop)
                if current > 0 and action > 0:
                    customer_hops.append(hop)
        current = action

    split_idx = max(1, len(actions) // 2)
    family_regime_payloads = decoder_state_constraints or top_constraints
    regime_payloads = (
        decoder_dynamic_states
        or decoder_state_constraints
        or top_constraints
    )
    regime_source = (
        "états dynamiques du décodeur"
        if (decoder_dynamic_states or decoder_state_constraints)
        else "attribution d'entrée"
    )
    early_share = _mean_constraint_share(regime_payloads[:split_idx])
    late_share = _mean_constraint_share(regime_payloads[split_idx:]) or dict(early_share)
    early_top = _constraint_label(_dominant_constraint_name(early_share))
    late_top = _constraint_label(_dominant_constraint_name(late_share))
    late_family_share = _mean_constraint_share(family_regime_payloads[split_idx:]) or {}
    late_capacity_share = float(late_family_share.get("capacity_demands", float("nan")))
    depot_share = depot_returns / max(len(actions), 1)
    mean_customer_hop = (
        float(np.mean(customer_hops)) if customer_hops else float("nan")
    )
    recourse_count = float(sum(1 for flag in recourse_flags if flag))
    selected_is_customer = [
        bool(v) for v in (solution_features.get("selected_is_customer", []) or [])
    ]
    tw_slack_norm_customer = _masked_finite_series(
        solution_features.get("selected_tw_slack_norm", []) or [],
        selected_is_customer,
    )
    wait_time_customer = _masked_finite_series(
        solution_features.get("selected_wait_time", []) or [],
        selected_is_customer,
    )
    linehaul_util = _finite_series(
        solution_features.get("used_capacity_linehaul_share", []) or []
    )
    backhaul_util = _finite_series(
        solution_features.get("used_capacity_backhaul_share", []) or []
    )
    late_start = max(1, len(actions) // 2)
    late_tw_slack_norm = _masked_finite_series(
        (solution_features.get("selected_tw_slack_norm", []) or [])[late_start:],
        selected_is_customer[late_start:],
    )
    recourse_tw_slack_norm: List[float] = []
    for tw_raw, is_customer, recourse in zip(
        solution_features.get("selected_tw_slack_norm", []) or [],
        selected_is_customer,
        recourse_flags,
    ):
        if not bool(is_customer) or not bool(recourse):
            continue
        try:
            tw_value = float(tw_raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(tw_value):
            recourse_tw_slack_norm.append(tw_value)
    tw_tight_share = _share_leq(tw_slack_norm_customer, 0.10)
    late_tw_tight_share = _share_leq(late_tw_slack_norm, 0.10)
    recourse_tw_tight_share = _share_leq(recourse_tw_slack_norm, 0.10)

    lines: List[str] = []
    if include_reading:
        lines.extend(
            _trajectory_reading_lines(
                depot_share=depot_share,
                customer_hop=mean_customer_hop,
                recourse_events=recourse_count,
                early_top=early_top,
                late_top=late_top,
                late_capacity_share=late_capacity_share,
                tw_tight_share=tw_tight_share,
                late_tw_tight_share=late_tw_tight_share,
                recourse_under_tw_tight_share=recourse_tw_tight_share,
            )
        )

    if instance_structural_constraints:
        lines.append(
            "Contraintes structurelles de l'instance: "
            f"{_format_structural_constraints(instance_structural_constraints)}"
        )

    lines.extend([
        (
            f"Trajectoire stockée: {customer_actions} actions client, {depot_returns} retours dépôt "
            f"({(depot_returns / max(len(actions), 1)) * 100.0:.1f}% des étapes)"
        )
    ])
    if first_depot is None:
        lines.append("Premier retour dépôt: aucun dans la trace stockée")
    else:
        lines.append(f"Premier retour dépôt: étape {first_depot}")
    if first_recourse is not None:
        lines.append(
            f"Premier recours: étape {first_recourse} ({sum(1 for flag in recourse_flags if flag)} événement(s))"
        )
    mean_hop = float(np.mean(hop_distances)) if hop_distances else float("nan")
    if math.isfinite(mean_hop):
        lines.append(f"Distance moyenne par action: {mean_hop:.3f}")
    if math.isfinite(mean_customer_hop):
        lines.append(f"Distance moyenne entre deux clients: {mean_customer_hop:.3f}")
    if tw_slack_norm_customer:
        lines.append(
            f"Marge TW normalisée moyenne (clients): {float(np.mean(tw_slack_norm_customer)):.3f}"
        )
    if math.isfinite(tw_tight_share):
        lines.append(
            f"Part d'étapes clients avec TW serrée (<=10%): {tw_tight_share * 100.0:.1f}%"
        )
    if math.isfinite(late_tw_tight_share):
        lines.append(
            f"Part tardive d'étapes clients avec TW serrée: {late_tw_tight_share * 100.0:.1f}%"
        )
    wait_share = _share_gt(wait_time_customer, 0.0)
    if math.isfinite(wait_share):
        lines.append(f"Part d'étapes clients avec attente: {wait_share * 100.0:.1f}%")
    if math.isfinite(recourse_tw_tight_share):
        lines.append(
            f"Parmi les étapes en recours, part sous TW serrée: {recourse_tw_tight_share * 100.0:.1f}%"
        )
    if linehaul_util:
        lines.append(
            f"Utilisation linehaul moyenne en cours de tournée: {float(np.mean(linehaul_util)) * 100.0:.1f}%"
        )
    if backhaul_util:
        lines.append(
            f"Utilisation backhaul moyenne en cours de tournée: {float(np.mean(backhaul_util)) * 100.0:.1f}%"
        )
    lines.append(f"Début de tournée dominé par ({regime_source}): {early_top}")
    lines.append(f"Fin de tournée dominée par ({regime_source}): {late_top}")
    return lines


def _format_report_trajectory(summary: dict) -> List[str]:
    trajectory = summary.get("trajectory", {}) or {}
    if not trajectory:
        return []
    items: List[str] = []
    num_traces = int(trajectory.get("num_traces", 0) or 0)
    try:
        depot_share = float(trajectory.get("mean_depot_return_share", float("nan")))
    except (TypeError, ValueError):
        depot_share = float("nan")
    try:
        hop_value = float(trajectory.get("mean_customer_hop_distance", float("nan")))
    except (TypeError, ValueError):
        hop_value = float("nan")
    try:
        recourse_value = float(
            trajectory.get("mean_recourse_events_per_instance", float("nan"))
        )
    except (TypeError, ValueError):
        recourse_value = float("nan")
    early_decoder_state_key = str(
        trajectory.get("early_top_decoder_state_constraint_state", "none")
    )
    late_decoder_state_key = str(
        trajectory.get("late_top_decoder_state_constraint_state", "none")
    )
    early_dynamic_key = str(trajectory.get("early_top_decoder_dynamic_state", "none"))
    late_dynamic_key = str(trajectory.get("late_top_decoder_dynamic_state", "none"))
    early_decoder_key = str(trajectory.get("early_top_decoder_state_constraint", "none"))
    late_decoder_key = str(trajectory.get("late_top_decoder_state_constraint", "none"))
    early_state_key = str(trajectory.get("early_top_constraint_state", "none"))
    late_state_key = str(trajectory.get("late_top_constraint_state", "none"))
    if early_dynamic_key != "none" or late_dynamic_key != "none":
        early_key = early_dynamic_key
        late_key = late_dynamic_key
        regime_source = "états dynamiques du décodeur"
        late_constraint_share = trajectory.get("late_decoder_dynamic_state_share", {}) or {}
    elif early_decoder_state_key != "none" or late_decoder_state_key != "none":
        early_key = early_decoder_state_key
        late_key = late_decoder_state_key
        regime_source = "états dynamiques du décodeur"
        late_constraint_share = (
            trajectory.get("late_decoder_state_constraint_state_share", {}) or {}
        )
    elif early_decoder_key != "none" or late_decoder_key != "none":
        early_key = early_decoder_key
        late_key = late_decoder_key
        regime_source = "familles dynamiques du décodeur"
        late_constraint_share = (
            trajectory.get("late_decoder_state_constraint_share", {}) or {}
        )
    elif early_state_key != "none" or late_state_key != "none":
        early_key = early_state_key
        late_key = late_state_key
        regime_source = "attribution d'entrée"
        late_constraint_share = trajectory.get("late_constraint_share", {}) or {}
    else:
        early_key = str(trajectory.get("early_top_constraint", "none"))
        late_key = str(trajectory.get("late_top_constraint", "none"))
        regime_source = "attribution d'entrée"
        late_constraint_share = trajectory.get("late_constraint_share", {}) or {}
    early = _constraint_label(early_key)
    late = _constraint_label(late_key)
    solution = trajectory.get("solution_features", {}) or {}
    characteristic_explanations = trajectory.get("characteristic_explanations", {}) or {}
    try:
        late_capacity_share = float(
            late_constraint_share.get("capacity_demands", float("nan"))
        )
    except (TypeError, ValueError):
        late_capacity_share = float("nan")
    try:
        tw_tight_share = float(solution.get("tw_tight_step_share", float("nan")))
    except (TypeError, ValueError):
        tw_tight_share = float("nan")
    try:
        late_tw_tight_share = float(solution.get("late_tw_tight_step_share", float("nan")))
    except (TypeError, ValueError):
        late_tw_tight_share = float("nan")
    try:
        recourse_tw_tight_share = float(
            solution.get("recourse_under_tw_tight_share", float("nan"))
        )
    except (TypeError, ValueError):
        recourse_tw_tight_share = float("nan")

    items.extend(
        _trajectory_reading_lines(
            depot_share=depot_share,
            customer_hop=hop_value,
            recourse_events=recourse_value,
            early_top=early_key,
            late_top=late_key,
            late_capacity_share=late_capacity_share,
            tw_tight_share=tw_tight_share,
            late_tw_tight_share=late_tw_tight_share,
            recourse_under_tw_tight_share=recourse_tw_tight_share,
        )
    )

    if num_traces > 0:
        items.append(
            f"Résumé global sur {num_traces} trace(s) stockée(s): "
            f"{float(trajectory.get('mean_depot_returns_per_instance', 0.0)):.2f} retours dépôt moyens par instance"
        )
    if math.isfinite(hop_value):
        items.append(f"Distance moyenne entre deux clients: {hop_value:.3f}")
    if early != "none" or late != "none":
        if early == late and early != "none":
            items.append(f"Régime dominant ({regime_source}): stable ({early})")
        else:
            items.append(f"Glissement dominant ({regime_source}): {early} -> {late}")
    if math.isfinite(recourse_value) and recourse_value > 0:
        items.append(f"Recours moyen par instance stockée: {recourse_value:.2f}")
    for label, key, scale in [
        ("Marge TW normalisée moyenne (clients)", "mean_selected_tw_slack_norm", 1.0),
        ("Part d'étapes clients avec TW serrée (<=10%)", "tw_tight_step_share", 100.0),
        ("Part tardive d'étapes clients avec TW serrée", "late_tw_tight_step_share", 100.0),
        ("Part d'étapes de recours sous TW serrée", "recourse_under_tw_tight_share", 100.0),
        ("Attente moyenne sur étapes clients", "mean_selected_wait_time", 1.0),
        ("Part d'étapes clients avec attente", "wait_step_share", 100.0),
        ("Utilisation linehaul moyenne par route", "route_linehaul_utilization_mean", 100.0),
        ("Utilisation backhaul moyenne par route", "route_backhaul_utilization_mean", 100.0),
        ("Longueur moyenne de route", "route_length_mean", 1.0),
        ("Profondeur moyenne de route", "route_depth_mean", 1.0),
        ("Largeur moyenne de route", "route_width_mean", 1.0),
    ]:
        raw = solution.get(key, float("nan"))
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value):
            continue
        if scale == 100.0:
            items.append(f"{label}: {value * scale:.1f}%")
        else:
            items.append(f"{label}: {value:.3f}")

    for key, label in [
        ("depot_return_share", "retours dépôt"),
        ("late_tw_tight_step_share", "TW serrées en fin de tournée"),
        ("recourse_under_tw_tight_share", "recours sous TW serrée"),
        ("wait_step_share", "attente sur étapes clients"),
    ]:
        payload = characteristic_explanations.get(key, None)
        if not isinstance(payload, dict):
            continue
        try:
            support_rate = float(payload.get("support_rate", float("nan")))
        except (TypeError, ValueError):
            support_rate = float("nan")
        support_txt = (
            f"{support_rate * 100.0:.1f}% des étapes éligibles"
            if math.isfinite(support_rate)
            else "taux indisponible"
        )
        constraint_items = payload.get("top_constraints", []) or []
        feature_items = payload.get("top_features", []) or []
        chunks: List[str] = []
        if constraint_items:
            chunks.append(f"contraintes: {_format_top_constraints(constraint_items)}")
        if feature_items:
            chunks.append(f"features: {_format_top_features(feature_items)}")
        if chunks:
            items.append(f"Pourquoi {label} ({support_txt}): " + " | ".join(chunks))
    return items


def _category_fallbacks() -> List[Tuple[str, str]]:
    return [
        (
            "Explication abductive",
            "Pas de signal saillant dans cette catégorie à cette étape",
        ),
        (
            "Explication contrastive",
            "Pas d'alternative contrastive exploitable à cette étape",
        ),
        (
            "Deletion faithfulness",
            "Pas de mesure step-level disponible à cette étape",
        ),
        (
            "Contrefactuels locaux",
            "Aucun contrefactuel local simple trouvé à cette étape",
        ),
    ]


def _step_payload_with_deletion(
    trace: Dict[str, Any],
    step: int,
    step_records: List[Any],
) -> Tuple[int, str, Dict[str, List[str]]]:
    payload = explain_step_structured(trace, step)
    step_idx = int(payload.get("step", step))
    summary = str(payload.get("summary", "hors plage"))
    categories = dict(payload.get("categories", {}) or {})

    deletion_items: List[str] = []
    if 0 <= step_idx < len(step_records) and isinstance(step_records[step_idx], dict):
        deletion_items = _format_step_deletion(step_records[step_idx])
    categories["Deletion faithfulness"] = deletion_items

    normalized: Dict[str, List[str]] = {}
    for category_name, _ in _category_fallbacks():
        normalized[category_name] = [
            str(item) for item in categories.get(category_name, []) if str(item)
        ]
    return step_idx, summary, normalized


def _render_bundle_step_first_lines(
    report: Dict[str, Any],
    report_path: Path,
    instance_index: int,
    steps_raw: str | None,
) -> List[str]:
    method_payloads: List[Dict[str, Any]] = []
    report_refs = report.get("reports", {}) or {}
    all_steps: set[int] = set()

    for method_key in ["gradient", "integrated_gradients"]:
        ref = report_refs.get(method_key)
        if not isinstance(ref, dict):
            continue
        candidate = Path(str(ref.get("path_resolved") or ref.get("path") or "").strip())
        if not candidate.exists():
            continue

        method_report = _load_json(candidate)
        instances = method_report.get("instances", [])
        if not isinstance(instances, list) or not instances:
            continue
        if not (0 <= instance_index < len(instances)):
            raise ValueError(
                f"Invalid instance index {instance_index} for method report {candidate}. "
                f"Available range: [0, {len(instances)-1}]"
            )

        trace = instances[instance_index]
        actions = trace.get("actions", [])
        step_records = method_report.get("steps", []) or []
        method_steps = _parse_step_list(steps_raw, len(actions))
        all_steps.update(method_steps)

        step_map: Dict[int, Dict[str, Any]] = {}
        for step in method_steps:
            step_idx, summary, categories = _step_payload_with_deletion(
                trace=trace, step=step, step_records=step_records
            )
            step_map[step_idx] = {"summary": summary, "categories": categories}

        cfg = method_report.get("config", {}) or {}
        variant_code = str(trace.get("instance_variant_code", "")).strip()
        variant_flags = trace.get("instance_variant_flags", {})
        active_constraints = trace.get("instance_active_constraints", [])
        structural_constraints = _trace_instance_structural_constraints(trace)

        method_payloads.append(
            {
                "label": _method_heading(method_report),
                "report_path": candidate,
                "mode": cfg.get("node_importance_mode", "decision-only"),
                "num_steps": len(actions),
                "variant_code": variant_code,
                "variant_flags": variant_flags,
                "active_constraints": active_constraints,
                "structural_constraints": structural_constraints,
                "trajectory": _summarize_trace_trajectory(trace, include_reading=True),
                "steps": step_map,
            }
        )

    if not method_payloads:
        raise ValueError(
            f"Bundle has no usable method reports with instances: {report_path}"
        )

    lines: List[str] = []
    lines.append("## Contexte")
    lines.append("")
    lines.append(f"- bundle: `{report_path}`")
    lines.append(f"- instance: `{instance_index}`")
    lines.append("")
    lines.extend(_explanation_category_lines())

    lines.append("## Méthodes")
    lines.append("")
    for payload in method_payloads:
        lines.append(f"- {payload['label']}")
        lines.append(f"  - report: `{payload['report_path']}`")
        lines.append(f"  - nombre_etapes: `{payload['num_steps']}`")
        lines.append(f"  - mode_importance_noeud: `{payload['mode']}`")
        variant_code = str(payload.get("variant_code", "")).strip()
        if variant_code:
            lines.append(f"  - variante_instance: `{variant_code}`")
        variant_flags = payload.get("variant_flags", {})
        if isinstance(variant_flags, dict) and variant_flags:
            lines.append(f"  - drapeaux_variante: `{variant_flags}`")
        active_constraints = payload.get("active_constraints", [])
        if isinstance(active_constraints, list):
            lines.append(
                "  - contraintes_instance: "
                f"`{_format_variant_constraints(active_constraints)}`"
            )
        structural_constraints = payload.get("structural_constraints", [])
        if structural_constraints:
            lines.append(
                "  - contraintes_structurelles: "
                f"`{_format_structural_constraints(structural_constraints)}`"
            )
    lines.append("")

    lines.append("## Trajectoire de l'instance (par méthode)")
    lines.append("")
    for payload in method_payloads:
        lines.append(f"- {payload['label']}")
        trajectory_items = [str(item) for item in payload.get("trajectory", []) if str(item)]
        if not trajectory_items:
            lines.append("  - Aucune trajectoire stockée pour cette méthode")
        else:
            for item in trajectory_items:
                lines.append(f"  - {item}")
    lines.append("")

    lines.append("## Etapes")
    lines.append("")
    for step in sorted(all_steps):
        lines.append(f"### Etape {step}")
        lines.append("")
        lines.append("- Décision")
        for payload in method_payloads:
            step_data = payload["steps"].get(step)
            if step_data is None:
                lines.append(f"  - {payload['label']}: hors plage")
            else:
                lines.append(f"  - {payload['label']}: {step_data['summary']}")

        for category_name, fallback in _category_fallbacks():
            lines.append(f"- {category_name}")
            for payload in method_payloads:
                lines.append(f"  - {payload['label']}")
                step_data = payload["steps"].get(step)
                if step_data is None:
                    entries = ["Étape indisponible pour cette méthode"]
                else:
                    entries = [
                        str(item)
                        for item in step_data.get("categories", {}).get(category_name, [])
                        if str(item)
                    ]
                    if not entries:
                        entries = [fallback]
                for item in entries:
                    lines.append(f"    - {item}")
        lines.append("")
    return lines


def explain_step_structured(trace: dict, step: int) -> dict:
    locs = np.array(trace["locs"], dtype=float)
    actions = [int(a) for a in trace.get("actions", [])]
    top_nodes_all = trace.get("top_nodes", [])
    top_scores_all = trace.get("top_scores", [])
    top_nodes_feas_all = trace.get("top_nodes_feasibility", [])
    top_scores_feas_all = trace.get("top_scores_feasibility", [])
    top_features_all = trace.get("top_features", [])
    top_constraints_all = trace.get("top_constraints", [])
    decoder_state_constraints_all = trace.get("decoder_state_constraints", [])
    decoder_dynamic_states_all = _trace_decoder_dynamic_states(trace)
    contrastive_policy_alt_action_all = trace.get("contrastive_policy_alt_action", [])
    contrastive_policy_alt_feasible_all = trace.get(
        "contrastive_policy_alt_feasible", []
    )
    contrastive_policy_alt_recourse_all = trace.get(
        "contrastive_policy_alt_recourse", []
    )
    contrastive_policy_logit_gap_all = trace.get("contrastive_policy_logit_gap", [])
    contrastive_policy_logprob_gap_all = trace.get(
        "contrastive_policy_logprob_gap", []
    )
    contrastive_feasible_alt_action_all = trace.get(
        "contrastive_feasible_alt_action", []
    )
    contrastive_feasible_alt_feasible_all = trace.get(
        "contrastive_feasible_alt_feasible", []
    )
    contrastive_feasible_alt_recourse_all = trace.get(
        "contrastive_feasible_alt_recourse", []
    )
    contrastive_feasible_logit_gap_all = trace.get(
        "contrastive_feasible_logit_gap", []
    )
    contrastive_feasible_logprob_gap_all = trace.get(
        "contrastive_feasible_logprob_gap", []
    )
    contrastive_alt_action_all = trace.get("contrastive_alt_action", [])
    contrastive_alt_source_all = trace.get("contrastive_alt_source", [])
    contrastive_alt_feasible_all = trace.get("contrastive_alt_feasible", [])
    contrastive_alt_recourse_all = trace.get("contrastive_alt_recourse", [])
    contrastive_logit_gap_all = trace.get("contrastive_logit_gap", [])
    contrastive_logprob_gap_all = trace.get("contrastive_logprob_gap", [])
    contrastive_top_constraints_all = trace.get("contrastive_top_constraints", [])
    chosen_feasible_all = trace.get("chosen_feasible", [])
    recourse_flags_all = trace.get("recourse_triggered", [])
    recourse_cost_all = trace.get("recourse_cost_est", [])
    counterfactuals_all = trace.get("counterfactuals", [])

    if not actions or step >= len(actions):
        return {
            "step": step,
            "variant_code": "",
            "summary": "hors plage",
            "categories": {},
        }

    variant_code = str(trace.get("instance_variant_code", "")).strip()

    chosen = int(actions[step])
    current = _node_before_step(actions, step)
    num_nodes = int(locs.shape[0])
    top_nodes = [int(n) for n in top_nodes_all[step]] if step < len(top_nodes_all) else []
    top_scores = (
        [float(s) for s in top_scores_all[step]] if step < len(top_scores_all) else []
    )
    top_nodes_feas = (
        [int(n) for n in top_nodes_feas_all[step]]
        if step < len(top_nodes_feas_all)
        else []
    )
    top_scores_feas = (
        [float(s) for s in top_scores_feas_all[step]]
        if step < len(top_scores_feas_all)
        else []
    )
    top_features = top_features_all[step] if step < len(top_features_all) else []
    top_constraints = top_constraints_all[step] if step < len(top_constraints_all) else []
    decoder_state_constraints = (
        decoder_state_constraints_all[step]
        if step < len(decoder_state_constraints_all)
        else []
    )
    decoder_dynamic_states = (
        decoder_dynamic_states_all[step]
        if step < len(decoder_dynamic_states_all)
        else []
    )
    contrastive_policy_alt_action = (
        int(contrastive_policy_alt_action_all[step])
        if step < len(contrastive_policy_alt_action_all)
        else -1
    )
    contrastive_policy_alt_feasible = (
        bool(contrastive_policy_alt_feasible_all[step])
        if step < len(contrastive_policy_alt_feasible_all)
        else False
    )
    contrastive_policy_alt_recourse = (
        bool(contrastive_policy_alt_recourse_all[step])
        if step < len(contrastive_policy_alt_recourse_all)
        else False
    )
    contrastive_policy_logit_gap = (
        float(contrastive_policy_logit_gap_all[step])
        if step < len(contrastive_policy_logit_gap_all)
        else float("nan")
    )
    contrastive_policy_logprob_gap = (
        float(contrastive_policy_logprob_gap_all[step])
        if step < len(contrastive_policy_logprob_gap_all)
        else float("nan")
    )
    contrastive_feasible_alt_action = (
        int(contrastive_feasible_alt_action_all[step])
        if step < len(contrastive_feasible_alt_action_all)
        else -1
    )
    contrastive_feasible_alt_feasible = (
        bool(contrastive_feasible_alt_feasible_all[step])
        if step < len(contrastive_feasible_alt_feasible_all)
        else False
    )
    contrastive_feasible_alt_recourse = (
        bool(contrastive_feasible_alt_recourse_all[step])
        if step < len(contrastive_feasible_alt_recourse_all)
        else False
    )
    contrastive_feasible_logit_gap = (
        float(contrastive_feasible_logit_gap_all[step])
        if step < len(contrastive_feasible_logit_gap_all)
        else float("nan")
    )
    contrastive_feasible_logprob_gap = (
        float(contrastive_feasible_logprob_gap_all[step])
        if step < len(contrastive_feasible_logprob_gap_all)
        else float("nan")
    )
    contrastive_alt_action = (
        int(contrastive_alt_action_all[step])
        if step < len(contrastive_alt_action_all)
        else -1
    )
    contrastive_alt_source = (
        str(contrastive_alt_source_all[step])
        if step < len(contrastive_alt_source_all)
        else "none"
    )
    contrastive_alt_feasible = (
        bool(contrastive_alt_feasible_all[step])
        if step < len(contrastive_alt_feasible_all)
        else False
    )
    contrastive_alt_recourse = (
        bool(contrastive_alt_recourse_all[step])
        if step < len(contrastive_alt_recourse_all)
        else False
    )
    contrastive_logit_gap = (
        float(contrastive_logit_gap_all[step])
        if step < len(contrastive_logit_gap_all)
        else float("nan")
    )
    contrastive_logprob_gap = (
        float(contrastive_logprob_gap_all[step])
        if step < len(contrastive_logprob_gap_all)
        else float("nan")
    )
    contrastive_top_constraints = (
        contrastive_top_constraints_all[step]
        if step < len(contrastive_top_constraints_all)
        else []
    )
    chosen_feasible = (
        bool(chosen_feasible_all[step]) if step < len(chosen_feasible_all) else True
    )
    recourse_triggered = (
        bool(recourse_flags_all[step]) if step < len(recourse_flags_all) else False
    )
    recourse_cost_est = (
        float(recourse_cost_all[step]) if step < len(recourse_cost_all) else 0.0
    )
    counterfactual = (
        counterfactuals_all[step] if step < len(counterfactuals_all) else None
    )
    instance_structural_constraints = _trace_instance_structural_constraints(trace)
    rank = _rank_in_top(chosen, top_nodes)
    input_constraint_view = top_constraints
    decoder_constraint_view = (
        decoder_dynamic_states or decoder_state_constraints
    )

    abductive_items: List[str] = []
    contrastive_items: List[str] = []
    counterfactual_items: List[str] = []
    abductive_items.extend(_abductive_reading_lines(input_constraint_view, top_features))
    abductive_items.extend(
        _constraint_payload_reading_lines(
            decoder_constraint_view,
            "états dynamiques du décodeur",
        )
    )
    if instance_structural_constraints:
        abductive_items.append(
            "Contexte structurel de l'instance: "
            f"{_format_structural_constraints(instance_structural_constraints)}"
        )
    if decoder_constraint_view:
        abductive_items.append(
            "États dynamiques dominants du décodeur: "
            f"{_format_top_constraints(decoder_constraint_view)}"
        )
    if input_constraint_view:
        abductive_items.append(
            f"Contraintes d'entrée dominantes: {_format_top_constraints(input_constraint_view)}"
        )
    if decoder_constraint_view and input_constraint_view:
        decoder_state_top = str(
            decoder_constraint_view[0].get(
                "constraint_family",
                decoder_constraint_view[0].get("constraint", ""),
            )
        ).strip()
        input_top = str(
            input_constraint_view[0].get(
                "constraint_family",
                input_constraint_view[0].get("constraint", ""),
            )
        ).strip()
        if decoder_state_top and input_top and decoder_state_top != input_top:
            abductive_items.append(
                "Écart entre signaux: l'attribution d'entrée et l'état courant du décodeur ne pointent pas la même contrainte dominante"
            )
    if contrastive_alt_action >= 0 and contrastive_alt_source != "none":
        contrastive_items.extend(
            _contrastive_reading_lines(
                alt_action=contrastive_alt_action,
                alt_source=contrastive_alt_source,
                alt_feasible=contrastive_alt_feasible,
                alt_recourse=contrastive_alt_recourse,
                logit_gap=contrastive_logit_gap,
            )
        )
        primary_constraints_suffix = None
        if contrastive_top_constraints:
            primary_constraints_suffix = (
                "contraintes qui départagent (comparaison principale): "
                f"{_format_top_constraints(contrastive_top_constraints)}"
            )
        if contrastive_policy_alt_action >= 0:
            contrastive_items.extend(
                _format_contrastive_alternative(
                    title="Alternative de politique (2e meilleur score)",
                    alt_action=contrastive_policy_alt_action,
                    alt_feasible=contrastive_policy_alt_feasible,
                    alt_recourse=contrastive_policy_alt_recourse,
                    logit_gap=contrastive_policy_logit_gap,
                    logprob_gap=contrastive_policy_logprob_gap,
                    constraints_suffix=(
                        primary_constraints_suffix
                        if (
                            contrastive_alt_source == "policy_masked"
                            or contrastive_feasible_alt_action == contrastive_policy_alt_action
                        )
                        else None
                    ),
                )
            )
        if (
            contrastive_feasible_alt_action >= 0
            and contrastive_feasible_alt_action != contrastive_policy_alt_action
        ):
            contrastive_items.extend(
                _format_contrastive_alternative(
                    title="Alternative strictement faisable (sans recours)",
                    alt_action=contrastive_feasible_alt_action,
                    alt_feasible=contrastive_feasible_alt_feasible,
                    alt_recourse=contrastive_feasible_alt_recourse,
                    logit_gap=contrastive_feasible_logit_gap,
                    logprob_gap=contrastive_feasible_logprob_gap,
                    constraints_suffix=(
                        primary_constraints_suffix
                        if contrastive_alt_source == "full_feasible"
                        else None
                    ),
                )
            )
        if (
            contrastive_policy_alt_action < 0
            and contrastive_feasible_alt_action < 0
        ):
            contrastive_items.extend(
                _format_contrastive_alternative(
                    title=f"Comparaison principale ({_contrastive_source_label(contrastive_alt_source)})",
                    alt_action=contrastive_alt_action,
                    alt_feasible=contrastive_alt_feasible,
                    alt_recourse=contrastive_alt_recourse,
                    logit_gap=contrastive_logit_gap,
                    logprob_gap=contrastive_logprob_gap,
                    constraints_suffix=primary_constraints_suffix,
                )
            )
    if top_features:
        abductive_items.append(f"Features dominantes: {_format_top_features(top_features)}")
    feas_rank = _rank_in_top(chosen, top_nodes_feas)
    if feas_rank is not None and chosen > 0:
        feas_score = (
            top_scores_feas[feas_rank - 1]
            if feas_rank - 1 < len(top_scores_feas)
            else None
        )
        if feas_score is None:
            abductive_items.append(
                f"Noeud choisi aussi central dans l'analyse faisabilité/recours (rang {feas_rank})"
            )
        elif abs(float(feas_score)) > 1e-12:
            abductive_items.append(
                f"Noeud choisi aussi central dans l'analyse faisabilité/recours (rang {feas_rank}, score {feas_score:.3f})"
            )
    if not chosen_feasible and chosen > 0:
        abductive_items.append("Action choisie non faisable avant recours")
    if recourse_triggered:
        abductive_items.append(
            f"Action client infaisable sous contraintes complètes => recours (coût estimé {recourse_cost_est:.3f})"
        )
    if isinstance(counterfactual, dict) and counterfactual:
        counterfactual_items.extend(_counterfactual_reading_lines(counterfactual))
        counterfactual_items.append(_format_counterfactual(counterfactual))

    if chosen == 0:
        abductive_items.append("Retour depot")
        if top_nodes:
            max_attr_node = int(top_nodes[0])
            d_curr_depot = _dist(locs, current, 0)
            d_curr_attr = _dist(locs, current, max_attr_node)
            if d_curr_depot < d_curr_attr:
                abductive_items.append("Le depot est plus proche que le noeud le plus attribué")
        return {
            "step": step,
            "variant_code": variant_code,
            "summary": f"noeud 0 choisi depuis le noeud {current}",
            "categories": {
                "Explication abductive": abductive_items,
                "Explication contrastive": contrastive_items,
                "Contrefactuels locaux": counterfactual_items,
            },
        }

    if rank is not None:
        score = top_scores[rank - 1] if rank - 1 < len(top_scores) else None
        if score is None:
            abductive_items.append(f"Noeud choisi présent dans top attribution (rang {rank})")
        else:
            abductive_items.append(
                f"Noeud choisi présent dans top attribution (rang {rank}, score {score:.3f})"
            )

    visited = _visited_customers(actions, step)
    candidates = [n for n in range(1, num_nodes) if n not in visited]
    if chosen not in candidates:
        candidates.append(chosen)

    d_curr_chosen = _dist(locs, current, chosen)
    if candidates:
        d_curr_all = np.array([_dist(locs, current, n) for n in candidates], dtype=float)
        pct = _percentile_position(d_curr_chosen, d_curr_all)
        if pct <= 0.3:
            abductive_items.append(
                f"Noeud choisi proche du noeud courant (distance {d_curr_chosen:.3f}, quantile {pct:.2f})"
            )
        elif pct >= 0.7:
            abductive_items.append(
                f"Noeud choisi plutot éloigné du noeud courant (distance {d_curr_chosen:.3f}, quantile {pct:.2f})"
            )
        else:
            abductive_items.append(
                f"Distance intermédiaire au noeud courant (distance {d_curr_chosen:.3f}, quantile {pct:.2f})"
            )

    if top_nodes:
        top_neighbors = [n for n in top_nodes if n != chosen]
        if top_neighbors:
            d_top = np.mean([_dist(locs, chosen, n) for n in top_neighbors])
            others = [
                n for n in range(1, num_nodes) if n not in top_neighbors and n != chosen
            ]
            if others:
                d_other = np.mean([_dist(locs, chosen, n) for n in others])
                if d_top < d_other * 0.9:
                    abductive_items.append(
                        f"Noeud choisi proche du cluster attribué (dist top {d_top:.3f} vs autres {d_other:.3f})"
                    )
                elif d_top > d_other * 1.1:
                    abductive_items.append(
                        f"Noeud choisi éloigné du cluster attribué (dist top {d_top:.3f} vs autres {d_other:.3f})"
                    )
                else:
                    abductive_items.append(
                        f"Noeud choisi à distance comparable du cluster attribué (dist top {d_top:.3f} vs autres {d_other:.3f})"
                    )

    if not abductive_items:
        abductive_items.append(
            "Sélection probablement guidée par des contraintes internes non exposées ici"
        )

    return {
        "step": step,
        "variant_code": variant_code,
        "summary": f"noeud {chosen} choisi depuis le noeud {current}",
        "categories": {
            "Explication abductive": abductive_items,
            "Explication contrastive": contrastive_items,
            "Contrefactuels locaux": counterfactual_items,
        },
    }


def explain_step(trace: dict, step: int) -> str:
    payload = explain_step_structured(trace, step)
    summary = str(payload.get("summary", "hors plage"))
    categories = payload.get("categories", {}) or {}
    items: List[str] = []
    for cat_items in categories.values():
        items.extend(str(item) for item in (cat_items or []))
    variant_code = str(payload.get("variant_code", "")).strip()
    variant_prefix = f"(variante {variant_code}) " if variant_code else ""
    if summary == "hors plage":
        return f"Etape {step}: hors plage."
    return f"Etape {step}: {variant_prefix}{summary}. " + _format_reasons(items)


def _explanation_category_lines() -> List[str]:
    return [
        "## Catégories d'explication",
        "",
        "- Explication abductive: pourquoi ce nœud est choisi à cette étape (top nodes, top features, contraintes d'entrée, contexte structurel et états dynamiques du décodeur).",
        "- Explication contrastive: pourquoi ce nœud est choisi plutôt que la meilleure alternative disponible.",
        "- Deletion faithfulness: si on perturbe les clients les plus attribués, est-ce que la décision change réellement ?",
        "- Contrefactuels locaux: quel petit changement concret pourrait faire basculer la décision ou rendre l'alternative faisable ?",
        "- Trajectoire globale: quel style global de tournée se dégage de la trace stockée ?",
        "",
    ]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _method_heading(report: Dict[str, Any]) -> str:
    cfg = report.get("config", {}) or {}
    method = str(cfg.get("attribution_method", "")).strip()
    if method == "integrated_gradients":
        baseline = str(cfg.get("ig_baseline", "")).strip()
        if baseline:
            return f"Integrated Gradients ({baseline})"
        return "Integrated Gradients"
    return "Gradient local"


def _first_bundle_instance_count(report: Dict[str, Any], bundle_path: Path) -> int:
    if str(report.get("kind", "")).strip() != "xai_dual_bundle":
        instances = report.get("instances", [])
        return len(instances) if isinstance(instances, list) else 0

    report_refs = report.get("reports", {}) or {}
    for method_key in ["gradient", "integrated_gradients"]:
        ref = report_refs.get(method_key)
        if not isinstance(ref, dict):
            continue
        candidate = Path(
            str(ref.get("path_resolved") or ref.get("path") or "").strip()
        )
        if not candidate.exists():
            continue
        method_report = _load_json(candidate)
        instances = method_report.get("instances", [])
        if isinstance(instances, list) and instances:
            return len(instances)
    raise ValueError(
        f"Bundle has no usable method reports with instances: {bundle_path}"
    )


def _render_document_for_instance(
    report: Dict[str, Any],
    report_path: Path,
    instance_index: int,
    steps_raw: str | None,
) -> str:
    lines: List[str] = ["# Explications Textuelles XAI", ""]

    if str(report.get("kind", "")).strip() == "xai_dual_bundle":
        lines.extend(
            _render_bundle_step_first_lines(
                report=report,
                report_path=report_path,
                instance_index=instance_index,
                steps_raw=steps_raw,
            )
        )
    else:
        section_lines, _ = _render_single_report_lines(
            report=report,
            report_path=report_path,
            instance_index=instance_index,
            steps_raw=steps_raw,
            heading_prefix="##",
        )
        lines.extend(_explanation_category_lines())
        lines.extend(section_lines)

    return "\n".join(lines)


def _render_single_report_lines(
    report: Dict[str, Any],
    report_path: Path,
    instance_index: int,
    steps_raw: str | None,
    heading_prefix: str = "##",
    include_report_trajectory: bool = False,
) -> Tuple[List[str], int]:
    instances = report.get("instances", [])
    if not instances:
        raise ValueError(
            "Report has no 'instances'. Re-run action_explainer with --save-instance-traces"
        )
    if not (0 <= instance_index < len(instances)):
        raise ValueError(
            f"Invalid instance index {instance_index}. Available range: [0, {len(instances)-1}]"
        )

    trace = instances[instance_index]
    actions = trace.get("actions", [])
    step_records = report.get("steps", []) or []
    steps = _parse_step_list(steps_raw, len(actions))
    variant_code = str(trace.get("instance_variant_code", "")).strip()
    variant_flags = trace.get("instance_variant_flags", {})
    active_constraints = trace.get("instance_active_constraints", [])
    structural_constraints = _trace_instance_structural_constraints(trace)

    lines: List[str] = []
    lines.append(f"{heading_prefix} Contexte")
    lines.append("")
    lines.append(f"- report: `{report_path}`")
    lines.append(f"- instance: `{instance_index}`")
    lines.append(f"- nombre_etapes: `{len(actions)}`")
    mode = report.get("config", {}).get("node_importance_mode", "decision-only")
    lines.append(f"- mode_importance_noeud: `{mode}`")
    if variant_code:
        lines.append(f"- variante_instance: `{variant_code}`")
    if isinstance(variant_flags, dict) and variant_flags:
        lines.append(f"- drapeaux_variante: `{variant_flags}`")
    if isinstance(active_constraints, list):
        lines.append(
            f"- contraintes_instance: `{_format_variant_constraints(active_constraints)}`"
        )
    if structural_constraints:
        lines.append(
            "- contraintes_structurelles: "
            f"`{_format_structural_constraints(structural_constraints)}`"
        )
    lines.append("")

    report_trajectory_items = _format_report_trajectory(report.get("summary", {}) or {})
    if include_report_trajectory and report_trajectory_items:
        lines.append(f"{heading_prefix} Repères globaux du rapport")
        lines.append("")
        for item in report_trajectory_items:
            lines.append(f"- {item}")
        lines.append("")

    instance_trajectory_items = _summarize_trace_trajectory(
        trace, include_reading=True
    )
    if instance_trajectory_items:
        lines.append(f"{heading_prefix} Trajectoire de l'instance")
        lines.append("")
        for item in instance_trajectory_items:
            lines.append(f"- {item}")
        lines.append("")

    lines.append(f"{heading_prefix} Etapes")
    lines.append("")
    for step in steps:
        step_idx, summary, categories = _step_payload_with_deletion(
            trace=trace,
            step=step,
            step_records=step_records,
        )
        lines.append(f"- Etape {step_idx}")
        lines.append(f"  - Décision: {summary}")
        for category_name, fallback in _category_fallbacks():
            lines.append(f"  - {category_name}")
            entries = categories.get(category_name, [])
            if not entries:
                entries = [fallback]
            for item in entries:
                lines.append(f"    - {item}")
        lines.append("")
    return lines, len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate textual XAI explanations for selected instances/steps"
    )
    parser.add_argument("--report", required=True, help="Path to action_explainer JSON")
    parser.add_argument(
        "--instance",
        default="0",
        help='Instance index in report, or "all" to render all stored instances',
    )
    parser.add_argument(
        "--instances",
        dest="instances_legacy",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--steps",
        default=None,
        help='Comma-separated steps (e.g. "0,5,12") or "all". Default: shortlist.',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional markdown output path. If omitted, prints only to stdout.",
    )
    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    report = _load_json(report_path)
    instance_raw = args.instances_legacy if args.instances_legacy is not None else args.instance
    max_instances = _first_bundle_instance_count(report, report_path)
    instance_ids, all_instances_mode = _parse_instance_selection(instance_raw, max_instances)

    rendered: List[Tuple[int, str]] = []
    for instance_index in instance_ids:
        rendered.append(
            (
                instance_index,
                _render_document_for_instance(
                    report=report,
                    report_path=report_path,
                    instance_index=instance_index,
                    steps_raw=args.steps,
                ),
            )
        )

    if all_instances_mode and args.output:
        out_root = Path(args.output)
        if out_root.suffix:
            out_root = out_root.parent / out_root.stem
        out_root.mkdir(parents=True, exist_ok=True)
        written_paths: List[Path] = []
        for instance_index, text in rendered:
            out_path = out_root / f"inst{instance_index:03d}" / "explanations.md"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
            written_paths.append(out_path)
        print(f"Generated {len(written_paths)} explanation file(s)")
        for path in written_paths:
            print(path)
        return

    if all_instances_mode:
        text = "\n\n---\n\n".join(text for _, text in rendered)
    else:
        text = rendered[0][1] if rendered else ""
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
