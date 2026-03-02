import argparse
import json
import math

from pathlib import Path
from typing import List

import numpy as np

FEATURE_LABELS = {
    "locs": "localisation",
    "demand_linehaul": "demande linehaul",
    "demand_backhaul": "demande backhaul",
    "time_windows": "fenêtres de temps",
    "service_time": "temps de service",
    "vehicle_capacity": "capacité véhicule",
    "open_route": "route ouverte",
    "mixed_backhaul": "backhaul mixte",
    "distance_limit": "limite distance",
    "depot_tw_end": "horizon dépôt",
    "has_backhaul": "présence backhaul",
    "current_time": "temps courant",
    "current_route_length": "longueur route courante",
    "used_capacity_linehaul": "charge linehaul utilisée",
    "used_capacity_backhaul": "charge backhaul utilisée",
    "speed": "vitesse",
}

CONSTRAINT_LABELS = {
    "space_distance": "géométrie et distance",
    "time_windows_service": "fenêtres de temps et service",
    "capacity_demands": "capacités et demandes",
    "route_structure": "structure de route",
    "route_recourse": "structure de route",
    "other": "autres signaux",
}

COUNTERFACTUAL_FEATURE_LABELS = {
    "time_window_end": "borne haute TW",
    "service_time": "temps de service",
    "demand_linehaul": "demande linehaul",
    "demand_backhaul": "demande backhaul",
    "vehicle_capacity": "capacité véhicule",
    "distance_limit": "limite distance",
    "depot_tw_end": "horizon dépôt",
}


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
        name = _constraint_label(str(item.get("constraint", "")))
        share = float(item.get("share", 0.0)) * 100.0
        if idx <= 3:
            parts.append(f"{name} ({share:.1f}%)")
        else:
            parts.append(f"{name} ({share:.1f}%, hors top 3)")
    return ", ".join(parts)


def _contrastive_source_label(source: str) -> str:
    if source == "full_feasible":
        return "alternative faisable"
    if source == "policy_masked":
        return "alternative masque politique"
    return "aucune alternative"


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


def _summarize_trace_trajectory(trace: dict) -> List[str]:
    actions = [int(a) for a in trace.get("actions", [])]
    if not actions:
        return ["Aucune trajectoire stockée pour cette instance"]

    locs = np.array(trace.get("locs", []), dtype=float)
    recourse_flags = [bool(v) for v in trace.get("recourse_triggered", [])]
    top_constraints = trace.get("top_constraints", []) or []

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
    early_share = _mean_constraint_share(top_constraints[:split_idx])
    late_share = _mean_constraint_share(top_constraints[split_idx:]) or dict(early_share)
    early_top = _constraint_label(_dominant_constraint_name(early_share))
    late_top = _constraint_label(_dominant_constraint_name(late_share))

    lines = [
        (
            f"Trajectoire stockée: {customer_actions} actions client, {depot_returns} retours dépôt "
            f"({(depot_returns / max(len(actions), 1)) * 100.0:.1f}% des étapes)"
        )
    ]
    if first_depot is None:
        lines.append("Premier retour dépôt: aucun dans la trace stockée")
    else:
        lines.append(f"Premier retour dépôt: étape {first_depot}")
    if first_recourse is not None:
        lines.append(
            f"Premier recours: étape {first_recourse} ({sum(1 for flag in recourse_flags if flag)} événement(s))"
        )
    mean_hop = float(np.mean(hop_distances)) if hop_distances else float("nan")
    mean_customer_hop = (
        float(np.mean(customer_hops)) if customer_hops else float("nan")
    )
    if math.isfinite(mean_hop):
        lines.append(f"Distance moyenne par action: {mean_hop:.3f}")
    if math.isfinite(mean_customer_hop):
        lines.append(f"Distance moyenne entre deux clients: {mean_customer_hop:.3f}")
    lines.append(f"Début de tournée dominé par: {early_top}")
    lines.append(f"Fin de tournée dominée par: {late_top}")
    return lines


def _format_report_trajectory(summary: dict) -> List[str]:
    trajectory = summary.get("trajectory", {}) or {}
    if not trajectory:
        return []
    items = []
    num_traces = int(trajectory.get("num_traces", 0) or 0)
    if num_traces > 0:
        items.append(
            f"Résumé global sur {num_traces} trace(s) stockée(s): "
            f"{float(trajectory.get('mean_depot_returns_per_instance', 0.0)):.2f} retours dépôt moyens par instance"
        )
    hop = trajectory.get("mean_customer_hop_distance", None)
    if hop is not None:
        try:
            hop_value = float(hop)
        except (TypeError, ValueError):
            hop_value = float("nan")
        if math.isfinite(hop_value):
            items.append(f"Distance moyenne entre deux clients: {hop_value:.3f}")
    early = _constraint_label(str(trajectory.get("early_top_constraint", "none")))
    late = _constraint_label(str(trajectory.get("late_top_constraint", "none")))
    if early != "none" or late != "none":
        items.append(f"Glissement dominant: {early} -> {late}")
    recourse = trajectory.get("mean_recourse_events_per_instance", None)
    if recourse is not None:
        try:
            recourse_value = float(recourse)
        except (TypeError, ValueError):
            recourse_value = float("nan")
        if math.isfinite(recourse_value) and recourse_value > 0:
            items.append(f"Recours moyen par instance stockée: {recourse_value:.2f}")
    return items


def explain_step_structured(trace: dict, step: int) -> dict:
    locs = np.array(trace["locs"], dtype=float)
    actions = [int(a) for a in trace.get("actions", [])]
    top_nodes_all = trace.get("top_nodes", [])
    top_scores_all = trace.get("top_scores", [])
    top_nodes_feas_all = trace.get("top_nodes_feasibility", [])
    top_scores_feas_all = trace.get("top_scores_feasibility", [])
    top_features_all = trace.get("top_features", [])
    top_constraints_all = trace.get("top_constraints", [])
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
            "items": [],
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
    rank = _rank_in_top(chosen, top_nodes)

    items: List[str] = []
    if top_constraints:
        items.append(
            f"Contraintes dominantes: {_format_top_constraints(top_constraints)}"
        )
    if contrastive_alt_action >= 0 and contrastive_alt_source != "none":
        alt_label = (
            "depot" if contrastive_alt_action == 0 else f"noeud {contrastive_alt_action}"
        )
        contrastive_chunks = [
            f"vs {alt_label} ({_contrastive_source_label(contrastive_alt_source)})"
        ]
        if np.isfinite(contrastive_logit_gap):
            contrastive_chunks.append(f"marge logit {contrastive_logit_gap:+.3f}")
        if np.isfinite(contrastive_logprob_gap):
            contrastive_chunks.append(f"marge logprob {contrastive_logprob_gap:+.3f}")
        if contrastive_top_constraints:
            contrastive_chunks.append(
                f"contraintes qui départagent: {_format_top_constraints(contrastive_top_constraints)}"
            )
        if not contrastive_alt_feasible and contrastive_alt_action > 0:
            contrastive_chunks.append(
                "alternative non faisable sous contraintes complètes"
            )
        if contrastive_alt_recourse:
            contrastive_chunks.append("alternative impliquerait recours")
        items.append("Comparaison contrastive: " + ", ".join(contrastive_chunks))
    if top_features:
        items.append(f"Features dominantes: {_format_top_features(top_features)}")
    feas_rank = _rank_in_top(chosen, top_nodes_feas)
    if feas_rank is not None and chosen > 0:
        feas_score = (
            top_scores_feas[feas_rank - 1]
            if feas_rank - 1 < len(top_scores_feas)
            else None
        )
        if feas_score is None:
            items.append(
                f"Noeud choisi aussi important pour la faisabilité (rang {feas_rank})"
            )
        elif abs(float(feas_score)) > 1e-12:
            items.append(
                f"Noeud choisi aussi important pour la faisabilité (rang {feas_rank}, score {feas_score:.3f})"
            )
    if not chosen_feasible and chosen > 0:
        items.append("Action choisie non faisable avant recours")
    if recourse_triggered:
        items.append(
            f"Action client infaisable sous contraintes complètes => recours (coût estimé {recourse_cost_est:.3f})"
        )
    if isinstance(counterfactual, dict) and counterfactual:
        items.append(_format_counterfactual(counterfactual))

    if chosen == 0:
        items.append("Retour depot")
        if top_nodes:
            max_attr_node = int(top_nodes[0])
            d_curr_depot = _dist(locs, current, 0)
            d_curr_attr = _dist(locs, current, max_attr_node)
            if d_curr_depot < d_curr_attr:
                items.append("Le depot est plus proche que le noeud le plus attribué")
        return {
            "step": step,
            "variant_code": variant_code,
            "summary": f"noeud 0 choisi depuis le noeud {current}",
            "items": items,
        }

    if rank is not None:
        score = top_scores[rank - 1] if rank - 1 < len(top_scores) else None
        if score is None:
            items.append(f"Noeud choisi présent dans top attribution (rang {rank})")
        else:
            items.append(
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
            items.append(
                f"Noeud choisi proche du noeud courant (distance {d_curr_chosen:.3f}, quantile {pct:.2f})"
            )
        elif pct >= 0.7:
            items.append(
                f"Noeud choisi plutot éloigné du noeud courant (distance {d_curr_chosen:.3f}, quantile {pct:.2f})"
            )
        else:
            items.append(
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
                    items.append(
                        f"Noeud choisi proche du cluster attribué (dist top {d_top:.3f} vs autres {d_other:.3f})"
                    )
                elif d_top > d_other * 1.1:
                    items.append(
                        f"Noeud choisi éloigné du cluster attribué (dist top {d_top:.3f} vs autres {d_other:.3f})"
                    )
                else:
                    items.append(
                        f"Noeud choisi à distance comparable du cluster attribué (dist top {d_top:.3f} vs autres {d_other:.3f})"
                    )

    if not items:
        items.append(
            "Sélection probablement guidée par des contraintes internes non exposées ici"
        )

    return {
        "step": step,
        "variant_code": variant_code,
        "summary": f"noeud {chosen} choisi depuis le noeud {current}",
        "items": items,
    }


def explain_step(trace: dict, step: int) -> str:
    payload = explain_step_structured(trace, step)
    summary = str(payload.get("summary", "hors plage"))
    items = [str(item) for item in payload.get("items", [])]
    variant_code = str(payload.get("variant_code", "")).strip()
    variant_prefix = f"(variante {variant_code}) " if variant_code else ""
    if summary == "hors plage":
        return f"Etape {step}: hors plage."
    return f"Etape {step}: {variant_prefix}{summary}. " + _format_reasons(items)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate textual XAI explanations for selected instances/steps"
    )
    parser.add_argument("--report", required=True, help="Path to action_explainer JSON")
    parser.add_argument(
        "--instance", type=int, default=0, help="Instance index in report"
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

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)

    instances = report.get("instances", [])
    if not instances:
        raise ValueError(
            "Report has no 'instances'. Re-run action_explainer with +xai.save_instance_traces=True"
        )
    if not (0 <= args.instance < len(instances)):
        raise ValueError(
            f"Invalid instance index {args.instance}. Available range: [0, {len(instances)-1}]"
        )

    trace = instances[args.instance]
    actions = trace.get("actions", [])
    steps = _parse_step_list(args.steps, len(actions))
    variant_code = str(trace.get("instance_variant_code", "")).strip()
    variant_flags = trace.get("instance_variant_flags", {})
    active_constraints = trace.get("instance_active_constraints", [])

    lines: List[str] = []
    lines.append("# Explications Textuelles XAI")
    lines.append("")
    lines.append("## Contexte")
    lines.append("")
    lines.append(f"- report: `{report_path}`")
    lines.append(f"- instance: `{args.instance}`")
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
    lines.append("")
    trajectory_items = _format_report_trajectory(report.get("summary", {}) or {})
    trajectory_items.extend(_summarize_trace_trajectory(trace))
    if trajectory_items:
        lines.append("## Trajectoire globale")
        lines.append("")
        for item in trajectory_items:
            lines.append(f"- {item}")
        lines.append("")
    lines.append("## Etapes")
    lines.append("")
    for step in steps:
        payload = explain_step_structured(trace, step)
        step_idx = int(payload.get("step", step))
        summary = str(payload.get("summary", "hors plage"))
        items = [str(item) for item in payload.get("items", [])]
        lines.append(f"- Etape {step_idx}")
        lines.append(f"  - Décision: {summary}")
        for item in items:
            lines.append(f"  - {item}")
        lines.append("")

    text = "\n".join(lines)
    print(text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
