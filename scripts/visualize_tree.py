"""
Modulo: visualize_tree (Script - scripts/)
Visualizzazione degli alberi decisionali (data/trees/*.json) senza grovigli:
ogni nodo e' uno spicchio/rettangolo la cui AREA e' proporzionale al numero di soluzioni
(len(possible_solutions)) e il cui COLORE codifica la profondita' media del sotto-albero.

Tre layout (--kind): sunburst (default), icicle, treemap.

Tre modalita' di colore:
  - relative : media delle mosse DAL nodo in poi (average_depth(node)).
  - absolute : media delle mosse TOTALI delle partite che passano dal nodo
               (= turno del nodo - 1 + relative). Confrontabile tra nodi di livelli diversi.
  - delta    : si attiva passando DUE alberi. Colora la differenza (A - B) del costo
               dello STESSO stato (stesso insieme di soluzioni) nei due alberi.
               Blu = A meglio, rosso = B meglio, bianco = pari (o stato assente in B).
               La geometria e' quella di A.

Canale secondario: bordo spesso scuro sui nodi che contengono almeno una partita
con profondita' >= --highlight (default: il caso peggiore dell'albero A).
Configurabile con --highlight N (soglia), --no-highlight (disattiva),
--border-color e --border-width (aspetto del bordo).

Valore colorato (--metric), valido per tutte e tre le modalita' di colore:
  - avg   : media delle mosse (default).
  - worst : caso peggiore (profondita' massima raggiunta dal sotto-albero).
Etichetta dei nodi (--label):
  - guess    : la mossa giocata nel nodo (default).
  - feedback : il feedback che porta al nodo (∅ per (0,0)).
  - none     : nessun testo, resta solo la struttura colorata.

Una sola visita post-order per albero: nessun ricalcolo quadratico di average_depth.

Esempi (dalla root del progetto):
    python -m scripts.visualize_tree max_entropy
    python -m scripts.visualize_tree max_entropy --kind icicle --color absolute
    python -m scripts.visualize_tree max_entropy estimated_evm_ul        # delta A-B
    python -m scripts.visualize_tree max_entropy --branch 0,2            # solo il ramo (0,2)
    python -m scripts.visualize_tree max_entropy --min-n 1 --maxdepth 6  # niente pruning
    python -m scripts.visualize_tree max_entropy --metric worst --label feedback
    python -m scripts.visualize_tree max_entropy estimated_evm_ul --metric worst --label none
"""
from __future__ import annotations

import argparse
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go

from src.analysis.metrics import max_depth
from src.analysis.tree_builder import TreeNode
from src.analysis.tree_io import load_tree_from_json
from src.cli.game_loop import fb_tuple_to_string
from src.core.constants import DIM

TREES_DIR = Path("data/trees_to_draw" )
PLOTS_DIR = Path("data/plots")

Distribution = Dict[int, int]  # {mosse_dal_nodo: numero_partite}

COLOR_RELATIVE = "relative"
COLOR_ABSOLUTE = "absolute"
COLOR_DELTA = "delta"

METRIC_AVG = "avg"
METRIC_WORST = "worst"

LABEL_GUESS = "guess"
LABEL_FEEDBACK = "feedback"
LABEL_NONE = "none"

LINE_NORMAL = ("rgba(255,255,255,0.55)", 0.4)
LINE_HIGHLIGHT = ("#111111", 2.4)


# ---------------------------------------------------------------------------
# STRUTTURE DATI
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VizConfig:
    """Parametri di una visita."""
    mode: str
    min_n: int
    highlight_depth: int
    reference: Optional[Dict[str, float]] = None  # {state_id: valore dal nodo} dell'albero B (solo delta)
    metric: str = METRIC_AVG
    label_mode: str = LABEL_GUESS
    border_color: str = LINE_HIGHLIGHT[0]
    border_width: float = LINE_HIGHLIGHT[1]


@dataclass(slots=True)
class Rows:
    """Colonne piatte pronte per Plotly (una riga per nodo disegnato)."""
    ids: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    colors: List[float] = field(default_factory=list)
    hover: List[str] = field(default_factory=list)
    line_colors: List[str] = field(default_factory=list)
    line_widths: List[float] = field(default_factory=list)
    highlight_line: Tuple[str, float] = LINE_HIGHLIGHT

    def add(self, node_id: str, label: str, parent: str, value: float,
            color: float, hover: str, highlight: bool) -> None:
        line_color, line_width = self.highlight_line if highlight else LINE_NORMAL
        self.ids.append(node_id)
        self.labels.append(label)
        self.parents.append(parent)
        self.values.append(value)
        self.colors.append(color)
        self.hover.append(hover)
        self.line_colors.append(line_color)
        self.line_widths.append(line_width)


@dataclass(slots=True)
class _PrunedAcc:
    """Accumulatore dei sotto-alberi potati (N < min_n) di un nodo."""
    count: int = 0
    n: int = 0
    leaves: int = 0
    color_sum: float = 0.0
    extreme: float = 0.0   # valore di colore con il modulo massimo (usato per il caso peggiore)
    worst: int = 0

    def add(self, n: int, dist: Distribution, color: float, child_depth: int) -> None:
        leaves = sum(dist.values())
        self.count += 1
        self.n += n
        self.leaves += leaves
        self.color_sum += color * leaves
        if self.count == 1 or abs(color) > abs(self.extreme):
            self.extreme = color
        self.worst = max(self.worst, child_depth - 1 + max(dist))

    def color(self, metric: str) -> float:
        """Media: media pesata sulle partite. Caso peggiore: il valore piu' estremo."""
        return self.color_sum / self.leaves if metric == METRIC_AVG else self.extreme


# ---------------------------------------------------------------------------
# UTILITA'
# ---------------------------------------------------------------------------
def _state_id(solutions: List[str]) -> str:
    """Chiave canonica dello stato (identica a generate_id del progetto)."""
    return ",".join(sorted(solutions))


def _fb_label(fb: Tuple[int, int]) -> str:
    """'++-' oppure '∅' per il feedback (0,0), che altrimenti sarebbe una stringa vuota."""
    text = fb_tuple_to_string(fb)
    return text if text else "∅"


def _avg(dist: Distribution) -> float:
    """Media delle mosse dalla distribuzione {mosse: partite}."""
    leaves = sum(dist.values())
    return sum(k * v for k, v in dist.items()) / leaves


def _metric_value(dist: Distribution, metric: str) -> float:
    """Valore dal nodo: media delle mosse oppure caso peggiore (mosse massime)."""
    return _avg(dist) if metric == METRIC_AVG else float(max(dist))


def _node_label(guess: str, path_text: str, label_mode: str) -> str:
    """Testo mostrato sul nodo: mossa, feedback che porta al nodo, oppure nulla."""
    if label_mode == LABEL_GUESS:
        return guess
    if label_mode == LABEL_FEEDBACK:
        return path_text.rsplit(" → ", 1)[-1]
    return ""


def _add_shifted(target: Distribution, child_dist: Distribution) -> None:
    """Aggiunge la distribuzione del figlio spostata di +1 mossa (la mossa del padre)."""
    for k, v in child_dist.items():
        target[k + 1] = target.get(k + 1, 0) + v


# ---------------------------------------------------------------------------
# ALBERO DI RIFERIMENTO (B) PER LA MODALITA' DELTA
# ---------------------------------------------------------------------------
def build_state_costs(tree: TreeNode, metric: str = METRIC_AVG) -> Dict[str, float]:
    """
    PRE: tree e' un albero caricato da JSON. metric e' 'avg' o 'worst'.
    POST: Ritorna {state_id: media mosse (o caso peggiore) dal nodo} per ogni nodo
          dell'albero (visita unica).
    """
    costs: Dict[str, float] = {}

    def visit(node: TreeNode) -> Distribution:
        dist: Distribution = {}
        for child in node["children"].values():
            if child is None:
                dist[1] = dist.get(1, 0) + 1
            else:
                _add_shifted(dist, visit(child))
        costs[_state_id(node["possible_solutions"])] = _metric_value(dist, metric)
        return dist

    visit(tree)
    return costs


# ---------------------------------------------------------------------------
# VISITA PRINCIPALE
# ---------------------------------------------------------------------------
def _visit(
    node: TreeNode,
    depth: int,
    node_id: str,
    parent_id: str,
    path_text: str,
    cfg: VizConfig,
    rows: Rows,
    emit: bool,
) -> Tuple[Distribution, float]:
    """
    Visita post-order. Ritorna (distribuzione_mosse_dal_nodo, valore_colore).

    Se emit=True aggiunge a `rows` la riga del nodo; i figli con N < cfg.min_n non
    vengono disegnati ma aggregati in un'unica riga "rest" colorata con la media pesata.

    Valore della riga (branchvalues="remainder"): e' la parte di area NON coperta dai figli,
    cioe' 1 se la mossa del nodo e' essa stessa una soluzione (la partita vinta qui), 0 altrimenti.
    Cosi' l'area totale di ogni nodo risulta esattamente N.
    """
    solutions = node["possible_solutions"]
    n = len(solutions)
    children = node["children"]

    dist: Distribution = {}
    pruned = _PrunedAcc()

    for fb, child in children.items():
        if child is None:
            dist[1] = dist.get(1, 0) + 1
            continue
        n_child = len(child["possible_solutions"])
        child_emit = emit and n_child >= cfg.min_n
        c_dist, c_color = _visit(
            child,
            depth + 1,
            f"{node_id}|{fb[0]}{fb[1]}",
            node_id,
            f"{path_text} → {_fb_label(fb)}",
            cfg,
            rows,
            child_emit,
        )
        _add_shifted(dist, c_dist)
        if emit and not child_emit:
            pruned.add(n_child, c_dist, c_color, depth + 1)

    avg = _avg(dist)
    abs_avg = depth - 1 + avg
    worst_abs = depth - 1 + max(dist)

    # Valore su cui si basa il colore (relativo al nodo): media o caso peggiore
    rel_value = _metric_value(dist, cfg.metric)

    ref: Optional[float] = None
    if cfg.mode == COLOR_RELATIVE:
        color = rel_value
    elif cfg.mode == COLOR_ABSOLUTE:
        color = depth - 1 + rel_value
    else:
        ref = cfg.reference.get(_state_id(solutions)) if cfg.reference else None
        color = (rel_value - ref) if ref is not None else 0.0

    if not emit:
        return dist, color

    # --- riga aggregata dei sotto-alberi potati ---
    if pruned.count:
        rest_color = pruned.color(cfg.metric)
        fmt = "+.3f" if cfg.mode == COLOR_DELTA else ".3f"
        how = "media pesata" if cfg.metric == METRIC_AVG else "valore massimo"
        rest_hover = (
            f"<b>{pruned.count} sotto-bivi aggregati</b> (N &lt; {cfg.min_n})<br>"
            f"Dopo: {path_text}<br>Soluzioni totali: {pruned.n}<br>"
            f"Colore ({how}): {rest_color:{fmt}}<br>"
            f"Caso peggiore: T{pruned.worst}"
        )
        rest_label = f"…×{pruned.count}" if cfg.label_mode != LABEL_NONE else ""
        rows.add(
            f"{node_id}|rest", rest_label, node_id, pruned.n,
            rest_color, rest_hover, pruned.worst >= cfg.highlight_depth,
        )

    # --- riga del nodo ---
    dist_text = ", ".join(f"T{k + depth - 1}: {v}" for k, v in sorted(dist.items()))
    hover = (
        f"<b>{node['guess']}</b> (turno {depth})<br>"
        f"Percorso: {path_text}<br>"
        f"Soluzioni N = {n}<br>"
        f"Media dal nodo: {avg:.3f}<br>"
        f"Media totale partite: {abs_avg:.3f}<br>"
        f"Caso peggiore: T{worst_abs}<br>"
        f"Distribuzione: {dist_text}"
    )
    if cfg.mode == COLOR_DELTA:
        if ref is None:
            hover += "<br><i>Stato assente nell'albero B</i>"
        else:
            what = "Media" if cfg.metric == METRIC_AVG else "Caso peggiore"
            hover += f"<br>{what} B (dal nodo): {ref:.3f} → Δ = {rel_value - ref:+.3f}"

    win_here = 1 if (DIM, 0) in children else 0
    rows.add(node_id, _node_label(node["guess"], path_text, cfg.label_mode), parent_id,
             win_here, color, hover, worst_abs >= cfg.highlight_depth)
    return dist, color


def flatten_tree(root: TreeNode, root_depth: int, root_id: str, path_text: str, cfg: VizConfig) -> Rows:
    """
    PRE: root e' il nodo da cui partire (root_depth = suo turno, 1 per la radice vera).
    POST: Ritorna le colonne piatte (ids, parents, values, colors, ...) per Plotly.
    """
    rows = Rows(highlight_line=(cfg.border_color, cfg.border_width))
    _visit(root, root_depth, root_id, "", path_text, cfg, rows, emit=True)
    return rows


# ---------------------------------------------------------------------------
# FIGURA
# ---------------------------------------------------------------------------
def resolve_color_range(
    values: List[float], mode: str, cmin: Optional[float], cmax: Optional[float]
) -> Tuple[float, float]:
    """
    Dominio colore robusto: percentili 2-98 (non min-max) cosi' gli outlier non
    schiacciano le differenze piccole. In delta il range e' simmetrico attorno a 0
    (cmax = semi-ampiezza).
    """
    arr = np.asarray(values, dtype=float)
    if mode == COLOR_DELTA:
        half = cmax if cmax is not None else float(np.percentile(np.abs(arr), 98))
        if half <= 0:
            half = float(np.max(np.abs(arr)))
        half = max(half, 1e-3)
        return -half, half
    lo = cmin if cmin is not None else float(np.percentile(arr, 2))
    hi = cmax if cmax is not None else float(np.percentile(arr, 98))
    if hi <= lo:
        hi = lo + 1e-3
    return lo, hi


def build_figure(
    rows: Rows,
    kind: str,
    mode: str,
    metric: str,
    title: str,
    maxdepth: int,
    cmap: Optional[str],
    cmin: Optional[float],
    cmax: Optional[float],
) -> go.Figure:
    """POST: Ritorna la figura Plotly (sunburst / icicle / treemap) pronta da salvare."""
    lo, hi = resolve_color_range(rows.colors, mode, cmin, cmax)

    #default_cmap = {COLOR_RELATIVE: "YlOrRd", COLOR_ABSOLUTE: "Blues", COLOR_DELTA: "RdBu_r"}[mode]
    default_cmap = {COLOR_RELATIVE: "YlOrRd", COLOR_ABSOLUTE: "YlGn", COLOR_DELTA: "RdBu_r"}[mode]
    colorbar_title = {
        (COLOR_RELATIVE, METRIC_AVG): "Media mosse<br>dal nodo",
        (COLOR_ABSOLUTE, METRIC_AVG): "Media mosse<br>totali partita",
        (COLOR_DELTA, METRIC_AVG): "Δ media (A − B)<br>blu = A meglio",
        (COLOR_RELATIVE, METRIC_WORST): "Caso peggiore<br>dal nodo (mosse)",
        (COLOR_ABSOLUTE, METRIC_WORST): "Caso peggiore<br>partita (turno)",
        (COLOR_DELTA, METRIC_WORST): "Δ caso peggiore (A − B)<br>blu = A meglio",
    }[(mode, metric)]

    marker = dict(
        colors=rows.colors,
        colorscale=cmap or default_cmap,
        cmin=lo,
        cmax=hi,
        showscale=True,
        colorbar=dict(title=colorbar_title),
        line=dict(color=rows.line_colors, width=rows.line_widths),
    )

    common = dict(
        ids=rows.ids,
        labels=rows.labels,
        parents=rows.parents,
        values=rows.values,
        branchvalues="remainder",
        hovertext=rows.hover,
        hoverinfo="text",
        marker=marker,
        maxdepth=maxdepth,
    )

    if kind == "sunburst":
        trace = go.Sunburst(insidetextorientation="radial", **common)
    elif kind == "icicle":
        trace = go.Icicle(**common)
    else:
        trace = go.Treemap(**common)

    fig = go.Figure(trace)
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(t=70, l=10, r=10, b=10),
        height=900,
    )
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_named_tree(name: str) -> TreeNode:
    path = TREES_DIR / f"{name}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in TREES_DIR.glob("*.json"))) or "(nessuno)"
        raise SystemExit(f"Albero '{name}' non trovato in {TREES_DIR}/. Disponibili: {available}")
    return load_tree_from_json(str(path))


def descend_branch(tree: TreeNode, branch: str) -> Tuple[TreeNode, int, str, str]:
    """
    PRE: branch es. '0,2' oppure '0,2/1,1' (sequenza di feedback bulls,cows separati da '/').
    POST: Ritorna (nodo, turno_del_nodo, root_id, path_text) per ripartire da quel sotto-albero.
    """
    node = tree
    depth = 1
    node_id = "START"
    path_text = "START"
    for token in branch.split("/"):
        try:
            bulls, cows = (int(x) for x in token.split(","))
        except ValueError:
            raise SystemExit(f"Feedback '{token}' non valido: usa il formato bulls,cows (es. 0,2)")
        child = node["children"].get((bulls, cows))
        if child is None:
            options = ", ".join(f"{b},{c}" for (b, c), ch in node["children"].items() if ch is not None)
            raise SystemExit(f"Il feedback {bulls},{cows} non esiste qui. Disponibili: {options}")
        node = child
        depth += 1
        node_id += f"|{bulls}{cows}"
        path_text += f" → {_fb_label((bulls, cows))}"
    return node, depth, node_id, path_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualizza alberi decisionali come sunburst/icicle/treemap.")
    p.add_argument("trees", nargs="+", metavar="ALBERO",
                   help="Nome (senza .json) in data/trees/. Uno solo = colore assoluto/relativo; "
                        "due = confronto delta (A - B).")
    p.add_argument("--kind", choices=["sunburst", "icicle", "treemap"], default="sunburst")
    p.add_argument("--color", choices=[COLOR_ABSOLUTE, COLOR_RELATIVE], default=COLOR_ABSOLUTE,
                   help="Metrica del colore con un solo albero (default: absolute).")
    p.add_argument("--metric", choices=[METRIC_AVG, METRIC_WORST], default=METRIC_AVG,
                   help="Valore da colorare: avg = media mosse (default), worst = caso peggiore.")
    p.add_argument("--label", choices=[LABEL_GUESS, LABEL_FEEDBACK, LABEL_NONE], default=LABEL_GUESS,
                   help="Testo sui nodi: guess = mossa (default), feedback = feedback che porta al nodo, "
                        "none = nessuna etichetta.")
    p.add_argument("--min-n", type=int, default=3,
                   help="Sotto-alberi con meno di N soluzioni vengono aggregati (default 3; 1 = nessun pruning).")
    p.add_argument("--maxdepth", type=int, default=4,
                   help="Livelli visibili all'apertura; si espande cliccando (default 4).")
    p.add_argument("--branch", type=str, default=None,
                   help="Parti da un ramo, es. '0,2' oppure '0,2/1,1' (feedback bulls,cows).")
    p.add_argument("--highlight", type=int, default=None,
                   help="Bordo spesso sui nodi con partite di profondita' >= valore (default: caso peggiore di A).")
    p.add_argument("--no-highlight", action="store_true",
                   help="Disattiva del tutto il bordo di evidenziazione del caso peggiore.")
    p.add_argument("--border-color", type=str, default=LINE_HIGHLIGHT[0],
                   help="Colore del bordo di evidenziazione (default #111111).")
    p.add_argument("--border-width", type=float, default=LINE_HIGHLIGHT[1],
                   help="Spessore del bordo di evidenziazione (default 2.4).")
    p.add_argument("--cmap", type=str, default=None, help="Colorscale Plotly (default YlOrRd / RdBu_r).")
    p.add_argument("--cmin", type=float, default=None, help="Minimo colore (assoluto/relativo).")
    p.add_argument("--cmax", type=float, default=None,
                   help="Massimo colore; in delta e' la semi-ampiezza simmetrica.")
    p.add_argument("--out", type=Path, default=None, help="File HTML di output.")
    p.add_argument("--offline", action="store_true", help="Incorpora plotly.js nell'HTML (file piu' pesante).")
    p.add_argument("--no-open", action="store_true", help="Non aprire il browser.")
    args = p.parse_args()
    if len(args.trees) > 2:
        p.error("Al massimo due alberi (A e B).")
    return args


def main() -> None:
    args = parse_args()
    name_a = args.trees[0]
    name_b = args.trees[1] if len(args.trees) == 2 else None

    print(f"Caricamento albero A: {name_a}...")
    tree_a = load_named_tree(name_a)

    reference: Optional[Dict[str, float]] = None
    if name_b is not None:
        print(f"Caricamento albero B: {name_b}...")
        reference = build_state_costs(load_named_tree(name_b), args.metric)
        mode = COLOR_DELTA
    else:
        mode = args.color

    if args.no_highlight:
        highlight = 10**9  # soglia irraggiungibile: nessun nodo evidenziato
    else:
        highlight = args.highlight if args.highlight is not None else max_depth(tree_a)
    cfg = VizConfig(
        mode=mode,
        min_n=args.min_n,
        highlight_depth=highlight,
        reference=reference,
        metric=args.metric,
        label_mode=args.label,
        border_color=args.border_color,
        border_width=args.border_width,
    )

    if args.branch:
        root, root_depth, root_id, path_text = descend_branch(tree_a, args.branch)
    else:
        root, root_depth, root_id, path_text = tree_a, 1, "START", "START"

    print("Appiattimento albero (visita unica)...")
    rows = flatten_tree(root, root_depth, root_id, path_text, cfg)
    border_info = "bordo disattivato" if args.no_highlight else f"bordo spesso da T{highlight}"
    print(f"Nodi disegnati: {len(rows.ids)} (min_n={args.min_n}, {border_info})")

    what = "media mosse" if args.metric == METRIC_AVG else "caso peggiore"
    if mode == COLOR_DELTA:
        title = f"{name_a}  vs  {name_b} — Δ {what} per stato (A − B)"
    else:
        scope = "totale partita" if mode == COLOR_ABSOLUTE else "dal nodo"
        title = f"{name_a} — colore: {what} ({scope})"
    if args.branch:
        title += f" — ramo {args.branch}"

    fig = build_figure(rows, args.kind, mode, args.metric, title, args.maxdepth, args.cmap, args.cmin, args.cmax)

    suffix = f"{name_a}_vs_{name_b}" if name_b else name_a
    if args.metric == METRIC_WORST:
        suffix += "_worst"
    if args.branch:
        suffix += "_branch_" + args.branch.replace(",", "").replace("/", "-")
    out = args.out or PLOTS_DIR / f"{suffix}_{args.kind}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs=True if args.offline else "cdn")
    print(f"Salvato in {out}")

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()