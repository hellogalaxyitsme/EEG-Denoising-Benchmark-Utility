#!/usr/bin/env python3
"""Generate the simplified Figure 1 architecture schematic.

The figure is a schematic, not a quantitative result plot. It therefore saves
the plotted nodes/edges and verification checks as figure-data artifacts.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from jne_figure_style import COLORS, WIDTH_FULL, save_figure, set_jne_style


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    ROOT / "Revised_Paper" / "main.tex",
    ROOT / "Revised_Paper" / "supplementary_material.tex",
]
OUT_DIR = ROOT / "Revised_Paper" / "Images"
DATA_DIR = ROOT / "Revised_Paper" / "figure_data"


@dataclass(frozen=True)
class Node:
    node_id: str
    label: str
    role: str
    x: float
    y: float
    w: float
    h: float

    @property
    def left(self) -> float:
        return self.x

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y

    @property
    def top(self) -> float:
        return self.y + self.h

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0


def node_color(role: str) -> tuple[str, str]:
    if role in {"input", "output"}:
        return "#F3F4F6", "#6B7280"
    if role == "bottleneck":
        return "#FDE7CF", "#C56B1F"
    if role in {"up", "concat", "decoder", "head"}:
        return "#E6F4EF", "#2A9D8F"
    return "#EAF2FB", "#4C78A8"


def make_nodes() -> list[Node]:
    specs = [
        ("input", "Contaminated\nEEG", "input", 1.20),
        ("stem", "Stem\n$1\\!\\times\\!1$ Conv\nC", "encoder", 0.94),
        ("enc1", "Encoder 1\nDSConv\nC", "encoder", 1.02),
        ("down1", "Down\nC → 2C", "encoder", 0.72),
        ("enc2", "Encoder 2\nDSConv\n2C", "encoder", 1.02),
        ("down2", "Down\n2C → 4C", "encoder", 0.78),
        ("bneck", "Bottleneck\nDSConv\nd=4,8,16\n4C", "bottleneck", 1.12),
        ("up1", "UpConv\n4C → 2C", "up", 0.78),
        ("cat1", "Concat", "concat", 0.70),
        ("dec1", "Decoder 1\nDSConv\n2C", "decoder", 1.02),
        ("up2", "UpConv\n2C → C", "up", 0.78),
        ("cat2", "Concat", "concat", 0.70),
        ("dec2", "Decoder 2\nDSConv\nC", "decoder", 1.02),
        ("head", "$1\\!\\times\\!1$\nclean\nhead", "head", 0.82),
        ("output", "Denoised\nEEG", "output", 1.08),
    ]
    x = 0.22
    y = 1.00
    h = 0.74
    gap = 0.145
    nodes = []
    for node_id, label, role, w in specs:
        nodes.append(Node(node_id=node_id, label=label, role=role, x=x, y=y, w=w, h=h))
        x += w + gap
    return nodes


def draw_box(ax, node: Node):
    face, edge = node_color(node.role)
    patch = FancyBboxPatch(
        (node.x, node.y),
        node.w,
        node.h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        facecolor=face,
        edgecolor=edge,
        linewidth=0.75,
        zorder=2,
    )
    ax.add_patch(patch)
    fontsize = 5.45
    if node.role in {"input", "output", "bottleneck"}:
        fontsize = 5.15
    if node.node_id == "input":
        fontsize = 4.35
    if node.node_id == "stem":
        fontsize = 4.85
    if node.node_id in {"down1", "down2", "up1", "up2", "head"}:
        fontsize = 5.05
    text = ax.text(
        node.cx,
        node.cy,
        node.label,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=COLORS["text"],
        linespacing=1.05,
        zorder=3,
    )
    return patch, text


def draw_main_arrow(ax, left: Node, right: Node):
    patch = FancyArrowPatch(
        (left.right + 0.012, left.cy),
        (right.left - 0.012, right.cy),
        arrowstyle="-|>",
        mutation_scale=6.0,
        linewidth=0.62,
        color="#4B5563",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(patch)
    return patch


def draw_skip(ax, start: Node, end: Node, y_offset: float):
    y_top = start.top + y_offset
    points = [
        (start.cx, start.top),
        (start.cx, y_top),
        (end.cx, y_top),
        (end.cx, end.top),
    ]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    line = ax.plot(xs, ys, color="#4B5563", linewidth=0.78, solid_capstyle="butt", zorder=1)[0]
    arrow = FancyArrowPatch(
        (end.cx, end.top + 0.07),
        (end.cx, end.top + 0.012),
        arrowstyle="-|>",
        mutation_scale=5.0,
        linewidth=0.78,
        color="#4B5563",
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    ax.add_patch(arrow)
    return line, arrow, points


def bboxes_overlap(a: Node, b: Node, pad: float = 0.012) -> bool:
    return not (
        a.right + pad <= b.left
        or b.right + pad <= a.left
        or a.top + pad <= b.bottom
        or b.top + pad <= a.bottom
    )


def segment_intersects_display_bbox(p0, p1, bbox) -> bool:
    x0, y0 = p0
    x1, y1 = p1
    xmin, ymin, xmax, ymax = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    if xmin <= x0 <= xmax and ymin <= y0 <= ymax:
        return True
    if xmin <= x1 <= xmax and ymin <= y1 <= ymax:
        return True
    dx = x1 - x0
    dy = y1 - y0
    candidates = []
    if abs(dx) > 1e-9:
        candidates.extend([(xmin - x0) / dx, (xmax - x0) / dx])
    if abs(dy) > 1e-9:
        candidates.extend([(ymin - y0) / dy, (ymax - y0) / dy])
    for t in candidates:
        if 0.0 <= t <= 1.0:
            x = x0 + t * dx
            y = y0 + t * dy
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
    return False


def verify_geometry(fig, ax, nodes: list[Node], texts: dict[str, object], skip_paths: dict[str, list[tuple[float, float]]]) -> list[dict[str, str]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    checks: list[dict[str, str]] = []

    lookup = {node.node_id: node for node in nodes}
    expected = {
        "skip_enc2_to_cat1": ("enc2", "cat1"),
        "skip_enc1_to_cat2": ("enc1", "cat2"),
    }
    for name, (source_id, target_id) in expected.items():
        source = lookup[source_id]
        target = lookup[target_id]
        points = skip_paths[name]
        starts_correctly = abs(points[0][0] - source.cx) < 1e-9 and abs(points[0][1] - source.top) < 1e-9
        ends_correctly = abs(points[-1][0] - target.cx) < 1e-9 and abs(points[-1][1] - target.top) < 1e-9
        orthogonal = all(
            abs(points[i][0] - points[i + 1][0]) < 1e-9 or abs(points[i][1] - points[i + 1][1]) < 1e-9
            for i in range(len(points) - 1)
        )
        arrowhead_points_toward_concat = points[-2][0] == points[-1][0] and points[-2][1] > points[-1][1]
        arrowhead_outside_text_area = points[-1][1] >= target.top and points[-1][1] <= target.top + 0.02
        checks.extend(
            [
                {"check": f"{name}_starts_on_{source_id}_top_edge", "status": "pass" if starts_correctly else "fail"},
                {"check": f"{name}_ends_on_{target_id}_top_edge", "status": "pass" if ends_correctly else "fail"},
                {"check": f"{name}_uses_orthogonal_segments", "status": "pass" if orthogonal else "fail"},
                {"check": f"{name}_arrowhead_points_toward_{target_id}", "status": "pass" if arrowhead_points_toward_concat else "fail"},
                {"check": f"{name}_arrowhead_clear_of_{target_id}_text", "status": "pass" if arrowhead_outside_text_area else "fail"},
            ]
        )

    overlap_pairs = []
    for i, node_a in enumerate(nodes):
        for node_b in nodes[i + 1 :]:
            if bboxes_overlap(node_a, node_b):
                overlap_pairs.append(f"{node_a.node_id}-{node_b.node_id}")
    checks.append({"check": "node_boxes_do_not_overlap", "status": "pass" if not overlap_pairs else "fail", "detail": ";".join(overlap_pairs)})

    text_spill = []
    for node in nodes:
        text_bbox = texts[node.node_id].get_window_extent(renderer=renderer)
        node_bbox_points = ax.transData.transform([(node.left, node.bottom), (node.right, node.top)])
        x0, y0 = node_bbox_points[0]
        x1, y1 = node_bbox_points[1]
        pad = 1.5
        if text_bbox.x0 < min(x0, x1) + pad or text_bbox.x1 > max(x0, x1) - pad or text_bbox.y0 < min(y0, y1) + pad or text_bbox.y1 > max(y0, y1) - pad:
            text_spill.append(node.node_id)
    checks.append({"check": "text_labels_fit_inside_boxes", "status": "pass" if not text_spill else "fail", "detail": ";".join(text_spill)})

    text_hits = []
    for skip_name, points in skip_paths.items():
        display_points = [ax.transData.transform(point) for point in points]
        for text_id, text in texts.items():
            bbox = text.get_window_extent(renderer=renderer).expanded(1.02, 1.08)
            for idx in range(len(display_points) - 1):
                if segment_intersects_display_bbox(display_points[idx], display_points[idx + 1], bbox):
                    text_hits.append(f"{skip_name}:{text_id}:segment{idx+1}")
    checks.append({"check": "skip_paths_do_not_intersect_text_labels", "status": "pass" if not text_hits else "fail", "detail": ";".join(text_hits)})

    failures = [row for row in checks if row["status"] != "pass"]
    if failures:
        detail = "; ".join(f"{row['check']}={row.get('detail', '')}" for row in failures)
        raise RuntimeError(f"Architecture geometry verification failed: {detail}")
    return checks


def write_source_data(nodes: list[Node], edges: list[dict[str, str]], checks: list[dict[str, str]]) -> tuple[Path, Path, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    node_path = DATA_DIR / "fig01_architecture_nodes.csv"
    edge_path = DATA_DIR / "fig01_architecture_edges.csv"
    check_path = DATA_DIR / "fig01_architecture_verification.csv"

    with node_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label", "role", "x", "y", "width", "height"])
        writer.writeheader()
        for node in nodes:
            writer.writerow(
                {
                    "id": node.node_id,
                    "label": node.label.replace("\n", " / "),
                    "role": node.role,
                    "x": f"{node.x:.4f}",
                    "y": f"{node.y:.4f}",
                    "width": f"{node.w:.4f}",
                    "height": f"{node.h:.4f}",
                }
            )

    with edge_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "target", "edge_type", "note"])
        writer.writeheader()
        writer.writerows(edges)

    with check_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "status", "detail"])
        writer.writeheader()
        for row in checks:
            writer.writerow({"check": row["check"], "status": row["status"], "detail": row.get("detail", "")})

    stale_inset = DATA_DIR / "fig01_architecture_inset.csv"
    if stale_inset.exists():
        stale_inset.unlink()
    return node_path, edge_path, check_path


def main() -> None:
    set_jne_style()
    nodes = make_nodes()
    right_edge = nodes[-1].right

    fig, ax = plt.subplots(figsize=(WIDTH_FULL, 2.0), constrained_layout=False)
    fig.subplots_adjust(left=0.018, right=0.988, top=0.91, bottom=0.18)
    ax.set_xlim(0.0, right_edge + 0.18)
    ax.set_ylim(0.35, 2.38)
    ax.axis("off")

    texts = {}
    for node in nodes:
        _patch, text = draw_box(ax, node)
        texts[node.node_id] = text

    edges: list[dict[str, str]] = []
    for source, target in zip(nodes[:-1], nodes[1:], strict=True):
        draw_main_arrow(ax, source, target)
        edges.append({"source": source.node_id, "target": target.node_id, "edge_type": "main_path", "note": ""})

    lookup = {node.node_id: node for node in nodes}
    _line1, _arrow1, skip1 = draw_skip(ax, lookup["enc2"], lookup["cat1"], y_offset=0.42)
    _line2, _arrow2, skip2 = draw_skip(ax, lookup["enc1"], lookup["cat2"], y_offset=0.62)
    skip_paths = {"skip_enc2_to_cat1": skip1, "skip_enc1_to_cat2": skip2}
    edges.extend(
        [
            {"source": "enc2", "target": "cat1", "edge_type": "skip", "note": "Encoder 2 output at 2C to first Concat"},
            {"source": "enc1", "target": "cat2", "edge_type": "skip", "note": "Encoder 1 output at C to second Concat"},
        ]
    )

    ax.text(
        right_edge / 2.0,
        0.55,
        r"Only base width $C$ is varied.",
        ha="center",
        va="center",
        fontsize=6.5,
        color=COLORS["text"],
    )

    checks = verify_geometry(fig, ax, nodes, texts, skip_paths)
    node_path, edge_path, check_path = write_source_data(nodes, edges, checks)
    pdf_path, png_path = save_figure(fig, OUT_DIR / "fig01_architecture_revised", dpi=400)
    plt.close(fig)

    print("source file(s) used:")
    for source in SOURCE_FILES:
        print(f"- {source}")
    print(f"number of observations plotted: {len(nodes)} architecture nodes and {len(edges)} connectors")
    print("observation type: schematic architecture components, not subjects or training seeds")
    print("aggregation performed before plotting: none")
    print("geometry verification: passed")
    print(f"source data: {node_path}")
    print(f"source data: {edge_path}")
    print(f"source data: {check_path}")
    print(f"final PDF: {pdf_path}")
    print(f"final PNG: {png_path}")


if __name__ == "__main__":
    main()
