"""
make_architecture_diagram.py — generates reports/figures/architecture.png
Run once: python src/make_architecture_diagram.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(13, 7))
ax.set_xlim(0, 13)
ax.set_ylim(0, 7)
ax.axis("off")

BOX_STYLE = dict(boxstyle="round,pad=0.4", linewidth=1.5)

boxes = {
    "data":      (0.4, 3.5, 2.0, 1.0, "UCI Heart Disease\nDataset", "#DCE6F1"),
    "prep":      (3.0, 3.5, 2.0, 1.0, "Preprocessing\n+ EDA", "#DCE6F1"),
    "train":     (5.6, 3.5, 2.2, 1.0, "Training\n(LR / RF / XGB)\n+ MLflow tracking", "#D9EAD3"),
    "registry":  (8.4, 3.5, 2.0, 1.0, "Model Artifact\n(model.joblib)", "#D9EAD3"),
    "docker":    (8.4, 1.8, 2.0, 1.0, "Docker Image\n(FastAPI /predict)", "#FCE5CD"),
    "k8s":       (8.4, 0.1, 2.0, 1.0, "Kubernetes\n(Minikube / Cloud)", "#FCE5CD"),
    "monitor":   (11.0, 0.1, 1.7, 2.7, "Monitoring\nPrometheus\n+ Grafana", "#F4CCCC"),
    "cicd":      (3.0, 5.3, 4.6, 1.2, "GitHub Actions CI/CD\nlint -> train -> test -> docker build", "#EAD1DC"),
}

centers = {}
for key, (x, y, w, h, label, color) in boxes.items():
    box = FancyBboxPatch((x, y), w, h, **BOX_STYLE, facecolor=color, edgecolor="#444444")
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5)
    centers[key] = (x, y, w, h)


def arrow(src, dst, src_side="right", dst_side="left"):
    x, y, w, h = centers[src]
    x2, y2, w2, h2 = centers[dst]
    side_point = {
        "right": (x + w, y + h / 2), "left": (x, y + h / 2),
        "top": (x + w / 2, y + h), "bottom": (x + w / 2, y),
    }
    p1 = side_point[src_side]
    p2 = side_point[dst_side]
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15,
                                  color="#555555", linewidth=1.3))


arrow("data", "prep")
arrow("prep", "train")
arrow("train", "registry")
arrow("registry", "docker", src_side="bottom", dst_side="top")
arrow("docker", "k8s", src_side="bottom", dst_side="top")
arrow("k8s", "monitor", src_side="right", dst_side="left")

ax.text(9.4, 3.05, "logs metrics/artifacts to\nreports/figures + mlruns/", fontsize=7, ha="center", color="#555555")

ax.set_title("Heart Disease Risk Prediction — MLOps Architecture", fontsize=13, fontweight="bold")

fig.tight_layout()
fig.savefig(FIG_DIR / "architecture.png", dpi=160)
print(f"saved -> {FIG_DIR / 'architecture.png'}")
