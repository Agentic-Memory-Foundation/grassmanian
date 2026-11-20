import json
from pathlib import Path

r = json.loads(Path("results/results.json").read_text())
out = Path("../generated")
out.mkdir(exist_ok=True)

lines = []
lines.append("% auto table rows")
lines.append("\\begin{tabular}{llrrrrr}")
lines.append("\\toprule")
lines.append("Router & $k$ & $P$ & Neigh.\\ R@10 & Rel.\\ R@10 & Probe\\% & ms \\\\")
lines.append("\\midrule")
rows = [x for x in r["scifact"]["rows"] if x["n_clusters"] == 32]
for x in sorted(rows, key=lambda z: (z["routing"], z["subspace_k"], z["n_probe"])):
    rel = x.get("relevance_recall@10")
    rels = f"{rel*100:.1f}" if rel is not None else "--"
    routing_latex = x['routing'].replace('_', '\\_')
    lines.append(
        f"{routing_latex} & {x['subspace_k']} & {x['n_probe']} & "
        f"{x['neighbor_recall@10']*100:.1f} & {rels} & "
        f"{x['probed_frac']*100:.1f} & {x['query_ms']:.2f} \\\\"
    )
lines.append("\\bottomrule")
lines.append("\\end{tabular}")
(out / "table_scifact.tex").write_text("\n".join(lines) + "\n")

# oracle synthetic
olines = ["\\begin{tabular}{lrrr}", "\\toprule", "Router / partition & $P$ & Neigh.\\ R@10 & Probe\\% \\\\", "\\midrule"]
for x in sorted(
    [z for z in r["synthetic"]["rows"] if z["subspace_k"] == 4 and z["n_probe"] in (1, 2, 4)],
    key=lambda z: (z.get("partition", ""), z["routing"], z["n_probe"]),
):
    routing_latex = x['routing'].replace('_', '\\_')
    olines.append(
        f"{x.get('partition')}/{routing_latex} & {x['n_probe']} & "
        f"{x['neighbor_recall@10']*100:.1f} & {x['probed_frac']*100:.1f} \\\\"
    )
olines += ["\\bottomrule", "\\end{tabular}"]
(out / "table_synthetic.tex").write_text("\n".join(olines) + "\n")
print("wrote tables")
