"""从 results/benchmark_results.json 汇总实验结论，输出 Markdown 表格。

用法：python summarize_results.py [--input results/benchmark_results.json]
"""

import argparse
import json
import os


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="results/benchmark_results.json")
    args = ap.parse_args()
    with open(args.input, encoding="utf-8") as f:
        res = json.load(f)

    names = sorted(res)
    print("### 数据集规模\n")
    print("| 数据集 | 文档数 | 查询数(全) | 评估用查询 | grid 用查询 | 领域 |")
    print("|---|---:|---:|---:|---:|---|")
    domains = {
        "arguana": "论点/议论文检索",
        "fiqa": "金融问答",
        "nfcorpus": "医学信息检索",
        "nq": "开放域问答（超大规模）",
        "quora": "社区问答去重",
        "scidocs": "科学文献检索",
        "scifact": "科学论断证据",
        "trec-covid": "COVID 学术文献",
        "trec-covid-beir": "COVID 学术文献（同语料/不同标注）",
        "trec-covid-v2": "COVID 学术文献（v2 语料）",
        "vihealthqa": "越南语健康问答",
        "webis-touche2020": "论点检索",
    }
    for name in names:
        r = res[name]
        f = r["features"]
        try:
            with open(r["path"] + "/queries.jsonl", encoding="utf-8") as fh:
                q_total = sum(1 for _ in fh)
        except OSError:
            q_total = r.get("num_queries", "-")
        print(
            f"| {name} | {f['doc_count']:,} | {q_total:,} | "
            f"{r.get('eval_num_queries', '-')} | {r.get('grid_num_queries', '-')} | "
            f"{domains.get(name, '-')} |"
        )

    print("\n### 预测参数\n")
    print("| 数据集 | k1 | b | k3 | δ | IDF | 模型变体 |")
    print("|---|---:|---:|---:|---:|---|---|")
    for name in names:
        p = res[name]["predicted_params"]
        print(
            f"| {name} | {fmt(p['k1'])} | {fmt(p['b'])} | {fmt(p.get('k3', 0.0))} | "
            f"{fmt(p['delta'])} | "
            f"{p['idf_type']} | {p['model_variant']} |"
        )

    print("\n### 检索性能（default vs predicted vs grid search 上界）\n")
    print("| 数据集 | 指标 | default | predicted | 提升% | grid 上界 | 达上界% |")
    print("|---|---|--:|--:|--:|--:|--:|")
    for name in names:
        r = res[name]
        if "default_metrics" not in r:
            continue
        first = True
        for m in ("mrr@10", "ndcg@10", "recall@100"):
            d = r["default_metrics"][m]
            p = r["predicted_metrics"][m]
            imp = r.get(f"improve_vs_default_{m}", 0)
            if "grid_best_metrics" in r:
                g = r["grid_best_metrics"][m]
                reach = r.get(f"reach_grid_{m}", 0)
                gs = f"{g:.4f}"
                rr = f"{reach:.1f}"
            else:
                gs, rr = "-", "-"
            ds = name if first else ""
            first = False
            print(f"| {ds} | {m} | {d:.4f} | {p:.4f} | {imp:+.1f} | {gs} | {rr} |")

    # 同口径对照：grid 上界与 default/predicted 都基于同一查询子集
    matched_path = "results/benchmark_grid_sample_eval.json"
    if os.path.exists(matched_path):
        with open(matched_path, encoding="utf-8") as f:
            sam = json.load(f)
        print("\n### grid 上界同口径对照（default/predicted 与 grid 使用相同查询子集）\n")
        print("| 数据集 | 指标 | default | predicted | grid 上界 | 达上界% |")
        print("|---|---|--:|--:|--:|--:|")
        for name in names:
            r = res[name]
            if "grid_best_metrics" not in r:
                continue
            s = sam.get(name, {})
            if not s or "default_metrics" not in s:
                continue
            first = True
            for m in ("mrr@10", "ndcg@10", "recall@100"):
                d = s["default_metrics"][m]
                p = s["predicted_metrics"][m]
                g = r["grid_best_metrics"][m]
                reach = p / g * 100 if g > 0 else 0.0
                ds = name if first else ""
                first = False
                print(f"| {ds} | {m} | {d:.4f} | {p:.4f} | {g:.4f} | {reach:.1f} |")


if __name__ == "__main__":
    main()
