import re
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_readme_benchmark_table_and_summary_agree():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if (
            len(cells) == 4
            and re.fullmatch(r"[0-9]+\.[0-9]+ → [0-9]+\.[0-9]+", cells[1])
        ):
            rows.append([float(value) for cell in cells[1:] for value in cell.split(" → ")])

    assert len(rows) == 16
    assert all(row[index + 1] > row[index] for row in rows for index in (0, 2, 4))

    ndcg_gain = [(row[3] / row[2] - 1) * 100 for row in rows]
    assert round(statistics.median(ndcg_gain), 1) == 18.0
    assert round(statistics.mean(ndcg_gain), 1) == 30.5
    assert "21 dictionary features" in text


def test_chinese_is_confined_to_the_chinese_readme():
    chinese = re.compile(r"[\u3400-\u9fff]")
    checked_suffixes = {".md", ".py", ".rs", ".toml", ".yaml", ".yml", ".in"}
    offenders = []
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    for relative in tracked:
        if not relative:
            continue
        path = ROOT / relative
        if path.suffix not in checked_suffixes or path.name == "README_CN.md":
            continue
        if not path.is_file():
            continue
        if chinese.search(path.read_text(encoding="utf-8", errors="ignore")):
            offenders.append(str(path.relative_to(ROOT)))

    assert offenders == []
