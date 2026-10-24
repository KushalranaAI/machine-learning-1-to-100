"""Content checks: every lesson and project README has valid frontmatter that the web app can render."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LESSONS = sorted(ROOT.glob("curriculum/*/*/README.md"))
PROJECTS = sorted(ROOT.glob("projects/*/README.md"))
LESSON_IDS = {f"{p.parent.parent.name}/{p.parent.name}" for p in LESSONS}
PROJECT_IDS = {p.parent.name for p in PROJECTS}


def frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\n"), f"{path} has no frontmatter"
    return yaml.safe_load(text.split("---\n", 2)[1])


@pytest.mark.parametrize("path", LESSONS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_lesson_frontmatter(path):
    meta = frontmatter(path)
    for key in ("title", "summary", "why", "background", "use_cases", "workflow"):
        assert meta.get(key), f"{path}: missing {key}"
    for pre in meta.get("prerequisites") or []:
        assert pre in LESSON_IDS, f"{path}: unknown prerequisite {pre}"
    for proj in meta.get("where_used") or []:
        assert proj in PROJECT_IDS, f"{path}: unknown project {proj}"
    for spec in meta.get("code") or []:
        file = spec.split(":")[0]
        assert (ROOT / file).exists(), f"{path}: code file {file} not found"
    demo = meta.get("demo")
    if demo:
        assert (path.parent / demo).exists(), f"{path}: demo {demo} not found"


@pytest.mark.parametrize("path", PROJECTS, ids=lambda p: p.parent.name)
def test_project_frontmatter(path):
    meta = frontmatter(path)
    for key in ("title", "level", "summary", "problem", "dataset", "steps", "workflow"):
        assert meta.get(key), f"{path}: missing {key}"
    for lesson in meta["lessons"]:
        assert lesson in LESSON_IDS, f"{path}: unknown lesson {lesson}"
    for step in meta["steps"]:
        if step.get("script"):
            assert (path.parent / step["script"].split()[0]).exists(), f"{path}: missing script {step['script']}"


STRING_LISTS = ("use_cases", "where_used", "prerequisites", "pros", "cons", "skills", "lessons", "algorithms", "widgets", "code")


@pytest.mark.parametrize("path", LESSONS + PROJECTS, ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_frontmatter_lists_are_plain_text(path):
    """An unquoted item like `- A fair protocol: train / test` is parsed by YAML as a mapping and shows up as
    "[object Object]" in the web app. Quote such items."""
    meta = frontmatter(path)
    for key in STRING_LISTS:
        for item in meta.get(key) or []:
            assert isinstance(item, str), f"{path}: an item of '{key}' is not plain text ({item!r}); put it in quotes"
