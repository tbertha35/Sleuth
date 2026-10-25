"""Checks on the test suites themselves: a wrong "expected" answer would
quietly make every model look worse, so verify what can be verified."""

import contextlib
import io

from sleuth.analyzer.suites import load_suite


def test_knowledge_suite_is_big_enough_and_well_formed():
    cases = load_suite("suites/knowledge")
    # 40 items moved exact match in steps of 2.5%; ~200 gives ~0.5% steps and
    # a standard error small enough to tell neighbouring models apart.
    assert len(cases) >= 200
    assert len({c.id for c in cases}) == len(cases)
    assert len({c.prompt for c in cases}) == len(cases)
    for c in cases:
        assert c.expected, c.id
        if c.prompt.startswith("Q: "):
            # Answers follow "A:" with a space, the way the text continues in web data.
            assert c.prompt.endswith("\nA:") and c.expected.startswith(" ") and not c.expected.endswith(" "), c.id


def test_python_print_answers_are_what_python_prints():
    checked = 0
    for c in load_suite("suites/knowledge/python.jsonl"):
        if not c.prompt.endswith("# prints"):
            continue  # a code completion: nothing to run
        code = c.prompt[: -len("# prints")]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(compile(code, c.id, "exec"), {})  # noqa: S102 - our own suite file
        assert out.getvalue().strip() == c.expected.strip(), c.id
        checked += 1
    assert checked >= 30
