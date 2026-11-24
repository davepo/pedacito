"""Bug Hunt #12 Phase 1 regression: a no-op `pass` is not a stub.

The autonomous build of `webhook_dash` produced a complete `server.py` whose
only bare `pass` was the idiomatic `BaseHTTPRequestHandler.log_message`
override. The placeholder preflight counted every `pass` line, rejected the
whole file, and the model re-sent the same correct code until the run stalled.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito.edits import Edit
from pedacito.verify import check_placeholders


def _findings(text: str):
    return check_placeholders("build the server", [Edit("server.py", "", text)])


def test_no_op_override_beside_real_code_is_not_a_stub():
    text = '''\
import http.server


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def main():
    print("serving")
'''
    assert _findings(text) == []


def test_best_effort_except_pass_is_not_a_stub():
    text = '''\
def load(path):
    try:
        return open(path).read()
    except OSError:
        return ""
'''
    assert _findings(text) == []


def test_a_definition_whose_whole_body_is_pass_is_still_a_stub():
    findings = _findings("def todo_feature():\n    pass\n")
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "placeholder" in findings[0].message


def test_a_mostly_stub_file_is_still_refused():
    text = '''\
def a():
    pass


def b():
    pass


def c():
    return 3
'''
    assert len(_findings(text)) == 1


def test_todo_and_notimplemented_still_trigger():
    assert len(_findings("def f():\n    # TODO: write this\n    return 1\n")) == 1
    assert len(_findings("def f():\n    raise NotImplementedError\n")) == 1


def test_single_top_level_pass_stub_among_real_functions_is_refused():
    """One unfinished top-level function (`def get_wave_difficulty(): pass`)
    in an otherwise implemented module must still fail the placeholder gate."""
    text = '''\
class GameError(Exception):
    pass


def a():
    return 1


def b():
    return 2


def c():
    return 3


def get_wave_difficulty(wave):
    pass
'''
    findings = _findings(text)
    assert len(findings) == 1
    assert findings[0].severity == "error"
    assert "placeholder" in findings[0].message

