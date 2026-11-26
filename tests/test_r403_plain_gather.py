"""R4-BUG-03: plain-word gather must not advertise edits and must understand
the shorthand small models actually emit.

Covers:
  * `plain_step_instructions(allow_write=False)` omits write/append and says
    edits happen after `action done`; `allow_write=True` keeps them.
  * `parse_plain_step` accepts verb-leading shorthand (`done`,
    `read a.py lines 1 to 23`, `outline a.py`), coerces edit-block replies to
    `done`, and preserves the canonical `action ...` / `key: value` shapes.
  * A supplied schema narrows which shorthand verbs are recognised.

Offline and deterministic: no model server, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pedacito import tools


# ==================================================== 1. instructions
class TestPlainStepInstructions:
    def test_read_only_instructions_omit_write_and_append(self):
        text = tools.plain_step_instructions(allow_write=False)
        assert "write" not in text.lower()
        assert "append" not in text.lower()
        assert "`content` for todo" in text

    def test_read_only_instructions_say_edits_follow_done(self):
        text = tools.plain_step_instructions(allow_write=False)
        assert "happen after `action done`" in text

    def test_write_enabled_instructions_mention_write_and_append(self):
        text = tools.plain_step_instructions(allow_write=True)
        assert "write" in text.lower()
        assert "append" in text.lower()
        assert "`content` for write/append/todo" in text

    def test_default_signature_is_backwards_compatible_and_read_only(self):
        # Existing callers pass no argument; they must not start advertising
        # edit actions in a read-only gather.
        text = tools.plain_step_instructions()
        assert "write" not in text.lower()
        assert "append" not in text.lower()


# ======================================================== 2. shorthand
class TestParsePlainStepShorthand:
    def test_bare_done(self):
        assert tools.parse_plain_step("done") == {"action": "done"}

    def test_done_with_trailing_reason(self):
        step = tools.parse_plain_step("done / ready to edit")
        assert step["action"] == "done"
        assert step["reasoning"] == "ready to edit"

    def test_read_with_slashed_range(self):
        assert tools.parse_plain_step(
            "read calcbox/ops.py / lines 1 to 23") == {
                "action": "read", "file": "calcbox/ops.py",
                "start": 1, "end": 23}

    def test_read_with_glued_range(self):
        assert tools.parse_plain_step(
            "read calcbox/ops.py lines 1 to 23") == {
                "action": "read", "file": "calcbox/ops.py",
                "start": 1, "end": 23}

    def test_outline_file(self):
        assert tools.parse_plain_step("outline calcbox/ops.py") == {
            "action": "outline", "file": "calcbox/ops.py"}

    def test_shorthand_respects_schema_enum(self):
        schema = {"properties": {"action": {
            "enum": ["read", "outline", "done"]}}}
        assert tools.parse_plain_step(
            "outline calcbox/ops.py", schema)["action"] == "outline"
        # `write` is not offered by this read-only schema, so a shorthand
        # line leading with it is not a gather step.
        assert tools.parse_plain_step("write calcbox/ops.py", schema) is None


# ====================================================== 3. edit blocks
class TestEditBlockRepliesBecomeDone:
    def test_search_block(self):
        reply = ("--- FILE: calcbox/ops.py\n"
                 "<<<<<<< SEARCH\nold line\n=======\nnew line\n"
                 ">>>>>>> REPLACE")
        step = tools.parse_plain_step(reply)
        assert step["action"] == "done"
        assert "edit blocks" in step["reasoning"]

    def test_symbol_block(self):
        reply = ("<<<<<<< SYMBOL\nold_name\n=======\nnew_name\n"
                 ">>>>>>> REPLACE")
        assert tools.parse_plain_step(reply)["action"] == "done"

    def test_after_symbol_block(self):
        reply = ("<<<<<<< AFTER_SYMBOL\ndef extra():\n    pass\n"
                 ">>>>>>> REPLACE")
        assert tools.parse_plain_step(reply)["action"] == "done"

    def test_file_header_alone(self):
        reply = "--- FILE: calcbox/ops.py\nnew file body"
        assert tools.parse_plain_step(reply)["action"] == "done"

    def test_valid_step_wins_over_edit_markers(self):
        reply = ("action read / file calcbox/ops.py / lines 1 to 23\n"
                 "<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE")
        assert tools.parse_plain_step(reply)["action"] == "read"


# ================================================= 4. canonical formats
class TestCanonicalFormatsStillParse:
    def test_slash_shape(self):
        assert tools.parse_plain_step(
            "action read / file src/main.py / lines 10 to 40") == {
                "action": "read", "file": "src/main.py",
                "start": 10, "end": 40}

    def test_key_value_shape(self):
        assert tools.parse_plain_step(
            "action: read\nfile: src/main.py\nstart: 10\nend: 40\n"
            "reasoning: need the loop") == {
                "action": "read", "file": "src/main.py",
                "start": 10, "end": 40, "reasoning": "need the loop"}

    def test_slash_and_key_value_are_equal(self):
        slash = tools.parse_plain_step(
            "action read / file src/main.py / lines 10 to 40")
        kv = tools.parse_plain_step(
            "action: read\nfile: src/main.py\nstart: 10\nend: 40")
        assert slash == kv

    def test_action_done_and_start_end(self):
        assert tools.parse_plain_step("action done") == {"action": "done"}
        assert tools.parse_plain_step(
            "action read / file a.py / start 5 / end 9") == {
                "action": "read", "file": "a.py", "start": 5, "end": 9}

    def test_grep_pattern_and_file(self):
        assert tools.parse_plain_step(
            "action grep / pattern MAX_RETRIES / file src/sync.py") == {
                "action": "grep", "pattern": "MAX_RETRIES",
                "file": "src/sync.py"}

    def test_prose_and_empty_are_not_steps(self):
        assert tools.parse_plain_step("just some prose, no action") is None
        assert tools.parse_plain_step("") is None
