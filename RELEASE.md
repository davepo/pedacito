# Releasing Pedacito

This is the checklist for cutting a Pedacito (`pdc`) release. It covers the
two-file version contract, the pre-flight gates, and the merge → tag → build →
publish sequence.

## Current release: v0.3.2

| Item | State |
|---|---|
| Release branch | `v032/job9-release-gate`, a worktree off `v0.3.2-dev`, which holds all eight merged v0.3.2 implementation jobs |
| Version in the tree | `0.3.2` in `pyproject.toml` and `pedacito/__init__.py` |
| CHANGELOG | `## v0.3.2 — 2026-11-22` lists every v0.3.2 feature and fix |
| Merged to `main`, tagged, built for upload, or published | No. Each of these happens only after the release is approved (steps 1, 5, 6 and 9 below) |

Check PyPI and `git tag --list` before quoting what is published. This file
does not track the live state of either.

## The version lives in exactly two places

The version is duplicated today, and the two copies **must be bumped together
in the same commit**:

1. `pyproject.toml` — `[project] version`
2. `pedacito/__init__.py` — `__version__`

`tests/test_release_hygiene.py` is the drift guard: it parses both values,
fails if they differ, and fails if `CHANGELOG.md` has no `## v<version>`
heading for that version. Run it before every release commit.

## Release checklist

### 1. Confirm the release is approved

Do not merge, tag, build for upload or publish until the release is explicitly
approved. Until then, keep landing work on the release branch.

### 2. Bump the version in both files

```bash
# Edit both, in one commit:
#   pyproject.toml            -> [project] version = "X.Y.Z"
#   pedacito/__init__.py      -> __version__ = "X.Y.Z"
git diff -- pyproject.toml pedacito/__init__.py
```

Do not hand-edit a built artifact. The build reads `pyproject.toml`; the
runtime `pdc --version` and the update checker read `__version__`.

### 3. Run the gates

```bash
python -m ruff check pedacito/ tests/
python -m pytest -q tests/test_release_hygiene.py tests/test_i18n_and_docs_integrity.py tests/test_doc_sync.py
python -m pytest -q
```

All three must pass. The full suite takes several minutes (about 4,200 tests
at the v0.3.2 cut) and includes the drift guard. Run the
guard alone while iterating:

```bash
python -m pytest -q tests/test_release_hygiene.py
```

The i18n and docs guards check that every `Config` field is documented in
`docs/CONFIG.md` and `docs/es/CONFIG.md`, that every EN/ES mirror pair links to
each other, that every link and anchor resolves, and that `pedacito how` covers
each CLI subcommand.

### 4. Update CHANGELOG.md

Add a heading for the exact version, in the existing style:

```markdown
## vX.Y.Z — YYYY-MM-DD
```

Then list the notable changes below it. The drift-guard test fails when this
heading is missing, so this is not optional paperwork. Commit the version bump
and changelog together.

### 5. Merge the release branch into `main`

CI (`.github/workflows/ci.yml`) runs on **pushes to `main` and on pull
requests only**. It does not run on pushes to a release branch, so the merge is
the first full CI run on the release candidate.

```bash
git fetch origin
git status                     # must be clean
git switch main
git pull --ff-only origin main
git merge --no-ff <release-branch>
git push origin main
```

Wait for the CI matrix (Linux 3.11–3.14, Windows and macOS on 3.12) to pass
before tagging.

### 6. Tag the release

Tag the commit that CI just validated on `main`:

```bash
git switch main
git pull --ff-only origin main
git tag -a vX.Y.Z -m "pedacito vX.Y.Z"
git push origin vX.Y.Z
```

Check `git tag --list` first and never move an existing tag.

### 7. Build the sdist and wheel

Build from the tagged `main` checkout, and clear the stale artifacts first:

```bash
rm -rf dist build
python -m pip install --upgrade build twine
python -m build
ls -l dist/
```

`dist/` must contain exactly the new version in both forms:

- `pedacito-X.Y.Z.tar.gz` (sdist)
- `pedacito-X.Y.Z-py3-none-any.whl` (wheel)

Clearing `dist/` first matters: PyPI rejects duplicate filenames, and an old
artifact uploaded by mistake would republish a previous version.

Before uploading, confirm the wheel bundles the package data it needs (the
locales, the GUI assets, and the v0.3.2 modules):

```bash
python -m zipfile -l dist/pedacito-X.Y.Z-py3-none-any.whl | grep -E \
  "locales/(en|es)\.json|gui/static/|interleaved\.py|workspace_overlay\.py|transport_(gemini|openai)\.py|model_catalog\.py|secrets_store\.py"
```

### 8. Check the artifacts with twine

```bash
python -m twine check dist/*
```

Both files must report `PASSED`. A failure here stops the release: fix the
metadata and rebuild; never upload an unchecked artifact.

### 9. Publish to PyPI

```bash
python -m twine upload dist/*
```

Use a PyPI API token (`TWINE_USERNAME=__token__`, password = the token). This
is irreversible: PyPI does not allow replacing a published file, and a version
number cannot be reused after upload. Upload only the `dist/*` built in step 7.
If you want a rehearsal, use TestPyPI first with
`python -m twine upload --repository testpypi dist/*`.

### 10. Verify `pdc update` from a real prior-version install

Do this in a clean venv, not the checkout's editable environment. Start from
the last version actually published on PyPI, and verify the upgrade path and
the update checker in sequence:

```bash
# Real prior-version install (clean venv).
python -m venv /tmp/pdc-prior
source /tmp/pdc-prior/bin/activate          # Windows: /tmp/pdc-prior\Scripts\activate
python -m pip install --upgrade pip
python -m pip install "pedacito==<last-published>"
pdc --version                               # <last-published>

# Upgrade it the way a user would, then check the new command.
python -m pip install --upgrade pedacito
pdc --version                               # X.Y.Z
XDG_CONFIG_HOME=/tmp/pdc-prior-config pdc update
# Expected: installed X.Y.Z, latest X.Y.Z, "You are up to date"
```

Also confirm PyPI itself serves the new version:

```bash
python - <<'PY'
import json, urllib.request
info = json.load(urllib.request.urlopen("https://pypi.org/pypi/pedacito/json"))["info"]
print(info["version"])
PY
```

The update-available branch (an older install being told a newer version
exists) needs a prior release that already contains `pdc update`. A fresh
`XDG_CONFIG_HOME` avoids the 24-hour `update_check.json` cache when testing.

### 11. Finish the release

- Publish the GitHub release for the tag, with the CHANGELOG section as notes.
- Confirm `pip install pedacito` and `pdc --version` in a fresh venv resolve to
  the new version.
- If the docs name the new version as the PyPI install (the `Quick install from
  PyPI` line in `docs/INSTALLATION.md` and `docs/es/INSTALLATION.md`), re-check
  that wording after the upload.
