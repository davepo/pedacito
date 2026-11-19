# Releasing Pedacito

This is the checklist for cutting a Pedacito (`pdc`) release. It covers the
two-file version contract, the pre-flight gates, and the merge → tag → build →
publish sequence.

## v0.3.1 is DEFERRED

**All v0.3.1 work lands on `v0.3.1-dev`. Do not merge, tag, or publish v0.3.1
now.** The merge-to-`main`, tag, build and publish steps below are the *future*
procedure. Run them only when the release is explicitly approved.

Current state (2026-11-18):

| Item | State |
|---|---|
| Working branch | `v0.3.1-dev` (12 commits ahead of `main`, 0 behind) |
| Version in the tree | `0.3.1` in `pyproject.toml` and `pedacito/__init__.py` |
| `dist/` | Only **0.2.0** artifacts: `pedacito-0.2.0-py3-none-any.whl`, `pedacito-0.2.0.tar.gz` |
| PyPI (`pedacito`) | Latest is **0.2.0**; it is the only published release |
| Git tags | Only `v0.2.0` (plus a local `reviewed` tag); no v0.3.x tag |

Consequence: the v0.3.1 auto-update feature cannot fire in the wild until a
0.3.x release is actually published. Anyone running `pip install pedacito`
today gets v0.2.0, which predates all of v0.3.0 and v0.3.1.

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

For v0.3.1 this means an explicit decision to end the deferred period. Until
then, stop here and keep landing work on `v0.3.1-dev`.

### 2. Bump the version in both files

```bash
# Edit both, in one commit:
#   pyproject.toml            -> [project] version = "X.Y.Z"
#   pedacito/__init__.py      -> __version__ = "X.Y.Z"
git diff -- pyproject.toml pedacito/__init__.py
```

Do not hand-edit a built artifact. The build reads `pyproject.toml`; the
runtime `pdc --version` and the update checker read `__version__`.

### 3. Run the full suite and ruff

```bash
python -m pytest -q
python -m ruff check pedacito/ tests/
```

Both must be clean. The full suite takes several minutes (roughly 3,350 tests
at the v0.3.1 cut) and includes the drift guard. To iterate on the guard alone:

```bash
python -m pytest -q tests/test_release_hygiene.py
```

### 4. Update CHANGELOG.md

Add a heading for the exact version, in the existing style:

```markdown
## vX.Y.Z — YYYY-MM-DD
```

Then list the notable changes below it. The drift-guard test fails when this
heading is missing, so this is not optional paperwork. Commit the version bump
and changelog together.

### 5. Merge `v0.3.1-dev` → `main`

CI (`.github/workflows/ci.yml`) runs on **pushes to `main` and on pull
requests only** — it does not run on pushes to `v0.3.1-dev`. The merge is the
first full CI run on the release candidate.

```bash
git fetch origin
git switch v0.3.1-dev
git status                     # must be clean
git switch main
git pull --ff-only origin main
git merge --no-ff v0.3.1-dev
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

Check `git tag --list` first and never move an existing tag. (At the time of
writing only `v0.2.0` is tagged; v0.3.0 is in the changelog but was never
tagged or published.)

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

At the time of writing `dist/` holds only 0.2.0 artifacts. Uploading those by
mistake would republish an old version (and PyPI rejects duplicate filenames),
which is why the directory is cleared before building.

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

Do this in a clean venv, not the checkout's editable environment. The only
real prior release on PyPI is 0.2.0, and it predates the `pdc update` command,
so verify the upgrade path and the new checker in sequence:

```bash
# Real prior-version install (clean venv).
python -m venv /tmp/pdc-prior
source /tmp/pdc-prior/bin/activate          # Windows: /tmp/pdc-prior\Scripts\activate
python -m pip install --upgrade pip
python -m pip install "pedacito==0.2.0"
pdc --version                               # 0.2.0

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
exists) can only be exercised by a release that already contains `pdc update`.
Once 0.3.1 is published, the next release's verification starts from a clean
`pip install pedacito==0.3.1` and runs `pdc update`; it must report the new
version and the correct upgrade command for the install method. A fresh
`XDG_CONFIG_HOME` avoids the 24-hour `update_check.json` cache when testing.

### 11. Finish the release

- Publish the GitHub release for the tag, with the CHANGELOG section as notes.
- Confirm `pip install pedacito` and `pdc --version` in a fresh venv resolve to
  the new version.
- If the docs claim the new version is on PyPI, re-check that wording after the
  upload (the install guide currently names 0.3.1 while PyPI still serves
  0.2.0).
