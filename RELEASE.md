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

## v0.3.1 readiness (2026-11-20)

**Status: DEFERRED.** These were readiness checks only. Work stayed on
`v0.3.1-dev`; nothing was merged, tagged, uploaded to PyPI, or pushed to
another branch. `build` and `twine` were installed into the project `.venv`
for the checks only; they are not project dependencies.

### Build blocker fixed

The first build failed because current setuptools rejects the metadata entry
`[project.urls] Contact = "mailto:contact@pedacito.dev"` ("must be url"). The
invalid URL entry was removed; `contact@pedacito.dev` remains in the README,
`docs/SECURITY.md`, and the TUI help. This is a metadata-only change with no
runtime behavior change.

### Artifacts and twine

`python -m build` (build 1.6.1) produced, beside the untouched 0.2.0 files:

| Artifact | SHA-256 |
|---|---|
| `dist/pedacito-0.3.1-py3-none-any.whl` | `dd73752dee222a6a3e7183bc59edb4746eaf67366ea074af5d408148ce5712cd` |
| `dist/pedacito-0.3.1.tar.gz` | `8165f806b0b7ce13c1184b5aad7fec7215f2666ec72030f2061016f2d9d83c95` |

`python -m twine check` (twine 7.0.0) reported **PASSED** for both the wheel
and the sdist. The sdist contains all five `docs/*.md`; the wheel installs them
under `share/pedacito/docs/` with `README.md`, `QUICKSTART.md` and
`MODEL_GUIDE.md` alongside.

> **Packaging change after these hashes.** The §7.8 follow-up added
> `TECHNIQUES.md` and `CHANGELOG.md` to `[tool.setuptools.data-files]` so the
> README's links resolve in a wheel install. The SHA-256s above predate that
> change; rebuild both 0.3.1 artifacts at release time and re-run twine before
> uploading. Do not publish the recorded files.

### Non-editable install smoke test

A clean venv at `/tmp/opencode/pdc-wheel-smoke` (Python 3.14.10) installed the
0.3.1 wheel; the package reported 0.3.1 and imported from that venv, not the
checkout.

| Check | Result |
|---|---|
| Packaged docs | All five `share/pedacito/docs/*.md` plus README/QUICKSTART/MODEL_GUIDE present; `how.find_docs_root()` resolved to the venv's `share/pedacito` |
| `pdc how --offline` | Exit 0; answered the auto-tune question ("Missing telemetry ... makes auto-tune abstain ...") |
| Docs corpus from that install | `how.build_corpus()` included every `docs/*.md`; `how.select_references("how do I install pedacito on a GPU server?")` selected `docs/INSTALLATION.md` and included "installs the full core", a docs-only phrase |
| `pdc update` | `Installed: 0.3.1    Latest: 0.2.0` → "You are up to date." |
| `pdc profile --show` | Exit 0 |
| `pdc --help` | Exit 0 |
| Update opt-out | With `PEDACITO_CHECK_FOR_UPDATES=false`, `pdc update` printed "No network call was made."; a socket audit hook recorded no DNS, connect or HTTP events (only a local loopback bind). The unset control recorded the PyPI GET and connect. |

The offline CLI answer is rendered from the built-in `how_index` cards; it is
the model-synthesis `HowSession` path that consumes the docs corpus, so the
corpus itself was proven through the installed package's `build_corpus()` and
`select_references()` as recorded above.

### Known deferred state

- PyPI still serves 0.2.0. Until 0.3.1 is published, `pdc update` on 0.3.1
  correctly reports "up to date" against Latest: 0.2.0. This is expected for
  the deferred release, not a smoke-test failure.
- `pdc --version` is not implemented (the top-level parser exits 2); the
  verification commands in steps 10 and 11 above should use another version
  check unless that flag is added.
- The throwaway venv and its temporary config/log files were removed after
  the checks.
