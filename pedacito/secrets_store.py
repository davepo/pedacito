"""Optional OS keychain credential resolution via ``keyring``.

Core Pedacito has zero keychain dependencies. When ``keyring`` is installed
(``pip install "pedacito[keyring]"``), profiles can reference credentials via:

  * ``api_key = "keyring:<service>/<account>"`` (or ``"keyring:<account>"``
    using the default service ``"pedacito"``), or
  * ``api_key_keyring = "<service>/<account>"`` (or ``"<account>"``).

``keyring`` is always imported lazily inside the helper functions so installs
without the optional extra never pay import cost or fail at startup.
"""

from __future__ import annotations

import os

KEYRING_PREFIX = "keyring:"
DEFAULT_SERVICE = "pedacito"
INSTALL_HINT = 'Install optional OS keyring support with: pip install "pedacito[keyring]"'


class SecretStoreError(RuntimeError):
    """Raised when a keyring-backed credential cannot be resolved."""


def is_keyring_ref(value: str | None) -> bool:
    """True when ``value`` uses the ``keyring:...`` reference syntax."""
    return isinstance(value, str) and value.strip().lower().startswith(KEYRING_PREFIX)


def parse_keyring_ref(ref: str | None, *, allow_bare: bool = False) -> tuple[str, str] | None:
    """Parse ``keyring:<service>/<account>`` (or ``<service>/<account>`` when
    ``allow_bare=True``) into ``(service, account)``.

    If no ``/`` separator is present, defaults the service to ``"pedacito"``.
    Returns ``None`` when ``ref`` is empty or not a keyring reference.
    """
    raw = str(ref or "").strip()
    if not raw:
        return None
    low = raw.lower()
    if low.startswith(KEYRING_PREFIX):
        spec = raw[len(KEYRING_PREFIX):].strip()
    elif allow_bare:
        spec = raw
    else:
        return None
    if not spec:
        return None
    if "/" in spec:
        service, account = spec.split("/", 1)
        service = service.strip() or DEFAULT_SERVICE
        account = account.strip()
        return (service, account) if account else None
    return (DEFAULT_SERVICE, spec)


def _load_keyring():
    """Lazily import ``keyring`` or raise :class:`SecretStoreError` with a
    clear install hint."""
    try:
        import keyring  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SecretStoreError(
            f"A keyring credential was configured, but the `keyring` package is not installed. "
            f"{INSTALL_HINT}"
        ) from exc
    return keyring


def get_secret(service: str, account: str) -> str:
    """Read a secret from the OS keyring; returns ``""`` if not stored."""
    kr = _load_keyring()
    try:
        val = kr.get_password(service, account)
    except Exception as exc:
        raise SecretStoreError(
            f"Could not read keyring entry {service}/{account}: {exc}"
        ) from exc
    return str(val or "")


def set_secret(service: str, account: str, secret: str) -> None:
    """Store a secret in the OS keyring."""
    kr = _load_keyring()
    try:
        kr.set_password(service, account, secret)
    except Exception as exc:
        raise SecretStoreError(
            f"Could not store keyring entry {service}/{account}: {exc}"
        ) from exc


def delete_secret(service: str, account: str) -> None:
    """Delete a secret from the OS keyring (missing entries are ignored)."""
    kr = _load_keyring()
    not_found = getattr(getattr(kr, "errors", None), "PasswordDeleteError", None)
    try:
        kr.delete_password(service, account)
    except Exception as exc:
        if not_found is not None and isinstance(exc, not_found):
            return
        raise SecretStoreError(
            f"Could not delete keyring entry {service}/{account}: {exc}"
        ) from exc


def resolve_keyring_ref(ref: str, *, allow_bare: bool = False) -> str:
    """Resolve a keyring reference string to its stored secret."""
    parsed = parse_keyring_ref(ref, allow_bare=allow_bare)
    if parsed is None:
        return ""
    service, account = parsed
    return get_secret(service, account)


def resolve_api_key(
    api_key: str = "",
    api_key_env: str = "",
    api_key_keyring: str = "",
) -> str:
    """Resolve an API key from ``api_key_env``, ``api_key_keyring``, or
    ``api_key`` (including ``keyring:<service>/<account>``).

    Precedence:
      1. ``api_key_env`` (when set and non-empty in ``os.environ``; if
         ``api_key_env`` is set by itself without keyring/inline fallback,
         matches legacy ``os.environ.get(env, "")``).
      2. ``api_key_keyring``
      3. ``api_key`` (resolving ``keyring:...`` if present, else literal)
    """
    env = str(api_key_env or "").strip()
    if env:
        val = os.environ.get(env, "")
        if val or (not api_key_keyring and not api_key):
            return val
    kr_field = str(api_key_keyring or "").strip()
    if kr_field:
        return resolve_keyring_ref(kr_field, allow_bare=True)
    raw_key = str(api_key or "")
    if is_keyring_ref(raw_key):
        return resolve_keyring_ref(raw_key, allow_bare=False)
    return raw_key
