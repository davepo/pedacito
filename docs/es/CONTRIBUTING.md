🌐 [English](../CONTRIBUTING.md) | **Español**

# Contribuir

Gracias por ayudar. Los issues y pull requests son bienvenidos; por favor abre un issue antes de un PR
grande. Reporta los problemas de seguridad de forma privada, como se describe en
[SECURITY.md](SECURITY.md#reportar-una-vulnerabilidad).

## Entorno de desarrollo

```bash
git clone https://github.com/davepo/pedacito && cd pedacito
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
pytest -q                       # the full suite
pytest -q tests/test_doc_sync.py tests/test_release_hygiene.py   # docs and release guards
ruff check pedacito tests       # lint
```

Se requiere Python 3.11 o superior. Las pruebas no deben necesitar un servidor de modelos en ejecución.

## Mapa del código

| Ruta | Función |
|---|---|
| `pedacito/cli.py` | Analizador de argumentos y cada subcomando de `pedacito` |
| `pedacito/config.py` | El dataclass `Config`, valores por defecto, carga de TOML, preajustes |
| `pedacito/index.py`, `chunker*.py` | Análisis del proyecto en unidades con rangos de línea exactos |
| `pedacito/agent.py`, `loop.py`, `tools.py` | Ciclo de recolección, acciones y fase de respuesta |
| `pedacito/edits.py`, `verify.py` | Bloques de edición, reparación y verificación |
| `pedacito/llm.py`, `providers.py`, `servers.py` | Clientes de modelos, tabla de proveedores alojados, ciclo de vida de servidores |
| `pedacito/security.py`, `safety.py`, `sandbox.py` | Escaneo de secretos, redacción, sandbox del sistema operativo |
| `pedacito/skills.py`, `mcp.py`, `hooks.py`, `prompts.py` | Skills, puente MCP, hooks, sobrescritura de prompts |
| `pedacito/tui.py`, `pedacito/gui/` | Interfaz de terminal y panel de control en el navegador |
| `pedacito/i18n.py`, `pedacito/locales/` | Motor de localización y catálogos |
| `pedacito/how_index.py`, `how.py` | El índice con búsqueda de `pedacito how` |
| `tests/` | La suite de pruebas, incluidas las protecciones contra desvío de la documentación |

## Propiedad de la documentación

- `docs/` es la **referencia** canónica de ajustes, comandos y garantías.
- Las guías de la raíz ([QUICKSTART.es.md](../../QUICKSTART.es.md), [MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md),
  [TECHNIQUES.es.md](../../TECHNIQUES.es.md)) son los **tutoriales** canónicos.
- [README.es.md](../../README.es.md) es el índice y [CHANGELOG.md](../../CHANGELOG.md) es el historial.
- Enlaza entre los conjuntos en lugar de copiar texto. El párrafo `**Propiedad de la documentación.**` del README
  y de [ARCHITECTURE.md](ARCHITECTURE.md) debe permanecer idéntico.
- Un ajuste nuevo va en [CONFIG.md](CONFIG.md) con su valor por defecto real, un subcomando nuevo en
  la tabla de [USAGE.md](USAGE.md#la-cli-de-un-vistazo) y en `pedacito/how_index.py`.
  `tests/test_doc_sync.py` falla cuando se desvían.

## Agregar o mantener un idioma

1. Copia `pedacito/locales/en.json` a `pedacito/locales/<code>.json` y traduce los valores;
   conserva cada clave y cada `{placeholder}`.
2. Registra el código en `i18n.SUPPORTED_LANGUAGES` y en el enum `language` de `config.py`.
3. Traduce la documentación a `docs/<code>/` (las guías de la raíz pasan a ser `README.<code>.md`,
   `QUICKSTART.<code>.md`, y así sucesivamente) y agrega la barra selectora de idioma correspondiente en la parte superior
   de cada página en inglés.
4. Lista los archivos nuevos en `[tool.setuptools.data-files]` de `pyproject.toml`.
5. Ejecuta `pytest -q`; las protecciones de idioma y documentación deben seguir en verde.

El catálogo y la documentación en español (`pedacito/locales/es.json`, `docs/es/`) son el ejemplo
de referencia.
