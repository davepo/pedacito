🌐 [English](../SKILLS_AND_TOOLS.md) | **Español**

# Skills y herramientas

Esta guía cubre las formas de ampliar lo que Pedacito sabe y puede hacer: las 19 skills incluidas,
las skills personalizadas, los servidores MCP con nombre, los hooks JSON y la sobrescritura de prompts. Los ajustes viven en
[CONFIG.md](CONFIG.md); el modelo de confianza detrás de ellos está en [SECURITY.md](SECURITY.md).

## Contenido

- [Skills incluidas](#skills-incluidas)
- [Skills personalizadas](#skills-personalizadas)
- [Servidores MCP con nombre](#servidores-mcp-con-nombre)
- [Hooks JSON](#hooks-json)
- [Acción de generación de imágenes](#acción-de-generación-de-imágenes)
- [Formatos de recolección y observaciones grandes (v0.3.3)](#formatos-de-recolección-y-observaciones-grandes-v033)
- [Manifiestos de paquete (`pedacito-package.toml`)](#manifiestos-de-paquete-pedacito-packagetoml)
- [Sobrescritura de prompts](#sobrescritura-de-prompts)

## Skills incluidas

Las skills son listas de verificación cortas en markdown. El modelo ve un catálogo de una línea y carga un cuerpo solo
cuando lo necesita; las dos mejores coincidencias de disparadores para una tarea también se inyectan de antemano. Las skills
son texto de prompt: nada en una skill se ejecuta jamás. `pedacito skills list` muestra cada skill
con una insignia de alcance (bundled, user o project).

| Skill | Para qué sirve | Disparadores de ejemplo |
|---|---|---|
| `add-a-test` | Agregar una prueba enfocada para una función que no tiene ninguna | test, tests, coverage, untested, pytest |
| `fix-a-failing-test` | Diagnosticar y corregir una prueba que falla | failing test, failed test, test failure, broken test, assertionerror |
| `refactor-safely` | Renombrar o reestructurar código sin romper a los llamadores | refactor, restructure, signature, callers, extract |
| `review-my-diff` | Revisar los cambios actuales sin confirmar en busca de defectos | review, diff, uncommitted, defect, audit |
| `find-dead-code` | Comprobar si un archivo o función no se usa | dead code, unused, unreferenced, orphan |
| `surgical-symbol-edit` | Reemplazar o insertar funciones y métodos en archivos grandes usando empalmes SYMBOL de cero anclas o rangos LINES sin repetir el código antiguo. | symbol, splice, large file, outline, lines range |
| `frontend-dom-audio-debug` | Verificar y corregir interfaces web, contratos de módulos HTML/JS, renderizado en canvas y fallos silenciosos de Web Audio. | dom, getelementbyid, web audio, audiocontext, es module |
| `safe-multi-file-rename` | Renombrar una función, clase, método o variable en todo el proyecto de forma determinista sin ediciones manuales en varios archivos. | rename, multi-file, usages, identifier |
| `fastapi-rest-endpoint` | Agregar o corregir un endpoint REST de FastAPI con validación de Pydantic y pruebas | fastapi, pydantic, apireouter, testclient, endpoint |
| `canvas-svg-2d-3d` | Depurar y construir renderizado y bucles de animación con canvas/SVG/WebGL | canvas, svg, viewbox, webgl, three.js |
| `sqlite-sql-migrations` | Escribir migraciones de esquema de SQLite seguras y consultas SQL correctas | sqlite, sqlite3, sql, schema, migration |
| `async-concurrency-safety` | Corregir condiciones de carrera, interbloqueos y manejo de subprocesos en asyncio/threading | asyncio, async, await, threading, lock |
| `security-ssrf-xss-hardening` | Endurecer el código contra SSRF, XSS, CSRF, inyección y recorrido de rutas | ssrf, xss, csrf, sanitize, path traversal |
| `i18n-localization-catalog` | Externalizar cadenas de la interfaz y mantener sincronizados los catálogos de traducción | i18n, locale, translate, translation, spanish |
| `parser-lexer-state-machine` | Construir lexers, parsers, AST y máquinas de estados que no puedan entrar en bucle | lexer, tokenizer, parser, ast, rpn |
| `performance-memory-profiling` | Encontrar y corregir cuellos de botella, cachés sin límite y fugas de memoria | lru, ttl, cache, evict, complexity |
| `modern-web-design-ui` | Diseñar sitios web modernos, landing pages y páginas de bio/portafolio con CSS de alta factura | website, webpage, landing page, personal site, bio |
| `desktop-gui-and-games` | Construir GUIs de escritorio y juegos 2D que no instalen nada y nunca cuelguen las pruebas headless | calculator, tkinter, pyqt, pyside, gui |
| `safe-git-commit` | Inspeccionar los cambios y preparar un commit seguro que el usuario pueda ejecutar | commit, git, stage, push, message |

Las skills de proyecto anulan a las del usuario, que a su vez anulan a la biblioteca incluida.

## Skills personalizadas

Crea una desde la línea de comandos:

```bash
pedacito skills add NAME --description "..." --triggers "k1,k2" --body "1. ..."   # this project
pedacito skills add NAME --global ...                                              # your user library
pedacito skills draft "Skill idea in plain words"    # the model drafts it for you to review
pedacito skills list | show NAME | rm NAME
```

Los archivos se guardan como `.pedacito/skills/NAME.md` (proyecto) o en la carpeta `skills/` junto a tu
configuración (usuario), con frontmatter YAML (`name`, `description` y, opcionalmente, `triggers`).

En la GUI abre **Project ▾ → Skills library & creator…**, completa el formulario o presiona
**✦ Auto-Draft from Description**. Para forzar una skill en la siguiente tarea sin esperar una
coincidencia de disparador, usa **Use in next task** en ese diálogo o, en la TUI, `/skill NAME`. Una skill
fijada acompaña a cada tarea como `skill:NAME` hasta que ejecutes `/pin remove skill:NAME` o `/pin clear`.
Detalles: [USAGE.md](USAGE.md#skills-catálogo-bajo-demanda-e-inyección-manual).

## Servidores MCP con nombre

Declara servidores como tablas `[mcp_servers.<name>]` y el modelo llega a ellos mediante una sola
acción `mcp_call`:

```toml
enable_mcp = true

[mcp_servers.docs]
command = "python"
args = ["-m", "docs_server"]
trusted = false          # true: calls run without asking

[mcp_servers.jira]
url = "http://127.0.0.1:8000/mcp"
```

Las herramientas se podan por tarea (como máximo cuatro firmas coincidentes llegan al prompt) y cada llamada
está sujeta a permiso a menos que el servidor sea `trusted`. Los archivos `.pedacito.toml` del proyecto no pueden
definir servidores hasta que ejecutes `pedacito trust`. Adminístralos en **⚙ Settings → MCP Servers** o
revisa la vista podada con `pedacito mcp list`. Referencia completa:
[CONFIG.md](CONFIG.md#servidores-mcp-con-nombre).

## Hooks JSON

Los hooks ejecutan tu propio argv (nunca un shell) en eventos como `pre_task`, `post_edit`, `pre_run`
o `post_compact`. Un hook recibe un objeto JSON por stdin y puede responder con
`{"decision": "allow" | "deny", "message": "...", "context": "..."}`; un hook `pre_*` también puede
denegar con el código de salida 2. Los hooks a nivel de proyecto solo se ejecutan después de `pedacito trust`.

```toml
hooks = { post_edit = ["ruff format {file}"], pre_edit = ["./scripts/guard.sh"] }
hook_timeout = 10
```

Consulta [CONFIG.md](CONFIG.md#hooks) para ver la lista completa de eventos y el protocolo.

## Acción de generación de imágenes

`generate_image` es una acción opcional del agente: `prompt` (qué dibujar), `size` opcional (como `1024x1024`) y `file`
(un destino relativo al espacio de trabajo; por defecto `.pedacito/images/`). Aparece en la lista de herramientas del modelo solo cuando hay un
perfil de imagen configurado (`image_profile`, o un perfil `modality = "image"` autodetectado) **y** o bien
`allow_image_gen = true` o bien un intermediario de permisos en vivo (GUI/TUI) puede aprobar cada llamada. Sin perfil de imagen, el esquema de
herramientas no cambia. La escritura usa la misma contención del espacio de trabajo y escritura atómica que cualquier otra escritura de archivos.
Consulta [Generación local de imágenes](CONFIG.md#generación-local-de-imágenes-image_profile-allow_image_gen).

## Formatos de recolección y observaciones grandes (v0.3.3)

El bucle de recolección puede pedir cada paso de búsqueda en uno de tres formatos, según `tool_mode`:

- `grammar` (por defecto): un paso JSON, restringido por el servidor cuando `enforce_json_schema` está activo.
- `native`: el propio canal de llamadas a herramientas del modelo.
- `plain`: palabras simples y breves, por ejemplo `read app.py 1 40` o `grep parse_args`. Conviene a modelos
  pequeños que tropiezan con JSON. `calibrate` lo compara A/B con `grammar` y conserva el formato que
  puntúa mejor.

Cualquier observación por encima de `max_observation_tokens` de `run`, `run_test`, `grep`, `diff`, `git_diff`,
`fetch`, `web_search`, `diagnose` o `serve` se vuelca a `.pedacito/scratch/<id>.log`. El modelo recibe el
inicio, el final, el número de líneas y el nombre del archivo. Después puede usar `read` o `grep` sobre
líneas exactas de ese archivo, y solo sobre archivos `*.log` directos bajo `.pedacito/scratch/`. El modelo
nunca puede escribir en ese directorio. Consulta [Seguridad](SECURITY.md#commits-pull-requests-undo-y-archivos-scratch-v033).

## Manifiestos de paquete (`pedacito-package.toml`)

Cada extensión, skill, conjunto de comandos o paquete de modelos distribuible se declara en un
archivo `pedacito-package.toml` en la raíz del paquete:

```toml
schema_version = 1
name = "acme-lint"
version = "1.2.0"
kind = "extension"          # extension | skill | command | model-pack
description = "Reglas de lint de Acme para Pedacito"
capabilities = ["read_workspace", "run_commands"]
min_pedacito_version = "0.4.0"

[entry_points]
main = "acme_lint/plugin.py"
```

`schema_version` debe ser `1`. `name` acepta letras, dígitos, `.`, `_` y `-`.
`capabilities` solo puede usar `read_workspace`, `write_workspace`, `run_commands`,
`network`, `mcp_server`, `hooks`, `prompts` y `model_profiles`. Cada valor de
`entry_points` debe ser una ruta relativa dentro del paquete: las rutas absolutas
y el recorrido `..` se rechazan antes de cargar nada.

El manifiesto es declarativo; por sí mismo nunca ejecuta código. `load_manifest`
lo valida y calcula el SHA-256 de los bytes originales. `trust_manifest` fija
ese hash en `trusted_packages.json` dentro de la carpeta de estado del usuario, e
`is_manifest_trusted` devuelve `False` en cuanto el archivo cambia por un solo
byte. Un paquete no confiable o malformado se rechaza antes de que pueda
ejecutarse cualquier punto de entrada.

## Sobrescritura de prompts

Cualquier sección de prompt integrada puede reemplazarse por completo con un archivo markdown que lleve su nombre:
`.pedacito/prompts/*.md` en el proyecto, y luego `prompts/*.md` junto a tu configuración de usuario
(por ejemplo `gather.md`, `edit_format.md`, `answer.md`). Una sobrescritura omite la guía integrada de la
sección, así que conserva las instrucciones de formato en las que se apoya el motor. Consulta
[ARCHITECTURE.md](ARCHITECTURE.md) para ver cómo se ensambla el prompt.
