🌐 [English](../CONFIG.md) | **Español**

# Configuración

Pedacito se configura con archivos TOML, un pequeño conjunto de archivos de proyecto y variables
de entorno. Esta guía cubre dónde vive cada ajuste, el archivo de reglas del proyecto, los tokens de esquemas de
color, los perfiles de modelo, la selección de idioma y la referencia completa de ajustes.

## Dónde viven los ajustes

| Ubicación | Alcance | Qué contiene |
|---|---|---|
| `~/.config/pedacito/config.toml` (Linux y macOS), o `%APPDATA%\pedacito\config.toml` (Windows) | Tú, para cada proyecto | Servidores, perfiles, claves de API, valores por defecto. `XDG_CONFIG_HOME` reemplaza a `~/.config`. |
| `<project>/.pedacito.toml` | Un proyecto | Ajustes por proyecto. Los ajustes de ejecución se ignoran hasta que ejecutes `pedacito trust`. |
| `<project>/.pedacito/rules.md` | Un proyecto | Requisitos estrictos que aparecen en cada prompt. Consulta [Reglas del proyecto](#reglas-del-proyecto-pedacitorulesmd). |
| `<project>/.pedacito/config.json` | Un proyecto | Anclaje opcional de tema, como `{"theme": "nord"}`. |
| `themes.json`, junto a la configuración de usuario | Tú | Esquemas de color personalizados y el tema seleccionado. Consulta [Esquemas de color](#esquemas-de-color). |
| `PEDACITO.md`, `AGENTS.md`, `CLAUDE.md`, `.pedacito/instructions.md` | Un proyecto | Convenciones que se leen en cada prompt. Consulta [Instrucciones del proyecto](#instrucciones-del-proyecto). |

Los ajustes se resuelven en capas. Cada capa anula a la anterior: valores por defecto integrados, luego la
configuración de usuario, luego el `.pedacito.toml` del proyecto, luego `--config FILE`, luego `--profile` y por último
las opciones de línea de comandos. Las variables de entorno `PEDACITO_<KEY>` (por ejemplo `PEDACITO_MODEL`) y
`PEDACITO_PROFILE` también se respetan. Las claves desconocidas se rechazan con una sugerencia.

Para ver el valor de cada ajuste y de dónde provino, ejecuta `pedacito config`. Para imprimir la
plantilla anotada, ejecuta `pedacito config --reference`.

## Configurar

```bash
pedacito config --setup     # finds local servers, writes a profile per model
pedacito models scan        # re-run later to add models or servers
pedacito profiles           # see scores and role recommendations
pedacito config --init      # or write a commented starter file
pedacito config             # show every resolved setting and where it came from
pedacito config --reference # print the full annotated template
pedacito config --upgrade   # add newly documented options, commented out
pedacito config --upgrade --dry-run  # preview what would be added
```

`config --upgrade` agrega las opciones que un Pedacito más nuevo documenta pero que tu
archivo no tiene, comentadas, conservando cada ajuste y comentario existente;
`--dry-run` las lista sin escribir.

`config --setup` (y `models scan`) encuentra servidores de modelos locales, LM Studio,
Ollama, Lemonade, llama.cpp, **en cualquier puerto**, identifica cada uno por sus respuestas
en lugar de por su puerto (de modo que dos servidores puedan compartir una máquina), lista sus modelos
y escribe un perfil solo de identidad por modelo, y luego define uno predeterminado. Pasa
`--url` para apuntar a un servidor en otra máquina o a una API alojada; `models scan
--match PATTERN --calibrate quick|full|none` lo vuelve a ejecutar. ¿Prefieres editar a
mano? El archivo es `~/.config/pedacito/config.toml` (Linux/macOS) o
`%APPDATA%\pedacito\config.toml` (Windows); un `XDG_CONFIG_HOME` definido reemplaza a
`~/.config`. Lo mínimo:

```toml
[universal]
vram_gb = 8                             # GPU VRAM on the model server, in GB

[profiles.default]
base_url = "http://localhost:1234/v1"
model    = "your-model-id"              # exact id/name the server reports
```

`vram_gb` es obligatorio para servidores locales (junto con el tamaño del archivo del modelo dimensiona
la ventana de contexto) y no se necesita para APIs alojadas.

Los ajustes se resuelven en capas, cada una anulando a la anterior: valores por defecto integrados →
configuración de usuario → el `.pedacito.toml` del proyecto → `--config FILE` → `--profile` →
opciones de línea de comandos. Las variables de entorno `PEDACITO_<KEY>` (por ejemplo
`PEDACITO_MODEL`) y `PEDACITO_PROFILE` también se respetan. Las claves desconocidas se
rechazan con una sugerencia. La sección heredada `[lmstudio]` todavía se acepta como
alias de `[server]`.

## Reglas del proyecto: `.pedacito/rules.md`

`.pedacito/rules.md` es una lista corta y persistente de requisitos para un proyecto. Pedacito
la inserta en cada tarea como **requisitos estrictos**, de modo que el modelo los ve lea o no
los archivos afectados. Úsala para cosas que siempre deben cumplirse, como "all HTTP calls go
through `client.py`" o "never add a dependency without asking".

```markdown
- Use `pytest -q` for tests. Do not change `tests/conftest.py` fixtures.
- Public functions in `pedacito/api/` keep their signatures.
- Do not write to `legacy/`.
```

### Cómo se usa

- El texto se agrega al prompt como `PROJECT RULES (from .pedacito/rules.md; these are hard
  requirements for every task)`.
- El texto está limitado a **6,000 caracteres**. Un archivo más largo se corta con una nota que apunta a
  el archivo.
- Se guarda en caché por fecha de modificación del archivo, de modo que editarlo surte efecto en la siguiente tarea.

### Edítala

- **GUI:** abre el editor de reglas del proyecto, que guarda el archivo.
- **TUI:** `/rules` muestra las reglas actuales. `/rules set TEXT` las reemplaza y `/rules clear`
  vacía el archivo.
- **Cualquier editor:** el archivo es Markdown simple.

### Compártela con tu equipo

Pedacito agrega `.pedacito/` al `.gitignore` del proyecto en la primera ejecución, de modo que `rules.md` se ignora
por defecto. Para confirmarlo en el repositorio, reemplaza la línea `.pedacito/` con estas dos líneas:

```gitignore
.pedacito/*
!.pedacito/rules.md
```

### Las reglas comparadas con otros archivos de instrucciones

| Archivo | Se muestra cuando | Úsalo para |
|---|---|---|
| `.pedacito/rules.md` | Cada tarea, como requisitos estrictos | Invariantes que siempre deben cumplirse |
| `PEDACITO.md`, `AGENTS.md`, `CLAUDE.md` | Cada tarea, limitado a 2,000 caracteres | Comandos de compilación y prueba, estilo, directorios que evitar |
| `.pedacito/rules/*.md` | Solo cuando el modelo lee un archivo coincidente | Reglas para un área, con frontmatter `paths:` |
| `.pedacito/memory.md` | Cada tarea, como pistas y no como verdad | Hechos aprendidos de ejecuciones anteriores |

## Esquemas de color

La GUI y la TUI comparten un motor de temas (`pedacito/themes.py`). Un tema son exactamente **diez
tokens de color hexadecimales**. Cada token es `#rgb` o `#rrggbb`. Todos los tokens son obligatorios, y un
token desconocido se rechaza, de modo que un error tipográfico no pueda perder un color en silencio.

| Token | Pinta |
|---|---|
| `bg` | El fondo principal |
| `surface` | Paneles y tarjetas |
| `surface_alt` | Entradas, estados de hover y superficies elevadas |
| `text` | Texto principal |
| `muted` | Texto secundario |
| `accent` | Color de marca y de la acción principal |
| `border` | Divisores y bordes |
| `success` | Estado de éxito |
| `warning` | Estado de precaución |
| `danger` | Errores y fallos |

El motor incluye nueve esquemas integrados, el mismo conjunto que ofrece la GUI: `slate` (el predeterminado,
GitHub Dark), `light` (GitHub Light), `daylight` (papel cálido), `nord` (oscuro moderno), `midnight`
(alto contraste), y cuatro paletas de los años 90: `jazz_solo`, `trapper_keeper`, `memphis_arcade` y
`windbreaker`. En la TUI, `/theme` los lista y `/theme NAME` cambia y guarda la elección.
En la GUI, el botón de paleta de la barra superior abre el Theme Manager, que lista los mismos
esquemas y tus temas personalizados.

La clave `gui_theme` en `config.toml` contiene uno de los mismos nueve nombres; el selector de Settings de la GUI
y las etiquetas de esquema del Theme Manager la escriben ambos. Un navegador sin preferencia `gui_theme`
sigue el tema seleccionado en el Theme Manager o en la TUI (`themes.json`); las paletas personalizadas se
guardan allí también. Una preferencia guardada que nombra un preajuste heredado eliminado recurre a `slate`.

### Crea un esquema personalizado

Los esquemas personalizados viven en `themes.json`, junto a tu configuración de usuario. Puedes crear uno desde el
Theme Manager de la GUI, o escribir el archivo a mano:

```json
{
  "active_theme": "midnight-ink",
  "custom_themes": {
    "midnight-ink": {
      "bg": "#0f1020",
      "surface": "#171a2e",
      "surface_alt": "#22263f",
      "text": "#e6e8f5",
      "muted": "#9aa0c3",
      "accent": "#8ab4ff",
      "border": "#2c3152",
      "success": "#8fd3a0",
      "warning": "#f0c674",
      "danger": "#ff7b8a"
    }
  }
}
```

Un tema personalizado no puede reutilizar un nombre integrado. Una entrada editada a mano que falla la validación se omite,
de modo que una entrada incorrecta no impida que la aplicación inicie.

### Ancla un tema a un proyecto

Para usar un esquema distinto en un repositorio, crea `<project>/.pedacito/config.json`:

```json
{"theme": "nord"}
```

Un anclaje de proyecto válido anula el tema seleccionado globalmente solo en ese proyecto. La
selección global no cambia para ningún otro proyecto.

## Caché de prompts alojada

Pedacito mantiene estable el prefijo del prompt (el prompt del sistema y el primer turno del usuario
son idénticos byte a byte dentro de una recolección), de modo que un proveedor alojado que guarda en caché por su cuenta
lo reutiliza entre pasos automáticamente. Una tabla de proveedores coincide con el host de
`base_url` y decide el modo de caché, además de algunas peculiaridades de compatibilidad:

| Host | Caché | Qué define Pedacito |
|---|---|---|
| `api.deepseek.com` | Automática (implícita) | nada extra; lee tokens de acierto/fallo |
| `api.openai.com` | Automática para prompts ≥ 1024 tokens | `prompt_cache_key` enruta a una caché caliente |
| Qwen / DashScope | Implícita automática; explícita opcional | marcadores `cache_control` en modo `explicit` |
| `api.anthropic.com` | Explícita optativa (nativa) | `/v1/messages` nativo con marcadores `cache_control` (hasta 4 puntos de corte, TTLs `5m`/`1h`), `json_mode = "schema"`; la temperatura por debajo de 1 se omite (Claude 4.7+). Consulta `anthropic_native` para volver a la capa compatible con OpenAI |
| Moonshot, Gemini, xAI, Groq | Automática / implícita | nada extra; los aciertos se cuentan cuando se devuelven |

Con `anthropic_native = true` (el valor por defecto) la propia API de Anthropic usa el
transporte nativo `/v1/messages` en lugar de la capa compatible con OpenAI. Pedacito coloca
marcadores `cache_control` explícitos en el prefijo estable (hasta 4), respeta el TTL
`5m` / `1h` de `prompt_cache_ttl` e informa `cache_read_input_tokens` y
`cache_creation_input_tokens` de la respuesta como lecturas y escrituras de caché.
El razonamiento extendido, las llamadas a herramientas y las imágenes también usan el
formato nativo. Define `anthropic_native = false` para volver a la capa compatible con OpenAI.

Define `prompt_cache` para anular la tabla por perfil:

```toml
[profiles.qwen]
prompt_cache = "explicit"   # auto (the table decides) | implicit | explicit | off
prompt_cache_ttl = "5m"     # explicit-mode breakpoint TTL: "5m" or "1h"
prompt_cache_key = "auto"   # a stable hash of project + profile + model, sent
                            # only where the provider accepts it; "" = nothing
```

Las escrituras de caché se facturan por separado de las lecturas: define
`price_cache_write_per_mtok` para contarlas en el costo. `pedacito config --doctor`
sondea el proveedor activo e informa la caché como *observed*, *not observed*
o *explicit-only* (las cachés estilo DeepSeek persisten de forma asíncrona, así que "not
observed" no es "unsupported").

## Las APIs alojadas conservan su propio muestreo

Un proveedor alojado ajusta el muestreo por modelo por sí mismo, así que Pedacito deja de dictarlo:
cualquier perfil cuyo `base_url` no sea un servidor local (loopback o LAN) no recibe
**ni `temperature`, ni `top_p`, ni `top_k`, ni `min_p`, ni `repeat_penalty`** por defecto,
incluidas las claves de la tabla `sampler`. Se aplican los valores por defecto ajustados del proveedor,
que son los que usa su propia interfaz de chat. Las claves que no son de muestreo, como `seed`, siguen
pasando, y `presence_penalty` nunca se elimina.

Dos opciones voluntarias, ambas envían **solo** `temperature` (las otras cuatro claves siguen
omitidas en cualquier caso):

```toml
send_temperature_to_hosted = true   # send the configured temperature anyway
hosted_temperature = 1.0            # or pin one value for hosted endpoints,
                                    # globally or in a profile; it wins over
                                    # phase overrides (e.g. 1.0 for Claude)
```

Los servidores locales conservan cada valor configurado. La fase de respuesta usa por defecto
`temperature_answer = 0.3`: lo bastante fría para copiar el texto SEARCH exactamente, lo bastante
cálida para escapar de los bucles degenerados textuales en los que un `0.0` voraz congelaba a los modelos
locales débiles; y las tareas creativas/de diseño se ejecutan a ≥ 0.6 (consulta
[Modo creativo / de diseño](USAGE.md#modo-creativo--de-diseño)). Cuando una respuesta local aún
se repite, el reintento recibe un suave empujón de `presence_penalty = 0.15`
en lugar de una temperatura fría o una penalización de repetición pesada.

## Aceleración de inferencia local

La caché alojada reutiliza un prefijo del lado del proveedor; un servidor autoalojado puede hacer
lo mismo localmente, y Pedacito se lo pide con dos claves del servidor de llama.cpp
(entendidas por LM Studio, el servidor de llama.cpp, Lemonade, vLLM y la superficie OpenAI de Ollama):

```toml
[profiles.local]
enable_prompt_cache = true      # send cache_prompt (default: true)
enable_prompt_lookup = false    # send prompt_lookup_num_tokens (default)
prompt_lookup_num_tokens = 10   # draft span length when speculation is on
```

`cache_prompt` mantiene caliente el prefijo KV entre pasos de recolección, de modo que el
prefijo de conversación repetido no se vuelva a hacer prefill en cada llamada de búsqueda.
`prompt_lookup_num_tokens` activa la decodificación por búsqueda de n-gramas en el prompt: el borrador
proviene del propio prompt, lo que rinde en salidas con mucha copia (los bloques SEARCH
citan líneas de código fuente que están ahí mismo en el contexto).

Se cumplen tres propiedades de seguridad: las claves se adjuntan solo a endpoints que
autoalojas (loopback, una dirección LAN en el rango RFC1918 / CGNAT / unique-local de IPv6,
un nombre `.local` o una entrada de proveedor que nombra un runtime local), de modo que una
API alojada nunca las ve; un 400 o 422 que llega mientras alguna de las claves está
adjunta se trata como un rechazo, la petición se reenvía una vez sin ellas
y la denegación se recuerda por par servidor+modelo, de modo que las llamadas posteriores empiecen
sin ellas (una petición desperdiciada, una sola vez, en servidores anteriores a las claves); y una
clave que escribas tú mismo en la tabla `[sampler]` del perfil es del usuario, no
nuestra, y se deja en paz.

## Servidores MCP con nombre

Más allá de la lista heredada `mcp_servers = ["command", ...]` detrás de la acción
`mcp`, Pedacito puede llegar a servidores **con nombre** declarados como tablas, y se los
entrega al modelo mediante una sola acción `mcp_call`:

```toml
enable_mcp = true                       # the whole bridge; default true

[mcp_servers.docs]
command = "python"                      # stdio transport: argv, no shell
args = ["-m", "docs_server"]
env = "DOCS_DB=~/docs.db"               # or a JSON object { "K": "V" }
trusted = false                         # true: calls run without asking
enabled = true                          # false: park this server

[mcp_servers.jira]
url = "http://127.0.0.1:8000/mcp"       # Streamable HTTP instead of stdio
```

Todo se **poda por tarea**. Una vez por sesión el puente lista las herramientas de cada
servidor habilitado (un presupuesto de 5 segundos por servidor; un servidor caído
simplemente no aporta nada), puntúa los nombres y descripciones de las herramientas contra el
texto de la tarea e inyecta como máximo cuatro coincidencias como firmas de una línea en el
prompt de recolección. `mcp_call` se ofrece en el esquema de acciones solo cuando al menos
una herramienta coincide, y el modelo solo puede llamar a lo que se le mostró: las herramientas
irrelevantes de un servidor concurrido son invisibles, y el esquema por defecto para quienes
no configuran nada permanece idéntico byte a byte.

Todo está **sujeto a permiso por llamada**. Un servidor marcado como `trusted = true`
se ejecuta en silencio; en cualquier otro servidor cada llamada consulta al intermediario de permisos con
el servidor, la herramienta y los argumentos exactos en el diálogo (categoría **MCP Tool**),
y un rechazo devuelve un texto fijo que le dice al modelo que continúe solo con las herramientas
locales del repositorio. El texto devuelto se limita a unos `max_read_tokens`
tokens con una nota de truncamiento. Los archivos `.pedacito.toml` de proyecto no pueden definir
servidores en absoluto (la misma compuerta de confianza que `hooks` y los ajustes de shell): los servidores
con nombre provienen de tu configuración de usuario, o de un proyecto en el que confiaste explícitamente.

```bash
pedacito mcp list        # what the bridge sees for this project, pruned view
```

En la GUI, **⚙ Settings → MCP Servers** agrega, edita, prueba, habilita, confía
y elimina servidores con nombre sin editar TOML a mano; se escriben en tu
configuración de usuario (un archivo de proyecto necesita confianza primero). El mismo listado de herramientas
se sirve en `GET /api/mcp/tools`. Consulta
[Técnicas §30](../../TECHNIQUES.es.md)
para saber por qué la poda tiene esta forma.

## Hooks

Los hooks ejecutan tu propio programa en momentos bien definidos:

```toml
hooks = { post_edit = ["ruff format {file}"], pre_edit = ["./scripts/guard.sh"] }
```

Eventos: `session_start`, `pre_task`/`post_task`, `pre_tool`/`post_tool`,
`pre_edit`/`post_edit`, `pre_run`/`post_run`, `pre_compact`/`post_compact`,
`notification`.

El protocolo es JSON simple, sin ABI de plugins: el hook recibe un objeto JSON por
stdin (`{"event", "cwd", ...campos del evento}`) y puede responder por stdout con
`{"decision": "allow" | "deny", "message": "...", "context": "..."}`. Un
hook `pre_*` deniega con una decisión `deny`, o con el código de salida 2 y la
razón en stderr. El texto que imprime un hook (el campo `context`, o stdout simple
tras un evento `post_*`) se devuelve al modelo.

Los hooks se ejecutan como un argv, nunca mediante un shell, en la raíz del proyecto, con el
entorno depurado más `PEDACITO_EVENT`, y un tiempo límite (`hook_timeout`,
10 segundos por defecto). Los marcadores `{file}` / `{files}` se expanden a partir del
evento. Un hook que falla o excede el tiempo es una advertencia, no un error. Los hooks del propio
`.pedacito.toml` de un proyecto solo se ejecutan después de `pedacito trust`.

### Comandos del ciclo de vida del servidor

Los `load_cmd` / `unload_cmd` de un perfil local son la alternativa cuando la API de
ciclo de vida propia del runtime no puede cargar o descargar el modelo. Se ejecutan como un argv, nunca
mediante un shell, con el mismo entorno depurado que los hooks
(`scrub_env` / `env_passthrough`), un tiempo límite de 120 segundos y las
sustituciones textuales `{model}` / `{server}`. También reciben:

| Variable | Valor |
|---|---|
| `PEDACITO_SERVER_NAME` | el nombre del servidor configurado, o su URL base cuando no tiene nombre |
| `PEDACITO_MODEL_PATH` | el id o ruta del modelo pasado a la acción de carga/descarga |

Ninguno de los nombres es un ajuste de Pedacito, de modo que no puede eclipsar una anulación de configuración
`PEDACITO_<KEY>`. Con `scrub_env = false` estos comandos heredan el entorno
ambiental; agregar un nombre o un patrón `PREFIX*` a `env_passthrough` expone
solo esa variable mientras la depuración con lista permitida sigue activa. Un archivo de proyecto necesita
`pedacito trust` antes de que se ejecuten sus `load_cmd` / `unload_cmd`.

En **Lemonade Server**, un modelo cargado puede fijarse para que su expulsión LRU lo
omita. En la GUI abre **Servers**, elige el modelo residente en una fila de Lemonade y
haz clic en **Pin** o **Unpin**; en la TUI usa `/servers pin SERVER [MODEL]` o
`/servers unpin SERVER [MODEL]`. Pedacito llama a `POST /internal/pin` de Lemonade con
`{"model_name": "...", "pinned": true|false}` y
prueba `/api/v1/internal/pin` cuando una compilación solo responde a esa grafía. Cuando
`LEMONADE_ADMIN_API_KEY` está definido, se usa para este endpoint interno;
de lo contrario se aplica la clave normal de Lemonade o el `api_key` del perfil. El
modelo debe estar ya cargado. Los demás runtimes locales no tienen concepto de fijación, así que la
acción es una operación nula allí.

Cuando `auto_evict_servers` está activado (el valor por defecto), Pedacito descarga los residentes de los
*otros* servidores locales antes de que comience una ejecución. Primero lee el `/v1/health` de cada servidor
Lemonade; un residente fijado nunca se descarga con ese protocolo.
Solo se descargan los residentes sin fijar del mismo servidor, y los ids fijados
se informan en la lista `skipped` del protocolo. Los demás runtimes no exponen estado de fijación,
así que conservan el comportamiento existente de descargar todo.

## Memoria, skills e instrucciones

### Memoria, reglas y skills

Pedacito puede recordar lo que aprendieron las ejecuciones anteriores, como markdown simple que puedes
editar o eliminar:

- **Memoria** (`.pedacito/memory.md`, más un `memory.md` a nivel de usuario junto a tu
  configuración) contiene hechos breves bajo Commands / Conventions / Landmines / Notes.
  El modelo agrega uno con `remember`; tú también puedes:
  `pedacito memory PROJ --add "command: run tests with pytest -x"`.
  `pedacito memory PROJ` lista lo que se recuerda, `--forget TEXT` elimina
  las líneas coincidentes, y `--global` cambia cualquiera de los dos al archivo a nivel de usuario.
  Las líneas se deduplican, se limitan por categoría,
  los secretos se enmascaran, y cualquier cosa que suene a una instrucción para un
  asistente se rechaza. Se le muestra al modelo como *pistas, no verdad*.
- **Reglas** (`.pedacito/rules/*.md` con una línea de frontmatter `paths: src/api/*.py`,
  y archivos `AGENTS.md`/`CLAUDE.md` anidados) se muestran solo cuando el modelo
  lee un archivo coincidente. Pedacito también administra automáticamente archivos de reglas `auto_*.md` aquí
  (lecciones de reparación promovidas desde `micro_facts.json`; consulta
  [Lecciones de reparación](ARCHITECTURE.md#lecciones-de-reparación-y-trampas-del-repositorio)): cualquier cosa sin la
  marca de frontmatter `auto_generated: true` nunca es escrita ni eliminada por él.
- **Skills** son procedimientos con nombre (`.pedacito/skills/NAME/SKILL.md` o
  `.pedacito/skills/NAME.md`, y las mismas dos estructuras bajo `skills/` junto a
  tu configuración). El modelo ve un catálogo de una línea y carga el texto de una skill solo
  cuando lo necesita; además, las 2 mejores coincidencias de disparadores para una tarea se
  inyectan de antemano, limitadas a 1,200 caracteres de texto de lista de verificación. Vienen
  diecinueve iniciales incluidas: `add-a-test`, `fix-a-failing-test`, `refactor-safely`,
  `review-my-diff`, `find-dead-code`, `surgical-symbol-edit`,
  `frontend-dom-audio-debug`, `safe-multi-file-rename`, `fastapi-rest-endpoint`,
  `canvas-svg-2d-3d`, `sqlite-sql-migrations`, `async-concurrency-safety`,
  `security-ssrf-xss-hardening`, `i18n-localization-catalog`,
  `parser-lexer-state-machine`, `performance-memory-profiling`,
  `modern-web-design-ui`, `desktop-gui-and-games` y `safe-git-commit`.
  `pedacito skills list` las muestra todas con una insignia de alcance (bundled / user /
  project). Las skills son texto de prompt; nada en una carpeta de skill se ejecuta jamás.
- **Skills personalizadas** son tuyas para escribir: `pedacito skills add NAME
  --description "..." --triggers "k1,k2" --body "1. ..."` guarda una en el
  proyecto (o `--global` para la biblioteca del usuario), `pedacito skills draft "Skill
  for Alpine.js dark-mode components"` deja que el modelo activo escriba toda la
  skill a partir de una frase, `pedacito skills show NAME` imprime sus disparadores y
  lista de verificación, y `pedacito skills rm NAME` elimina una skill personalizada (las integradas
  son inmutables; una skill de proyecto puede eclipsar una integrada del mismo nombre).
  El equivalente en la GUI es **Project ▾ → Skills library & creator…** (también enlazado desde
  ⚙ Settings): explora cada alcance, previsualiza qué skills inyectaría una tarea
  y crea skills con el botón **✦ Auto-Draft from Description**, guardando
  en el proyecto o en la biblioteca global.

### Instrucciones del proyecto

Pedacito lee un archivo de convenciones en cada prompt: el primero que encuentre de
`PEDACITO.md`, `AGENTS.md`, `CLAUDE.md`, `.pedacito/instructions.md` o
`.pedacito/PEDACITO.md` (la grafía anterior `PEDACEITO.md` también se acepta).
Coloca ahí los comandos de compilación/prueba, las reglas de estilo y los directorios prohibidos. Está
limitado por `max_instruction_chars` (2000) y se desactiva con
`use_project_instructions = false`.

## Perfiles, niveles y presupuestos de tokens

`pedacito profiles` lista cada perfil con su ajuste al arnés medido (media ±
dispersión), la tasa de aprobación de la evaluación cuando existe un informe, la velocidad de generación, el contexto
que cabe en su VRAM y el estado de la tarjeta; `--recommend` agrega sugerencias por rol
y `--json` emite lo mismo como JSON. `pedacito models scan` escribe un
perfil solo de identidad por cada modelo que lista un servidor.

Define un perfil por modelo en lugar de editar `model =` cada vez:

```toml
[profiles.small]
model = "qwen2.5-coder:7b"

[profiles.big]
model = "some-27b-model"
tier  = "large"
```

```bash
pedacito ask PROJ -t "..." --profile big
```

Elige un perfil con `default_profile =`, `PEDACITO_PROFILE`, `--profile` o
`/profile` en `chat` (precedencia creciente). Un perfil puede anular cualquier
ajuste, no solo el modelo.

`tier = "tiny" | "small" | "medium" | "large" | "local-moe" | "hosted"` completa todos los presupuestos de tokens
(`max_tokens_*`, `flat_map_token_budget`, `max_gathered_chars`, ...)
a partir de un preajuste; `tiny` sirve para modelos de ≤4B con una ventana de ~4k. Cuando no se define, el nivel
se deduce del nombre del modelo (≤4B → tiny, ≤10B → small, ≤20B → medium, si no
large; una palabra clave de alojado gana). Rara vez necesitas ajustarlos: Pedacito detecta
la ventana de contexto real del servidor y el tamaño de archivo del modelo y ajusta cada
presupuesto a tu VRAM (`auto_fit`, activado por defecto). Los valores explícitos siempre ganan,
y `pedacito config` marca cada valor no predeterminado.

Si tu runtime usa una caché KV cuantizada, define `k_cache_quant` y
`v_cache_quant` (p. ej. `"q8_0"`, `"q5_1"`) para que la estimación de VRAM no sea demasiado
pesimista. Si el servidor informa una ventana por defecto desactualizada en lugar de la que
configuraste, define `context_tokens` explícitamente.

Cambiar de modelo hace que el servidor cargue el nuevo; Pedacito espera hasta
`load_wait_seconds` (240).

Los modelos de razonamiento se admiten entre familias: `<think>`/`<thinking>`/
`<reasoning>`, el `[THINK]` de Magistral, el `<seed:think>` de Seed-OSS, los canales `analysis`
de harmony de GPT-OSS y el bloque `<|channel>thought` de Gemma 4 se
eliminan antes del análisis, mientras que los campos separados `reasoning_content`/`reasoning`/
`thinking` se leen directamente. El
interruptor `disable_thinking` es consciente de la familia: las plantillas locales reciben los kwargs comunes
`enable_thinking=false` y `thinking=false` (las variables de plantilla desconocidas
se ignoran), más `reasoning_effort=none` para Magistral y
`thinking_budget=0` para Seed-OSS. Los proveedores alojados reconocidos reciben su propia
grafía (el `thinking={"type":"disabled"}` de Kimi, el `enable_thinking` de DashScope, el kwarg
de plantilla de DeepSeek, `reasoning_effort` en OpenAI/Cerebras/GPT-OSS); un modelo alojado desconocido no recibe ningún campo extra, de modo que una
API estricta no pueda rechazarlo. `thinking_suffix` agrega un interruptor suave en el prompt, y
selecciona automáticamente `/no_think` para Qwen3/SmolLM3. Las familias Qwen, Gemma, Nanbeige y
LiquidAI usan todas `enable_thinking` y se cubren automáticamente.
Algunas familias no tienen interruptor de apagado (GPT-OSS, Kimi K2.7/K3, LFM2.5-24B-A2B); sus
bloques de razonamiento aún se eliminan, pero los tokens son inevitables. Para una
familia no reconocida, agrega el interruptor que tu modelo espera bajo la tabla `sampler` del perfil
(las claves del usuario sobreviven y se combinan con las automáticas).
La alternancia de roles estilo Mistral y la falta de rol de sistema de Gemma se manejan
automáticamente.

`reasoning_effort` acepta `default`, `low`, `medium` y `high`. `default` es
la grafía incluida del `off` del plan: no envía ningún parámetro `reasoning_effort`.
Los otros tres valores se reenvían tal cual en los modelos local-MoE
y alojados con capacidad de razonamiento, nunca en modelos que no exponen el
control, y nunca cambian el muestreo. `disable_thinking = true` gana sobre
cualquier nivel de esfuerzo.

## Perfilado avanzado del sistema y auto-ajuste

`pedacito profile` muestra la instantánea de hardware, la última medición de velocidad en caché y el
ajuste del perfil activo. `pedacito models` (sin acción) imprime los mismos números junto al
modelo activo; `pedacito models scan` no cambia.

```bash
pedacito profile             # snapshot + last probe + active tuning (same as --show)
pedacito profile --show
pedacito profile --run       # force one prefill/generation probe now
pedacito profile --reset     # drop cached measurements, turn off auto_tune = true
pedacito profile --show --json
```

La TUI expone las mismas tres acciones como `/profile show`, `/profile run` y
`/profile reset`. `/profile NAME` simple sigue cambiando de perfil. Tanto `--reset` como
`/profile reset` descartan la caché y desactivan un `auto_tune = true` explícito en el mismo
archivo de configuración seleccionado (el archivo de `--config` cuando se pasó uno, de lo contrario la configuración de usuario).

### Qué lee la telemetría

Cada número lleva su fuente. Un lector ausente en esta plataforma no aporta
nada; nunca se trata como cero de memoria libre. Cada lector está limitado a dos segundos
y cada fallo es silencioso (solo a nivel de registro). La instantánea se memoiza por dos segundos, de modo que
una ráfaga de llamadores (carga de configuración, derivación de roles, carga útil de la GUI) comparte una pasada del lector.

| Plataforma | RAM del sistema | VRAM de la GPU | Etiqueta de fuente |
|---|---|---|---|
| Linux con AMD o Intel (amdgpu, Intel xe, cualquier controlador que exponga `mem_info_vram_*`) | `/proc/meminfo` | `/sys/class/drm/card*/device/mem_info_vram_total` y `_used` | `drm-sysfs` |
| Linux con NVIDIA | `/proc/meminfo` | `nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader,nounits` (argv, nunca una cadena de shell) | `nvidia-smi` |
| macOS | `sysctl -n hw.memsize` total, páginas libres/inactivas/especulativas de `vm_stat` | ninguna: la memoria unificada de Apple Silicon cuenta como RAM, nunca como una reclamación de VRAM separada | `vm_stat` |
| Windows | `kernel32.GlobalMemoryStatusEx` mediante `ctypes` de la biblioteca estándar (la misma API en x86-64 y ARM64) | NVIDIA mediante `nvidia-smi`; AMD/Intel no tienen lector independiente del fabricante, así que auto-tune se abstiene allí | `winapi` |

La insignia de hardware de la GUI muestra **solo totales**. No hay atribución de GPU por proceso,
y sus etiquetas lo indican. `servers.get_hardware_memory()` conserva su forma y valores originales;
la instantánea del perfilador es un superconjunto, de modo que los consumidores antiguos de la GUI/TUI siguen funcionando.

### Huella y confianza

`estimate_model_footprint` estima dos cosas:

- **Pesos residentes** = parámetros × bytes por parámetro. Las cuantizaciones aceptadas son
  f16 = 2, q8 = 1, q6 = 0.75, q5 = 0.625, q4 = 0.5 y q2 = 0.25 bytes por parámetro. Una
  cuantización desconocida usa el valor conservador de 0.6.
- **Caché KV** usa el mismo modelo de tamaño que `auto_fit`: un
  `kv_cache_bytes_per_token` explícito se usa tal cual; de lo contrario, un tamaño de archivo de modelo conocido
  da una caché de 32k tokens aproximadamente del tamaño de los pesos, escalada por
  `k_cache_quant`/`v_cache_quant` (sin definir = f16). Sin un tamaño de archivo, la
  fórmula de arquitectura es 2 (K y V) × capas × cabezas KV × dimensión de cabeza, usando
  la cuantización de los pesos cuando no hay cuantización de caché definida. El perfilador informa
  GB por cada 1,000 tokens.

Un modelo denso mantiene todos los pesos residentes en VRAM. Un modelo de mezcla de expertos informa ambos
números: todos los expertos en RAM (`resident_gb`) y solo los expertos activos residentes en VRAM
(`active_gb`/`vram_resident_gb`), de modo que se pueda razonar sobre un perfil Strata/Avalon por niveles.

Los metadatos de arquitectura se leen en este orden: anulaciones explícitas de configuración
(`model_params_b`, `model_active_params_b`, `model_layers`, `model_kv_heads`,
`model_head_dim`, `model_quant`, `model_arch`), `/props` y `/models` de llama.cpp, `/api/show` de
Ollama, y luego heurísticas del nombre del modelo. `confidence` es más alta cuando cada campo proviene
de configuración explícita o de metadatos del servidor (0.95 / 0.85), y baja por cada campo inferido
(una cuantización desconocida, un número de capas adivinado, un número de cabezas KV adivinado o una dimensión
de cabeza adivinada). Una confianza de `0.00` significa que la estimación es inutilizable. El informe nunca presenta una
suposición como medida; `estimated_fields` lista exactamente lo que se infirió.

### Reglas de `auto_tune`

`auto_tune` es **false por defecto**. Los usuarios existentes no ven cambio alguno: no se ejecuta ningún lector de hardware
ni ajuste de presupuestos hasta que lo activen. La sonda de velocidad automática posterior a la carga es un
comportamiento aparte, controlado por `profile_probe_on_load` (true por defecto); defínelo en false para
evitar que incluso esa se inicie tras una carga.

Cuando está activado para un perfil local:

- **context_tokens** está limitado por `(memoria libre − pesos − reserva de seguridad) / GB de KV por
  1,000 tokens`. Siempre queda un piso de seguridad de 4,096 tokens (o el valor menor del usuario).
  Cuando el conjunto está ajustado, la ventana se reduce a ese piso. Cuando el conjunto es holgado, el crecimiento
  se *propone* hasta la ventana informada por el servidor y nunca se fuerza. Cuando se conoce el tamaño del archivo del modelo,
  la propuesta también está limitada por la misma ventana de VRAM que deriva `auto_fit`,
  de modo que `k_cache_quant` / `v_cache_quant` nunca puedan hacer que el ajustador sugiera una ventana que el
  servidor no pueda sostener.
- **La confianza escala la acción.** Por debajo de una confianza de huella de 0.50 (por ejemplo,
  toda la geometría KV adivinada a partir del nombre del modelo), el ajustador aún calcula los
  números de contexto y de presupuesto de recuperación que resultarían, pero los informa solo como notas: no se
  escribe ningún valor. Con 0.50 o más se aplican los mismos cálculos de antes. Cuando un campo de
  geometría KV fue estimado, las notas lo dicen y apuntan a `model_layers` / `model_kv_heads` /
  `model_head_dim` / `model_quant` o a `pdc profile --run`.
- **Presupuestos de recuperación** (`max_gathered_chars`, `flat_map_token_budget`) escalan con el
  contexto ajustado; los perfiles local-MoE con mucha memoria reciben una ventana menor que los perfiles densos
  con el mismo contexto.
- **timeout** se mantiene en 600 s para tiered/local-MoE. Para local-dense pasa a ser
  `max(60, context / measured prefill tok/s × 1.5)` cuando existe una sonda. Un perfil alojado
  recibe el valor por defecto de 30 s para la nube; su contexto, recuperación y presupuestos derivados del hardware
  nunca se tocan.
- **max_concurrency** se mantiene en 1 para tiered/local-MoE y nunca se aumenta. Los perfiles local-dense y
  alojados conservan su valor.
- **Los valores fijados por el usuario ganan.** Cualquier campo cuya procedencia sea un archivo de configuración explícito,
  perfil, variable de entorno u opción de CLI se deja en paz. Un modelo con cada campo ajustable
  fijado no se toca.
- **Varias tarjetas no se suman para el margen.** Los totales de VRAM siguen siendo un inventario del sistema
  sumado para la insignia de la GUI y el encabezado del informe, pero cuando las tarjetas difieren en tamaño
  (por ejemplo una iGPU de 0.5 GB junto a una tarjeta discreta de 24 GB) el ajuste usa los números totales y libres de
  la tarjeta más grande y nombra las tarjetas ignoradas en una nota
  (`tuned against card1 (24.0 GB); card0 ignored`). `pdc profile` marca la base de ajuste
  por tarjeta. Las tarjetas del mismo tamaño y los sistemas de una sola tarjeta conservan la ruta sumada; una
  tarjeta con memoria libre desconocida nunca se trata como cero.
- **Sin telemetría, se abstiene.** Sin lector de VRAM, auto-tune usa solo el margen de RAM en
  una plataforma de memoria unificada (macOS). En todos los demás casos deja la configuración intacta; un
  lector ausente nunca se interpreta como cero de memoria libre.

**Quién es dueño de un valor.** Un valor que defines (archivo de configuración, perfil, entorno o CLI) es
explícito tanto para `auto_fit` como para el ajustador, y ninguno lo sobrescribe. Un valor escrito por
auto-tune se etiqueta `auto-tune (...)`; `auto_fit` trata esa etiqueta como explícita, de modo que su ventana de VRAM
no reemplazará el número ajustado en una carga posterior, mientras que el ajustador trata su propia
etiqueta como blanda y puede refinar una pasada de ajuste anterior cuando la instantánea de hardware o los metadatos
cambian. Ese anclaje unidireccional es deliberado: `auto_fit` respeta la ventana ajustada, y el
ajustador sigue pudiendo corregir su salida anterior. Para fijar un número frente a ambos, defínelo
explícitamente en la configuración o el perfil.

Los orígenes por campo aplicados usan la misma convención de procedencia `sources` que los valores por defecto de
`local-moe`, por ejemplo
`auto-tune (VRAM 19.8 GB free via nvidia-smi; weights 7.0 GB; KV 0.0458 GB/1k tok; confidence 0.85; floor 4096)`.

### Sonda de velocidad

`pedacito profile --run` (y, una vez por servidor+modelo, una carga explícita exitosa del modelo)
envía un completado diminuto (`max_tokens = 16`, "Reply with just: OK") más una medición
de prefill: unos 2,000 tokens de relleno cuando la ventana del modelo lo permite, de lo contrario 512.
Define `profile_probe_on_load = false` para evitar que la sonda automática posterior a la carga se inicie
en absoluto; las sondas explícitas siguen funcionando. Los tok/s de prefill y de generación provienen de los tiempos
SSE del servidor. La sonda maneja una llamada pública de un solo disparo (`LLMClient.complete_once()`)
y lee el resultado mediante el registro de tiempos público
(`LLMClient.timing_record()`): anclas de petición/primer token, el bloque de tiempos del servidor
y los conteos de tokens, nunca la contabilidad privada del stream. La sonda es de mejor esfuerzo
y silenciosa ante fallos; un resultado en caché vive en `profiler.json` junto a la configuración de usuario por
una semana y está indexado por servidor+modelo, de modo que cambiar cualquiera de los dos lo invalida. Las sondas nunca
corren para endpoints alojados: ambas rutas explícitas (`pdc profile --run` y `/profile run`)
se rehúsan con un mensaje "hosted model — refusing to probe" antes de enviar una petición, y
la sonda automática posterior a la carga se omite para una URL alojada.

## Niveles de verificación y la familia `auto_verify`

Cada edición aplicada pasa hasta por tres niveles de verificación, del más barato al más caro. Los
dos primeros son deterministas y no necesitan llamada al modelo; el tercero es el comando que
eliges.

| Nivel | Qué se ejecuta | Límites y valores por defecto |
|---|---|---|
| 1. Compuerta de sintaxis / estática | Los archivos de Python modificados se verifican en busca de sintaxis (`E999`) y nombres no definidos / entradas `__all__` incorrectas / importaciones fuera de lugar (`F821`, `F822`, `F823`, `F402`, `F404`) con `ruff`; un resolvedor de AST integrado toma el relevo cuando ruff falta, falla o excede su techo. Los hallazgos preexistentes se excluyen de la línea base, de modo que solo los errores que introdujo esta edición alimentan la reparación. | `static_gate = true`; `static_gate_timeout = 3.0` segundos por cada ejecución de ruff. |
| 2. Pruebas afectadas acotadas | `testsel.run_impacted` selecciona pruebas a partir del grafo de importaciones (pruebas con el nombre del archivo modificado, pruebas de sus importadores/llamadores, importadores directos) y ejecuta un `pytest FILE...` / `npm test -- FILE...` / `go test ./dir/...` acotado. | Se ejecuta tras las ediciones cuando `verify_tests = true` y el subproceso de pruebas está permitido (`allow_shell` o un comando de pruebas concedido por el usuario); una ejecución está limitada por `test_timeout = 300` segundos. |
| 3. Suite completa | Tu comando de verificación se ejecuta después de cada tarea de Auto (o una vez al final con `auto_verify_final`), y su salida de fallo alimenta las rondas de reparación. La suite completa deliberadamente nunca se selecciona automáticamente. | `auto_verify = ""` (desactivado) es una **cadena de comando**, nunca un bool; `auto_repair_rounds = 2`; `auto_verify_rounds = 2` para la verificación en tiempo de ejecución posterior a la ejecución. |

`auto_verify` es el comando que activan la casilla **Post-edit verify** de la GUI y el
`/verify on` de la TUI (ambos primero prueban
`testsel.detect_test_command`, y dejan la clave vacía cuando no hay un ejecutor
evidente). `auto_verify_final = true` difiere ese comando a la última tarea sin marcar
del plan, de modo que una verificación de toda la meta no pueda revertir correcciones parciales válidas.
`auto_repair_rounds` limita las rondas de corrección por cada verificación fallida por tarea;
`auto_verify_rounds` limita la verificación en tiempo de ejecución realizada tras una
ejecución `auto --apply`.

Para una ejecución de **una sola tarea**, `pedacito ask --verify-cmd "COMMAND"` anula
`auto_verify` en memoria (la entrada `sources` de la configuración en vivo pasa a ser
`--verify-cmd`; no se escribe ningún archivo de configuración) y ejecuta el comando una vez tras una
tarea aplicada. `pedacito ask --json` informa ese resultado como `verify_passed`
(`null` cuando no corrió ninguna verificación) en su resumen de un solo objeto; consulta
[USAGE.md — Modo headless y automatización](USAGE.md#modo-headless-y-automatización) para la
forma JSON exacta.

La exportación de trayectorias es un ajuste de conjuntos de datos separado y opcional:
`record_trajectories = false` por defecto. Cuando es true, cada llamada a `loop.run_task`
escribe una transcripción ChatML redactada bajo
`.pedacito/trajectories/` (consulta
[SECURITY.md — Exportación de trayectorias](SECURITY.md#exportación-de-trayectorias)).
`pedacito ask --save-trajectory` la fuerza solo para esa ejecución, con una entrada `sources` en vivo de
`--save-trajectory`; al igual que `--verify-cmd`, nunca escribe
el archivo de configuración.

`auto_verify` y `test_command` son ajustes de ejecución: en el
`.pedacito.toml` de un proyecto se respetan solo después de `pedacito trust` (consulta
[SECURITY.md](SECURITY.md#la-configuración-del-proyecto-no-es-confiable-por-defecto)).

Los niveles se conectan con dos protecciones existentes:

- **Trinquete verde** (`auto_green_ratchet = true`): antes de cada paso de Auto se guarda una
  instantánea del código fuente; un paso que aún falla tras sus rondas de reparación se
  revierte exactamente, de modo que los pasos posteriores se construyan solo sobre el último
  espacio de trabajo verificado en verde. Se conservan `auto_keep_checkpoints = 25` instantáneas verificadas en verde.
- **Bloqueo de interfaz** (`interface_lock = true`): las firmas públicas del proyecto
  se inyectan en cada tarea de Auto, y una verificación determinista de contratos entre archivos
  (existencia de importaciones locales y aridad de llamadas) se ejecuta tras cada edición antes de
  el subproceso de pruebas.

## Idioma y localización (`language`, `llm_reply_language`)

Pedacito incluye una interfaz en inglés y otra en español.

| Ajuste | Valores | Por defecto | Significado |
|---|---|---|---|
| `language` | `"en"` \| `"es"` | `"en"` | Idioma de la interfaz para menús, mensajes y el chrome de la GUI/TUI (catálogos en `pedacito/locales/<code>.json`). |
| `llm_reply_language` | `"match_ui"` \| `"auto"` \| `"en"` \| `"es"` | `"match_ui"` | Idioma de las explicaciones en prosa del modelo. `match_ui` sigue a `language`, `auto` sigue el idioma de tu prompt, `en`/`es` fuerzan uno. El código, los identificadores y los bloques de edición nunca se traducen. |

Formas de elegir el idioma, de mayor a menor precedencia:

- **CLI:** `pedacito --lang es ...` (también aceptado por `gui`, `config`, `how`, `models`, `profiles`, `update`).
- **Entorno:** `PEDACITO_LANG=es` (alias `PEDACITO_LANGUAGE`).
- **Configuración:** `language = "es"` en tu configuración de usuario o de proyecto.
- **GUI:** el selector `EN | ES` en la barra superior; la elección se guarda en `language`.
- **TUI:** `F3` alterna EN/ES, o escribe `/lang es`.

## Generación local de imágenes (`image_profile`, `allow_image_gen`)

La generación de imágenes es una modalidad opcional, no un modelo de chat. Un perfil con
`modality = "image"` (se define solo cuando `models scan` o la GUI ven una receta `sd-cpp` de
Lemonade Server, un servidor Stable Diffusion / Flux o un id de modelo de imagen) se enruta a
`/images/generations` en lugar del ciclo de herramientas de texto.

```toml
[profiles.flux]
base_url = "http://127.0.0.1:8000/api/v1"
model = "SD-Turbo"
modality = "image"
image_default_steps = 4

[universal]
image_profile = "flux"       # vacío = primer perfil de imagen encontrado
allow_image_gen = true       # expone la acción generate_image del agente
```

- `pedacito image PROMPT`, el comando `/image` de la TUI y el diálogo **Project ▾ →
  🎨 Generate image…** de la GUI usan `image_profile`. Los archivos se guardan en
  `.pedacito/images/` salvo que pases `--out`; la ruta debe quedar dentro del
  espacio de trabajo.
- `allow_image_gen` solo controla la acción `generate_image` del agente. Con él
  desactivado, el intermediario de permisos de la GUI aún puede aprobar una llamada a la vez. Es
  un interruptor de ejecución, así que el `.pedacito.toml` de un proyecto no puede activarlo
  hasta que ejecutes `pedacito trust`.
- `image_default_size` e `image_default_steps` son los valores de respaldo para cualquier
  petición que no indique ninguno.
- `pedacito calibrate` sobre un perfil de imagen ejecuta una sola sonda de disponibilidad de 512×512
  en lugar de las sondas de texto; el botón **Probe image** de la GUI hace lo mismo. Consulta la
  [guía de modelos](../../MODEL_GUIDE.es.md#modelos-de-imagen-modality--image).

## Referencia de configuración

`pedacito config` lista cada ajuste con su valor y origen; `pedacito
config --reference` imprime la plantilla anotada. Ajustes principales:

| Ajuste | Por defecto | Significado |
|---|---|---|
| `base_url` | `http://localhost:1234/v1` | Dirección del servidor. |
| `model` | `google/gemma-3-27b` | Id/nombre exacto que informa el servidor. |
| `api_key` / `api_key_env` | `"lm-studio"` / `""` | Para APIs alojadas y servidores protegidos. `api_key_env` guarda el *nombre* de una variable de entorno que contiene la clave, de modo que el secreto nunca llegue al archivo de configuración. |
| `vram_gb` | *(obligatorio para local)* | VRAM de la GPU; se define en `[universal]`, anulable por perfil. |
| `context_tokens` | `8192` | Ventana de contexto del lado del servidor. Con `auto_detect_context` activado (el valor por defecto) la ventana informada por el servidor la reemplaza; defínela explícitamente cuando la detección sea incorrecta. |
| `auto_tune` | false | Activa el ajuste de contexto, presupuestos de recuperación y tiempo límite guiado por hardware/sondas. Desactivado por defecto: nada cambia hasta que sea true. Consulta [Perfilado avanzado del sistema y auto-ajuste](#perfilado-avanzado-del-sistema-y-auto-ajuste). |
| `profile_probe_on_load` | true | Programa la sonda de velocidad en segundo plano una vez por servidor+modelo tras una carga explícita exitosa. `false` = no se inicia ningún hilo de sonda tras una carga; `pdc profile --run` sigue funcionando. |
| `load_cmd` / `unload_cmd` | `""` / `""` | argv alternativo para un runtime sin endpoint de ciclo de vida utilizable. Entorno depurado más `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH`. Consulta [Comandos del ciclo de vida del servidor](#comandos-del-ciclo-de-vida-del-servidor). |
| `model_params_b` / `model_active_params_b` | 0 / 0 | Conteos de parámetros explícitos (en miles de millones) para la estimación de huella; active son los expertos MoE usados por token. 0 = descubrir, y luego recurrir a heurísticas del nombre con menor confianza. |
| `model_layers` / `model_kv_heads` / `model_head_dim` | 0 / 0 / 0 | Geometría explícita de la caché KV. Los campos faltantes reducen la `confidence` de la estimación en lugar de informarse como medidos. |
| `model_quant` / `model_arch` | `""` / `""` | Cuantización de pesos explícita (`q4`, `Q4_K_M`, `q8`, `f16`, ...) y `dense`/`moe`. Vacío = inferir de los metadatos del servidor y luego del id del modelo. |
| `small_context_tokens` / `max_whole_file_lines` | 16384 / 120 | Una ventana igual o menor que `small_context_tokens` cuenta como "pequeña": el resumen de respuesta entonces exige ediciones fragmentadas/ancladas, y una reescritura de archivo completo de un archivo mayor que `max_whole_file_lines` se rechaza antes de aplicarse (a menos que la tarea pida explícitamente reescribir ese archivo, sea una tarea creativa/de diseño o apunte a `.html`/`.htm`/`.css`/`.md`; consulta [Modo creativo / de diseño](USAGE.md#modo-creativo--de-diseño)). `small_context_tokens = 0` desactiva la disciplina. |
| `k_cache_quant` / `v_cache_quant` | `""` | Cuantización de la caché KV que usa el runtime. |
| `json_mode` | `"schema"` | `"schema"` para servidores locales, `"object"` para alojados (cambio automático). |
| `enforce_json_schema` | true | Pide al motor que restrinja los pasos de recolección con `response_format` json_schema (estricto); los servidores que lo rechazan se degradan a json_object y luego a prompts simples, por modelo. |
| `enable_run_test_in_gather` | true | Deja que el ciclo de recolección ejecute una verificación de prueba acotada (pytest, Node/Vitest, Go o Cargo, elegidos según el objetivo) para ver tracebacks en vivo. `false` hace que cada ejecución consulte al intermediario de permisos. |
| `frontend_contracts` | true | Antes de que corran las pruebas, verifica los enlaces locales de scripts/hojas de estilo, las búsquedas literales de ids del DOM y la sintaxis del JavaScript modificado. |
| `auto_heal_imports` | true | Antes de que un hallazgo de la compuerta estática llegue al modelo, inserta la única importación que demostrablemente lo resuelve (un nombre de la biblioteca estándar, o un símbolo que exporta exactamente un módulo local), verificada por el verificador de AST; los nombres ambiguos nunca se adivinan. |
| `collapse_repair_turns` | true | Cuando una reparación tiene éxito, descarta los borradores fallidos de la cola de la conversación y conserva un recibo compacto, de modo que se reutilice la caché de prefijo de prompt del servidor en lugar de volver a hacer prefill de la trayectoria rota. |
| `micro_facts` / `max_micro_fact_tokens` | true / 150 | Registra comandos confirmados y lecciones de reparación en `.pedacito/micro_facts.json` y los inyecta como pistas; el tope de tokens aplica al bloque inyectado. Una lección confirmada dos veces para un directorio se promueve a una regla con alcance de ruta administrada automáticamente en `.pedacito/rules/auto_*.md` en lugar de quedarse en cada prompt. |
| `tier` | `""` | Preajuste de presupuestos: `tiny`/`small`/`medium`/`large`/`local-moe`/`hosted`; se deduce del nombre del modelo cuando no se define. |
| `max_steps` | 0 | Límite de búsquedas por tarea; 0 = ninguno. |
| `prefetch_targets` | 0 | Lee hasta esta cantidad de archivos/símbolos nombrados por la tarea mediante la ruta normal de herramientas antes del primer paso de búsqueda (y una vez más tras la recolección cuando un objetivo nombrado aún no se ha leído). Las lecturas de prefetch no cuentan para `max_steps`. |
| `temperature` | 0.2 | Valores más altos empeoran la selección de herramientas. Una tarjeta de modelo calibrada puede reemplazar este valor por defecto; un valor definido en un perfil, una variable `PEDACITO_*` o un flag de CLI siempre tiene prioridad. Solo servidores locales: las APIs alojadas no reciben claves de muestreo por defecto (consulta [Las APIs alojadas conservan su propio muestreo](#las-apis-alojadas-conservan-su-propio-muestreo)). |
| `temperature_answer` / `temperature_repair` | 0.3 / unset | Temperatura de muestreo para la respuesta y para los reintentos de reparación, que copian el texto SEARCH carácter por carácter. La respuesta usa `0.3` por defecto: lo bastante fría para copiar exactamente, lo bastante cálida para escapar de los bucles de repetición textual (un `0` voraz congelaba a los modelos locales débiles); las tareas creativas/de diseño se ejecutan a ≥ 0.6. Los reintentos de reparación están sin definir (usan `temperature`). La recolección conserva `temperature`; los candidatos de `best_of` se ejecutan a la temperatura de respuesta + 0.4. |
| `send_temperature_to_hosted` | false | Envía de todos modos la `temperature` configurada a las APIs alojadas. Las otras claves de muestreo (`top_p`, `top_k`, `min_p`, `repeat_penalty`) siguen omitidas en cualquier caso. |
| `hosted_temperature` | unset | Fija una temperatura para los endpoints alojados, globalmente o por perfil (p. ej. `1.0` para Claude); gana sobre las anulaciones por fase y es la única clave de muestreo que se envía. Sin definir: se aplica el valor por defecto del propio proveedor. Los servidores locales la ignoran. |
| `anthropic_native` | true | Enruta la propia API de Anthropic por el transporte nativo `/v1/messages` (caché de prompts, uso en streaming, razonamiento extendido, herramientas e imágenes); `false` usa la capa compatible con OpenAI. |
| `gemini_native` | true | Enruta Google Gemini (`generativelanguage.googleapis.com` sin `/openai`, o `provider = "gemini"`) por el transporte nativo `/v1beta` `generateContent` (partes de pensamiento, imágenes en línea, declaraciones de funciones, `thinkingConfig` y `cachedContent`); `false` usa la capa compatible con OpenAI. |
| `openai_responses` | false | Transporte opcional OpenAI Responses API (`/v1/responses`) para modelos de razonamiento (`o3`/`o4`/`GPT-5`), conservando elementos de razonamiento cifrados y `previous_response_id` entre turnos; también se activa con `provider = "openai-responses"`. |
| `auto_refresh_model_catalog` | true | Actualiza silenciosamente la caché local de `models.dev` (`model_catalog.json`, TTL 24 h) al sondear o actualizar proveedores alojados para mantener vigentes las ventanas de contexto y precios; `false` mantiene todo estrictamente sin conexión. Las búsquedas usan tres niveles: los metadatos `/models` del propio proveedor (nivel 1), esta caché (nivel 2) y luego la tabla de precios integrada (nivel 3); los ajustes `price_*` explícitos siempre tienen prioridad. `pedacito models refresh` fuerza una actualización. |
| `api_key_keyring` | `""` | Referencia opcional al llavero del sistema `<servicio>/<cuenta>` (o `<cuenta>`, con servicio `pedacito` por defecto) resuelta bajo demanda con `pedacito[keyring]`; alternativa a `api_key = "keyring:<servicio>/<cuenta>"`. |
| `backup` | true | Guarda una instantánea de los originales antes de escribir. |
| `log_sessions` | true | Escribe un registro de sesión por tarea. |
| `record_trajectories` | false | Exportación opcional de trayectorias de prompt completo para el filtrado de conjuntos de datos SFT/GRPO: un archivo JSON por tarea bajo `.pedacito/trajectories/`, que contiene los mensajes reales con forma ChatML/OpenAI (sistema, usuario, asistente, llamadas a herramientas y resultados de herramientas) más metadatos (`verify_passed`, `harness_version`, `edit_format`, `model_id`, `provider`, `task_hash`, `started_at`, `elapsed_s`, totales de tokens). A diferencia de `replay`, contiene prompts completos; cada cadena pasa por `safety.redact` y el archivo serializado se vuelve a escanear con `security.scan_text_for_secrets`, con un hallazgo residual enmascarado o la escritura rechazada. Solo local, sin red ni telemetría. `pdc ask --save-trajectory` fuerza la captura para una ejecución en memoria (entrada `sources` en vivo `--save-trajectory`) sin escribir la clave. Consulta [SECURITY.md — Exportación de trayectorias](SECURITY.md#exportación-de-trayectorias). |
| `review` | false | Usa siempre la página de revisión. |
| `fuzzy_match` | false | Ancla aproximadamente los bloques de búsqueda casi coincidentes. |
| `edit_style` | unset | `quote` (resumen por defecto) o `anchor` (prefiere `SYMBOL` + definición completa, SEARCH solo para ediciones pequeñas); sin definir deja que decida la tarjeta del modelo. |
| `creative_mode` | false | Modo creativo/de diseño (Builder) para CADA tarea: persona de diseñador, temperatura de respuesta ≥ 0.6 en servidores locales, un presupuesto de respuesta de ≥ 8192 tokens, reescrituras de HTML/CSS de archivo completo. `--creative`, `/creative on` de la TUI o el interruptor ✨ Creative de la GUI lo definen por ejecución. Consulta [Modo creativo / de diseño](USAGE.md#modo-creativo--de-diseño). |
| `creative_auto` | true | Detecta automáticamente las tareas de diseño (prompts de sitio web/portafolio/rediseño, objetivos `.html`/`.css`/`.svg`/`.jsx`/`.tsx`/`.vue`/`.svelte`) cuando `creative_mode` está desactivado. Ambos en false = ediciones quirúrgicas en todas partes (`/creative off`). |
| `creative_temperature_delta` | unset | Temperatura extra añadida a la línea base de respuesta en el modo creativo/de diseño (oscilación relativa, no un piso estático). Sin definir = automática por nivel (`+0.20` tiny/small, `+0.30` en el resto); un float explícito en `0.0`–`0.8` la anula. Limitada a `0.95` (`1.0` con un delta explícito) y nunca por debajo de la línea base, de modo que una Gemma calibrada en `0.6` sube a `0.9` mientras que un modelo tiny de `0.2` solo se empuja a `0.4`. |
| `post_edit_check` | true | Tras las ediciones aplicadas, devuelve solo los hallazgos NUEVOS del linter/sintaxis en los archivos modificados. |
| `auto_allowlist` | false | Restringe las escrituras de las tareas de `auto` a los archivos que nombra el título o que leyó la búsqueda (los archivos nuevos siempre se permiten). |
| `allow_write` | false | Escritura/anexado/movimiento/eliminación de archivo completo durante la recolección. Desde la configuración propia de un proyecto: se ignora hasta `pedacito trust`. |
| `allow_shell` | false | `run` y `diagnose`. Desde la configuración propia de un proyecto: se ignora hasta `pedacito trust`. |
| `allow_serve` / `serve_wait` | false / 3.0 | Servidores de desarrollo en segundo plano (acción `serve`, `pedacito serve`); segundos de espera por una URL impresa. Un servidor que imprime una más tarde igual la tiene registrada, en la siguiente consulta de `procs`/`logs`. |
| `allow_lsp` / `lsp_servers` / `lsp_timeout` | false / `{}` / 10 | Definición/referencias/hover/diagnósticos exactos mediante un servidor de lenguaje; los comandos por extensión anulan la detección en PATH. |
| `allow_screenshots` / `allow_browser` | false | Captura de pantalla y de navegador headless. Desde la configuración propia de un proyecto: se ignora hasta `pedacito trust`. |
| `vision_model` | `""` | Modelo multimodal usado para describir las capturas. |
| `modality` | `""` | Qué tipo de modelo sirve un perfil: `""` (autodetección por el id del modelo y los metadatos del servidor), `"text"`, `"image"`, `"audio"` o `"embedding"`. Solo `"image"` abre la ruta de [generación local de imágenes](#generación-local-de-imágenes-image_profile-allow_image_gen). |
| `image_profile` | `""` | La entrada `[profiles.*]` que sirve `/images/generations`. Vacío = usar el primer perfil de modelo de imagen encontrado. |
| `allow_image_gen` | false | Expone la acción opcional `generate_image` del agente. Desde la configuración propia de un proyecto: se ignora hasta `pedacito trust`. |
| `image_default_size` | `"1024x1024"` | `size` enviado a `/images/generations` cuando quien llama no indica ninguno. |
| `image_default_steps` | 0 | Pasos de difusión enviados cuando quien llama no indica ninguno (0 = el valor por defecto del propio servidor). |
| `semantic_search` | false | Alternativa con embeddings para `grep`. |
| `circuit_breaker` / `circuit_failures` / `circuit_cooldown` | true / 4 / 60 | Falla rápido tras fallos repetidos de transporte/servidor; los límites de tasa, los desbordamientos de contexto y los errores 4xx no lo activan. |
| `allow_diff` | true | Acción `diff` de solo lectura. |
| `disable_thinking` | false | Pide el modo sin razonamiento usando el interruptor de la familia del modelo; los bloques de salida se eliminan en cualquier caso. |
| `default_profile` | `""` | Perfil usado cuando no se da `--profile`. |
| `scout_profile` | `""` | Perfil cuyo cliente ejecuta los pasos de búsqueda de la recolección mientras el perfil activo (arquitecto) planifica y edita; vacío = sin scout. |
| `specialties` | `()` | En qué es bueno el modelo de este perfil, p. ej. `["coding", "testing_qa"]`; vacío = la tarjeta del modelo las detecta (heurísticas de nombre + biblioteca de arquitecturas + seis microsondas). Una lista explícita siempre gana sobre la tarjeta. Consulta [Enrutamiento automático por especialista](ARCHITECTURE.md#enrutamiento-automático-por-especialista). |
| `auto_route_specialties` | false | Enruta cada tarea al perfil especialista que mejor coincida en los límites de las tareas (transferencia limpia de VRAM en un servidor local); `--auto-route` / `--no-auto-route` lo alternan por ejecución. |
| `prompt_pack` | `""` | Densidad de instrucciones `tiny` / `small` / `full`; vacío = tarjeta del modelo y luego nivel. |
| `answer_mode` | `""` | `rebuild` / `continue` / `auto`; vacío = tarjeta del modelo y luego `auto` (alojado) o `rebuild` (local). |
| `compaction` | `"deterministic"` | Cómo se resumen las búsquedas plegadas: `deterministic`, `model` u `off` (recorte heredado por turno). |
| `rank_map` | true | Ordena el mapa para la tarea cuando no cabe completo. |
| `repo_brief` / `max_brief_tokens` | true / 500 | Bloque de orientación determinista sobre el mapa. |
| `verify_answer` | `"deterministic"` | `off` / `deterministic` / `full`: verifica las ediciones antes de escribirlas. |
| `best_of` | 1 | Muestrea N respuestas candidatas de edición y conserva la de mejor puntuación. |
| `verify_tests` / `test_command` | false / `""` | Ejecuta las pruebas afectadas tras las ediciones aplicadas (necesita `allow_shell`); `test_command` es la anulación de la detección del ejecutor. |
| `auto_verify` | `""` | Cadena de comando que se ejecuta tras cada tarea de Auto; un fallo devuelve su salida para reparación. Vacío = desactivado; nunca un bool. El `/verify on` de la TUI y **Post-edit verify** de la GUI lo detectan y lo completan. `pdc ask --verify-cmd "..."` lo anula para una ejecución, en memoria (consulta [USAGE.md](USAGE.md#modo-headless-y-automatización)). |
| `auto_verify_final` | false | Ejecuta `auto_verify` una vez tras la última tarea sin marcar del plan en lugar de tras cada tarea (útil cuando toda la suite no puede pasar a mitad del plan). |
| `auto_repair_rounds` / `auto_verify_rounds` | 2 / 2 | Rondas de reparación por verificación fallida por tarea; rondas de verificación en tiempo de ejecución tras una ejecución `auto --apply`. |
| `static_gate` / `static_gate_timeout` | true / 3.0 | Los archivos de Python modificados reciben una verificación estática con ruff o AST (solo errores nuevos) antes de cualquier subproceso de pruebas; segundos permitidos para una ejecución de ruff. |
| `auto_green_ratchet` / `auto_keep_checkpoints` | true / 25 | Guarda una instantánea de cada paso de Auto y revierte un paso que aún falla tras la reparación al último espacio de trabajo verificado en verde; instantáneas verificadas en verde conservadas por proyecto. |
| `interface_lock` | true | Inyecta las firmas públicas en las tareas de Auto y condiciona cada edición a la existencia de las importaciones locales y la aridad de las llamadas. |
| `verify_dom` / `dom_entry` / `dom_timeout` | false / `""` / 45 | Verificación de humo del DOM headless tras las ediciones web aplicadas (necesita `allow_shell` y la instalación opcional de Playwright): carga la página de entrada en Chromium headless, comprueba que el grafo de módulos se monta y compara la superficie montada con una instantánea previa a la edición. `dom_entry` anula el HTML de entrada detectado; `dom_timeout` limita una carga de página en segundos. `ask --verify-dom` lo alterna por ejecución; `pedacito domcheck` lo ejecuta a pedido. |
| `auto_smoke` / `auto_smoke_timeout` | true / 20 | Tras una ejecución `auto --apply`, ejercita el proyecto una vez (el comando `--verify` y/o un arranque de sus puntos de entrada de Python) y devuelve un fallo para reparación. `verify: passed` demuestra que el proyecto ARRANCA; no es la suite de pruebas; termina los planes con una tarea de ejecutar las pruebas o pasa `--verify`. |
| `use_memory` / `use_skills` | true | Empuja la memoria, las reglas y el catálogo de skills. |
| `allow_git` / `git_context` | true | Acción `git` de solo lectura; preámbulo de rama/sucio/commits recientes. |
| `allow_subagents` | false | La acción `task` (auxiliar de solo lectura con una ventana nueva). |
| `allow_fetch` / `fetch_allowed_hosts` | false / `[]` | La acción `fetch` y los hosts que puede leer. |
| `roles` | `{}` | Perfiles por trabajo: `planner`, `verifier`, `summarizer`, `repair`, `escalate`, `fallback`. |
| `auto_escalate` / `escalation_zero_edit_runs` | true / 3 | Amplía el reintento de `escalate` más allá de "no aterrizó ninguna edición": un fallo semántico de reparación, dos arrendamientos sin progreso, o esta cantidad de ejecuciones consecutivas sin ediciones en un perfil ganan un reintento incluso cuando se aplicaron ediciones. Cada razón emite un evento `escalation_suggested` incluso sin un rol configurado; 0 desactiva la estadística de ejecuciones sin ediciones. |
| `hooks` / `hook_timeout` | `{}` / 10 | Tus comandos en eventos, cada uno limitado a esta cantidad de segundos (consulta [Hooks](#hooks); los hooks a nivel de proyecto necesitan `pedacito trust`). |
| `scrub_env` / `env_passthrough` | true / `[]` | Entorno mínimo para los comandos elegidos por el modelo. |
| `protected_paths` / `shell_deny` | `[]` | Rutas en las que el modelo no puede escribir; patrones de comandos que `run` rechaza. |
| `wrap_observations` / `redact_secrets` | true | Delimita la salida no confiable; enmascara secretos en registros y sesiones. |
| `max_total_tokens` / `max_cost` | 0 | Se detiene antes de la siguiente llamada una vez gastado (0 = sin límite). |
| `price_input_per_mtok` / `price_output_per_mtok` / `price_cached_per_mtok` / `price_cache_write_per_mtok` | 0 | USD por millón de tokens, para el informe de costos; el precio de escritura cubre los tokens de creación de caché explícita. |
| `prompt_cache` / `prompt_cache_ttl` / `prompt_cache_key` | `auto` / `5m` / `""` | Caché de prompts alojada (consulta [Caché de prompts alojada](#caché-de-prompts-alojada)): `auto` deja que decida la tabla de proveedores, `implicit`/`explicit`/`off` la anulan, `prompt_cache_key = "auto"` deriva una clave de afinidad de caché estable. |
| `enable_prompt_cache` / `enable_prompt_lookup` / `prompt_lookup_num_tokens` | true / false / 10 | Aceleración del servidor local (consulta [Aceleración de inferencia local](#aceleración-de-inferencia-local)): envía `cache_prompt` y/o `prompt_lookup_num_tokens` solo a endpoints autoalojados; un rechazo 400/422 elimina la clave y se recuerda por servidor+modelo. |
| `enable_mcp` / `mcp_servers` | true / `{}` | El puente MCP de servidores con nombre (consulta [Servidores MCP con nombre](#servidores-mcp-con-nombre)): tablas `[mcp_servers.<name>]` con `command`/`args`/`url`/`env`/`enabled`/`trusted`; la lista heredada `mcp_servers = ["cmd", ...]` sigue alimentando la acción `mcp`. Los servidores definidos por el `.pedacito.toml` de un proyecto no confiable se ignoran. |
| `stuck_recovery` | true | Un prompt de recuperación dirigido antes de que la recolección se rinda. |
| `use_model_card` | true | Deja que la tarjeta de `pedacito calibrate` complete los ajustes sin definir. |
| `show_telemetry` | true | Cinta de telemetría de una línea en stderr tras cada tarea (barra de contexto, tokens del turno, velocidad, aciertos de caché); nunca se muestra en los modos `--jsonl` / rpc. Consulta [Telemetría en vivo](USAGE.md#telemetría-en-vivo). |
| `check_for_updates` | true | Verificación discreta de versiones en PyPI al iniciar: una petición GET anónima (tiempo límite de 2 s, en caché 24 h junto a la configuración de usuario), un banner descartable en la GUI con actualización en streaming + Restart, `pedacito update [--run]`, `/update` de la TUI. `false` = cero llamadas de red y cero subprocesos de actualización en todas partes. Consulta [Actualización](INSTALLATION.md#actualización). |
| `update_channel` | `"stable"` | Qué versiones de PyPI cuentan como "la más reciente": `"stable"` ignora las etiquetas pre/dev/rc/alpha/beta; `"any"` las incluye. `PEDACITO_UPDATE_CHANNEL` la anula como cualquier otro ajuste. Consulta [Actualización](INSTALLATION.md#actualización). |
| `log_events` / `keep_events` | true / 20 | Telemetría JSONL local bajo `.pedacito/events/`. |
| `summary_workers` / `refresh_summaries` | 0 / 12 | Resúmenes concurrentes (0 = automático: alojado 4, local 1); unidades vueltas a resumir tras una tarea. |
| `summary_batch` | 1 | Unidades cortas resumidas N por petición durante la indexación (menos viajes de ida y vuelta; las líneas sin analizar recurren a peticiones individuales). |
| `max_observation_tokens` | 0 | Techo de una observación que no es de lectura, en tokens (0 = automático: un cuarto de la ventana, 300-2200). |
| `plan_threshold` / `max_replans` | 0 / 1 | Planifica automáticamente las tareas que nombran esta cantidad de archivos o requisitos; las tareas que nombran 5+ archivos se planifican automáticamente incluso con 0 a menos que esto se defina explícitamente (`--plan` fuerza una tarea, `--no-plan` la excluye); revisiones del planificador tras un paso fallido. |
| `allow_steering` | true | Entrega al modelo en `chat` las líneas escritas a mitad de una tarea. |
| `tool_mode` | `grammar` | `native` envía los pasos de búsqueda por el canal de llamadas a herramientas del modelo; `calibrate` elige. |
| `execution_mode` | `two_phase` | El motor de tareas: `two_phase` (recopilar y luego responder), `interleaved` (una conversación agéntica de leer/editar/probar) o `auto` (interleaved para APIs alojadas, servidores local-MoE y modelos locales grandes calibrados cuya tarjeta eligió llamadas nativas a herramientas; `two_phase` para modelos locales tiny/small/medium y sin calibrar). `--execution-mode` lo sobrescribe por ejecución; `/mode` en la TUI lo cambia en vivo. |
| `step_reasoning_first` | true | `true` coloca `reasoning` antes de la acción en cada paso de búsqueda (el modelo dice qué necesita y luego elige); sin definir deja que decida la tarjeta del modelo. `calibrate` mide ambos órdenes con la misma regla de adoptar-solo-si-es-0.1-mejor que `tool_mode`. |
| `fewshot_steps` | unset | `true` muestra tres pasos resueltos de tu proyecto (grep de un símbolo, leerlo, done) una vez en el primer turno del usuario, incluso con gramática o llamadas nativas a herramientas; sin definir deja que decida la tarjeta del modelo. Unos 70 tokens de prompt extra en cada llamada de búsqueda (el servidor los guarda en caché tras la primera, pero un conteo de tokens de prompt como el de `eval` aún los incluye en cada llamada). `calibrate` lo adopta solo si elige la acción correcta 0.1 más a menudo sin pérdida de pasos bien formados. |
| `edit_plan_brief` | true | El resumen de edición pide 1-3 líneas de plan antes de los bloques (archivo, líneas o símbolo, por qué) y ningún otro texto entre los bloques ni dentro de ellos. Una sobrescritura de prompt (`edit_format.md`, `answer.md`) lo omite. |
| `sandbox` / `sandbox_network` | true / false | Sandbox del sistema operativo para `run` y los comandos de prueba; permite la red dentro de él. |
| `sandbox_writable` / `sandbox_strict` | `[]` / false | Directorios escribibles extra; se niega a ejecutar cuando no se puede aplicar ningún sandbox. |
| `atomic_edits` | false | Escribe una respuesta de varios archivos todo o nada (cualquier fallo no escribe nada). |

Todos los demás ajustes públicos se listan a continuación, agrupados por subsistema, con su valor por defecto real.
`pedacito config --reference` imprime las mismas claves como una plantilla anotada.

<details>
<summary>Referencia completa de ajustes avanzados (Conexión, presupuestos y selectores de perfil)</summary>

| Ajuste | Por defecto | Significado |
|---|---|---|
| `timeout` | `600.0` | Segundos de espera de una petición antes de fallar. |
| `max_concurrency` | `0` | Peticiones concurrentes que este endpoint puede atender. |
| `max_consecutive_failures` | `3` | Detiene un lote tras esta cantidad de fallos seguidos. |
| `max_tokens_summary` | `220` | Presupuesto de tokens de salida para una llamada de resumen de indexación. |
| `max_tokens_step` | `700` | Una llamada a herramienta más una nota breve; el esquema pone la acción primero, de modo que una respuesta truncada pierde solo la explicación. |
| `max_tokens_answer` | `3000` | Presupuesto de tokens de salida para la fase de respuesta. |
| `max_total_model_calls` | `30` | Límites de seguridad globales independientes de la acción que siguen aplicando cuando max_steps = 0 elimina el tope de búsquedas: un techo duro de invocaciones totales al LLM por tarea, y un techo de pasos consecutivos de recolección que no muestran progreso (ad... |
| `max_no_progress_steps` | `4` | Pasos consecutivos de recolección que no agregan código ni símbolos nuevos antes de que la recolección termine limpiamente. |
| `max_task_completion_tokens` | `60000` | Protección de presupuesto a nivel de tarea (BUG-218): techos de tokens de completado totales y segundos de reloj de UNA tarea, recolección + respuesta + reparación juntas. |
| `max_task_seconds` | `900` | Techo de reloj de una tarea; 0 lo desactiva. |
| `max_output_tokens` | `0` | Techo de SALIDA por mensaje que respeta el proveedor (las APIs alojadas limitan una respuesta muy por debajo de su ventana de contexto, p. ej. ~8k tokens). |
| `reserve_tokens` | `2048` | Tokens que se reservan de context_tokens para la respuesta misma; auto_fit lo reduce cuando la ventana es demasiado pequeña. |
| `request_max_tokens_cap` | `32768` | max_tokens enviado en las llamadas de generación; auto_fit lo ajusta a la ventana. |
| `max_tokens_plan` | `2000` | Presupuestos de las dos llamadas de andamiaje sin presupuesto de respuesta existente. |
| `max_tokens_build` | `8000` | Presupuesto de tokens de salida para la llamada de construcción del andamiaje. |
| `stream_usage` | `true` | Pide al servidor que informe el uso también en las respuestas en streaming (`stream_options.include_usage` de OpenAI); sin él, los números de tokens, costo y aciertos de caché de la fase de respuesta son desconocidos. |
| `language` | `"en"` | Idioma de la interfaz: "en" o "es". |
| `llm_reply_language` | `"match_ui"` | En qué idioma escribe el modelo sus explicaciones en prosa. |
| `benchmarks_dir` | `""` | Carpeta padre de los proyectos de benchmark de modelos integrados. |
| `profiles` | `{}` | Perfiles de modelo con nombre (tablas `[profiles.NAME]`); administrados por la pestaña Profiles y `pedacito models`. |
| `active_profile` | `''` | Nombre del perfil seleccionado para esta ejecución (definido por `--profile` / `default_profile`). |
| `active_tier` | `''` | Nivel de presupuestos resuelto para esta ejecución (a partir de `tier` o del nombre del modelo). |
| `model_size_bytes` | `0` | Tamaño de los pesos del modelo informado por el servidor, usado por la estimación de huella; 0 cuando se desconoce. |

</details>

<details>
<summary>Referencia completa de ajustes avanzados (Indexación y fragmentación)</summary>

| Ajuste | Por defecto | Significado |
|---|---|---|
| `file_card_line_ranges` | `true` | Adjunta L<inicio>-<fin> a los nombres en las tarjetas de archivo (ahorra una llamada a `outline`). |
| `class_split_lines` | `60` | Las clases más cortas que esto se indexan como un solo fragmento en lugar de por método. |
| `max_unit_lines` | `220` | Una función/método (o preámbulo/bloque final de módulo) mayor que cualquiera de los límites se divide en los límites de sentencias de nivel superior en name#2, name#3, ... |
| `max_unit_chars` | `12000` | Divide una función, método o preámbulo mayor que esta cantidad de caracteres en name#2, name#3, ... |
| `max_unit_tokens` | `3000` | Protección de tokens (~3.6 caracteres/token) para contextos pequeños; 0 la desactiva. |
| `unit_overlap_lines` | `0` | Líneas repetidas entre ventanas de líneas cuando una sentencia gigante debe dividirse en ventanas. |
| `max_chunk_chars` | `6000` | A los fragmentos mayores que esto se les trunca el código fuente antes de resumirlos. |
| `min_lines_to_summarise` | `4` | Omite resumir fragmentos trivialmente cortos; la firma dice lo suficiente. |
| `summarise_methods` | `false` | Racionamiento de resúmenes: un docstring ya es un resumen, y un método está cubierto por el resumen de su clase más su firma. |
| `summarise_documented` | `false` | También resume las unidades que ya tienen un docstring. |
| `include_references` | `true` | Indexa de forma estructural los archivos que no son de Python (.md .csv .json .toml .yaml ...): el mapa muestra las columnas de un CSV o las secciones de una configuración; el contenido nunca se incorpora al mapa. |
| `extra_extensions` | `[]` | Extensiones extra que tratar como archivos de referencia, p. ej. [".jinja", ".proto"]. |
| `max_reference_bytes` | `2000000` | Por encima de este tamaño un archivo de referencia se registra solo por nombre y encabezado. |
| `summarise_references` | `false` | Resume los archivos de referencia (markdown, JSON, ...) en lugar de listar solo su estructura. |
| `use_project_instructions` | `true` | Incluye el PEDACITO.md / AGENTS.md / CLAUDE.md del repositorio en cada prompt. |
| `manage_gitignore` | `true` | En la primera ejecución, agrega .pedacito/ al .gitignore del proyecto. |
| `create_ignore_file` | `true` | En la primera ejecución, crea un .pedacitoignore comentado. |
| `use_ripgrep` | `true` | Usa ripgrep para `grep` cuando está instalado (mismo formato de salida). |

</details>

<details>
<summary>Referencia completa de ajustes avanzados (Recolección y arrendamientos del ciclo)</summary>

| Ajuste | Por defecto | Significado |
|---|---|---|
| `max_todo_chars` | `1500` | Tope de la lista de verificación propia del modelo cuando se inyecta en un prompt. |
| `dynamic_tool_pruning` | `true` | Poda dinámica de esquemas de herramientas: mantiene siempre disponibles las acciones centrales de navegación, pero omite del esquema del prompt una acción opcional/especializada habilitada una vez que las estadísticas de uso de herramientas del repositorio muestran que no se ha usado durante `... |
| `tool_prune_after_tasks` | `10` | Tareas consecutivas que una herramienta puede pasar sin usarse antes de que la poda la considere. |
| `max_completion_rounds` | `0` | Límite de turnos de respuesta de seguimiento cuando el modelo aplaza el trabajo restante; 0 = sin límite (se detiene con NOTHING MORE o cuando no se producen bloques). |
| `max_continuation_rounds` | `0` | Límite de turnos de continuación cuando una respuesta se corta a mitad de un bloque; 0 = sin límite (el presupuesto de caracteres de continuación es el límite real). |
| `max_rehydration_rounds` | `2` | Rondas de rehidratación de tramo exacto `NEED path/to/file.py:120-168` permitidas por generación de respuesta, reparación o verificador: el arnés obtiene SOLO esos tramos (verificados por hash, recortados por tokens como un `read`), los ANEXA al... |
| `max_read_lines` | `200` | Máximo de líneas devueltas por una llamada `read`. |
| `read_context_hint` | `true` | Adjunta la firma contenedora y las importaciones del módulo cuando un `read` comienza a mitad de archivo. |
| `read_skeleton` | `true` | Cuando un `read` apunta a una clase/módulo mayor que el presupuesto de lectura, devuelve un esqueleto AST determinista (firmas + docstrings + punteros [lines X-Y], cuerpos plegados) en lugar de la primera página de la unidad gigante; contin... |
| `compact_step_schema` | `false` | EXPERIMENTAL: exige solo `action` y `reasoning` en el esquema del paso de recolección para que la decodificación restringida por gramática pueda omitir campos sin usar (~20-40 tokens de salida por paso). |
| `fold_tail_steps` | `0` | Pasos de recolección recientes que se conservan textuales cuando los más antiguos se pliegan en un resumen STATE por presión de la ventana; 0 = elegido según el nivel. |
| `max_memory_chars` | `1500` | Tope del texto de memoria del proyecto y del usuario incluido en los prompts. |
| `max_skill_catalog_chars` | `900` | Tope del bloque del catálogo de skills en los prompts. |
| `subagent_max_steps` | `8` | Límite de pasos de búsqueda para un subagente `task`. |
| `max_fetch_chars` | `6000` | Tope del texto de página obtenida que se devuelve al modelo. |

</details>

<details>
<summary>Referencia completa de ajustes avanzados (Edición y reparación)</summary>

| Ajuste | Por defecto | Significado |
|---|---|---|
| `auto_diagnostics` | `false` | Tras cada tarea de `auto` (cuando no se define ningún comando --verify), ejecuta el linter/verificador de tipos detectado del proyecto y devuelve los fallos a las rondas de reparación. |
| `diagnostics_timeout` | `90` | Segundos permitidos para una ejecución del linter/verificador de tipos de auto-diagnostics. |
| `max_diagnostics_chars` | `8000` | Tope del texto de diagnósticos devuelto a la reparación. |
| `max_write_chars` | `200000` | Techo del contenido de una llamada `write`, en caracteres. |
| `max_write_lines` | `120` | Máximo de líneas que una respuesta puede emitir para un archivo (`write` de recolección o bloque de edición de la fase de respuesta). |
| `allow_rename` | `true` | Trata la creación de una nueva ruta emparejada con un bloque que vacía un archivo existente de la misma extensión como un movimiento real (el archivo antiguo se elimina). |
| `fuzzy_min_ratio` | `0.85` | Similitud mínima para un ancla de edición aproximada; por debajo de esto la edición falla en lugar de adivinar. |
| `repair_rounds` | `2` | Viajes de ida y vuelta de reparación por aplicación cuando un bloque de búsqueda/reemplazo no coincide. |
| `incremental_edit_loop` | `true` | Microciclos incrementales de edición-verificación anclados a símbolos. |
| `max_new_files` | `24` | Tope de archivos que un plan puede proponer; el exceso se informa. |
| `max_spec_chars` | `60000` | Los documentos de especificación más largos que esto se truncan, con una advertencia. |
| `test_timeout` | `300` | Segundos permitidos para una ejecución de comando de pruebas. |

</details>

<details>
<summary>Referencia completa de ajustes avanzados (Ciclo Auto y checkpoints)</summary>

| Ajuste | Por defecto | Significado |
|---|---|---|
| `auto_plan_file` | `"PLAN.md"` | Una meta se convierte en una lista de verificación; cada tarea se implementa en un contexto nuevo con el archivo del plan como único estado que sobrevive. |
| `auto_max_tasks` | `0` | Tope de tareas por ejecución de `auto`; 0 = sin tope. |
| `auto_max_minutes` | `0` | Tope de reloj por ejecución de `auto` en minutos; 0 = sin tope. |
| `auto_checkpoint_max_file_bytes` | `1000000` | Límites de instantánea por paso: un archivo mayor que esto se registra pero no se copia (la reversión lo deja en paz); el recorrido se detiene tras esta cantidad de archivos, de modo que una instantánea no pueda llenar el disco en un monorepo con código de terceros. |
| `auto_checkpoint_max_files` | `1500` | Máximo de archivos registrados en una instantánea de paso de auto; el recorrido se detiene allí. |
| `auto_split_lines` | `120` | `new --complex` / `auto`: un archivo fuente cuyo stub de andamiaje es más largo que esto se divide en varias tareas (una por grupo de definiciones de nivel superior). |
| `auto_install_deps` | `false` | Instala con pip un módulo de terceros faltante en el intérprete en ejecución cuando --verify falla por él, y luego reintenta. |
| `review_port` | `0` | Puerto de la página de revisión; 0 deja que el sistema operativo elija uno libre. |
| `open_browser` | `true` | Abre automáticamente la página de revisión en un navegador. |
| `review_timeout` | `1800` | Segundos antes de que una página de revisión sin respuesta expire. |
| `hybrid_routing` | `false` | Ejecuta cada tarea en dos etapas: un modelo arquitecto potente planifica primero (causa raíz, archivos objetivo, firmas exactas, casos límite), luego el modelo editor activo implementa el plano. |
| `architect_provider` | `""` | El perfil cuyo servidor/modelo planifica en la Etapa 1 híbrida. |
| `architect_model` | `""` | El id del modelo en el que corre el arquitecto. |
| `auto_route_allow_hosted` | `false` | Permite que el enrutamiento por especialista elija un perfil ALOJADO (API remota). |

</details>

<details>
<summary>Referencia completa de ajustes avanzados (Web, navegador, visión y semántica)</summary>

| Ajuste | Por defecto | Significado |
|---|---|---|
| `allow_mcp` | `false` | Cliente MCP opcional: expone una acción `mcp` más un catálogo breve de herramientas. |
| `mcp_timeout` | `30` | Segundos permitidos para una petición a un servidor MCP. |
| `screenshot_wait_seconds` | `5` | Segundos de espera tras el lanzamiento antes de capturar; el valor `start` de la acción lo anula. |
| `browser_wait_seconds` | `3` | Segundos de espera tras la navegación/interacción antes de capturar. |
| `browser_server_wait_seconds` | `15` | Segundos de espera a que un servidor lanzado imprima su URL de localhost. |
| `browser_headless` | `true` | Ejecuta la acción de navegador en modo headless (sin ventana visible). |
| `browser_timeout_seconds` | `20` | Tiempo límite por navegación/acción en segundos. |
| `enable_web` | `false` | Permite al agente obtener URL o buscar documentación en línea. |
| `vision_max_tokens` | `400` | Tope de tokens de salida para la descripción de capturas del modelo de visión. |
| `embedding_model` | `""` | Id del modelo de embeddings; vacío = el modelo de chat. |
| `semantic_top_k` | `5` | Cantidad de unidades indexadas más cercanas que informa la búsqueda semántica. |
| `semantic_min_score` | `0.2` | La similitud coseno por debajo de esto no se muestra; las escalas difieren según el modelo. |
| `browser_eval_hosts` | `[localhost, 127.0.0.1, ::1, 0.0.0.0]` | Hosts donde la acción `eval` del navegador puede ejecutar JavaScript escrito por el modelo. |

</details>

<details>
<summary>Referencia completa de ajustes avanzados (Seguridad, sandbox y retención)</summary>

| Ajuste | Por defecto | Significado |
|---|---|---|
| `shell_timeout` | `120` | Segundos antes de que se mate un comando `run`. |
| `shell_idle_seconds` | `10` | Mata un comando `run` tras esta cantidad de segundos SIN salida (evita que las aplicaciones GUI y los servidores bloqueen al agente); 0 desactiva la regla de inactividad. |
| `shell_allowlist` | `[]` | Lista permitida opcional de ejecutables para `run`, p. ej. ["pytest", "python"]; vacío = cualquier ejecutable una vez que allow_shell está activado. |
| `shell_allowlist_by_project` | `{}` | Reglas de permiso de shell por proyecto para `run`: una tabla que asigna raíces absolutas de proyecto a listas de patrones, p. ej. { "/home/me/proj" = ["pytest *", "cargo test *"] }. Un comando que coincide omite la solicitud de permiso solo en ese proyecto. Las reglas nunca cubren instalaciones de paquetes, shells ni comandos con metacaracteres de encadenamiento (punto y coma, ampersands, tuberías, comillas invertidas, sustitución de comandos). La respuesta «permitir en este proyecto» de la GUI/TUI las escribe aquí. Sujeta a confianza como `shell_allowlist`: las reglas de una configuración de proyecto solo se aplican tras `pedacito trust`. |
| `max_shell_output_chars` | `8000` | Tope de la salida de comandos devuelta al modelo, en caracteres. |
| `security_outbound_mode` | `"warn"` | Protección saliente hacia proveedores en la nube: antes de que una petición salga hacia un endpoint alojado, escanea sus mensajes en busca de secretos no reconocidos. |
| `keep_sessions` | `0` | Valores por defecto de `pedacito prune` (0 = conservar todo). |
| `keep_backups` | `0` | Respaldos que conserva `pedacito prune`; 0 conserva todo. |
| `log_max_bytes` | `2000000` | Rota el registro de eventos tras esta cantidad de bytes. |

</details>
