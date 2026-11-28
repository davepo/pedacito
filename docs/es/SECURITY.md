🌐 [English](../SECURITY.md) | **Español**

# Seguridad: arnés de programación con IA y buenas prácticas para agentes

Pedacito es un agente de programación local primero. Un modelo lee tus archivos, propone ediciones
y puede pedir ejecutar comandos. Esta guía explica cómo Pedacito contiene ese
comportamiento, de qué **no** protege y qué deberías configurar.

Está escrita para dos públicos: quienes deciden si confiar en un agente en
su máquina, y quienes construyen su propio arnés y quieren una lista de verificación de
patrones que funcionan.

Para ver cómo encajan los hooks, los servidores MCP y las skills en el modelo de confianza, consulta
[SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md). Para contribuir con correcciones, consulta [CONTRIBUTING.md](CONTRIBUTING.md).

## Modelo de amenazas

El agente es tan confiable como el texto menos confiable que lee. Pedacito
asume que cada una de estas fuentes puede ser hostil o simplemente estar equivocada:

| Fuente | Riesgo |
|---|---|
| **Salida del modelo** | Propone una ruta fuera del proyecto, un comando de shell destructivo o una edición en el archivo equivocado. |
| **Contenido del repositorio** | Un `.pedacito.toml`, un `Makefile` o un archivo de pruebas pueden contener comandos o instrucciones. Los symlinks pueden apuntar fuera del proyecto. |
| **Salida de herramientas** | La salida de comandos, las páginas web, las respuestas de MCP y los resultados de LSP pueden contener texto del tipo "ignora las instrucciones anteriores". |
| **Secretos en el espacio de trabajo** | Los archivos `.env` y las claves privadas pueden incorporarse a un prompt y enviarse a un proveedor alojado. |
| **Otros sitios web** | Una página abierta en tu navegador puede intentar llegar al servidor local de la GUI. |

Las defensas se agrupan en cuatro pilares. Cada pilar limita el daño si falla la capa
superior.

| Pilar | Pregunta que responde | Código principal | Por defecto |
|---|---|---|---|
| 1. Aislamiento de credenciales y escaneo previo | ¿Puede un secreto salir de la máquina? | `pedacito/security.py`, `pedacito/safety.py`, `pedacito/llm.py` | Activado (`warn`) |
| 2. Sandbox estricto del espacio de trabajo | ¿Puede una ruta escapar del proyecto? | `security.resolve_safe_path`, `pedacito/index.py`, manejadores de herramientas | Siempre activado |
| 3. Ejecución con mínimos privilegios | ¿Qué puede hacer un comando si se ejecuta? | `pedacito/sandbox.py`, `pedacito/proc.py`, `safety.child_env` | Sandbox activado, shell desactivado |
| 4. Registro de auditoría y viaje en el tiempo | ¿Se puede ver y revertir cada cambio? | `pedacito/checkpoints.py` | Activado en cada ejecución |

Pedacito es defensa en profundidad, no una garantía contra un atacante decidido ni contra un
error del kernel. Lee [Límites conocidos y brechas](#límites-conocidos-y-brechas) antes de depender de él
para código no confiable.

---

## Pilar 1: aislamiento de credenciales y escaneo previo

El objetivo es que una clave presente en tu repositorio no llegue por accidente a un proveedor
de modelos alojado, a un archivo de registro ni a un commit de git.

### Qué se detecta

Pedacito reconoce estas formas de credenciales. Los patrones están en
`pedacito/security.py` (`_TOKEN_RULES`, `_GENERIC_ASSIGNMENT_RE`) y
`pedacito/safety.py` (`_TOKEN_RES`, `_PRIVATE_KEY_RE`):

| Tipo | Patrón (simplificado) |
|---|---|
| Claves `sk-` de OpenAI, Anthropic y similares | `sk-` seguido de 20 o más de `[A-Za-z0-9_-]` (también `sk-proj-`, `sk-ant-`, `sk-live-`, `sk-or-v1-`) |
| Tokens de PyPI | `pypi-` seguido de 20 o más caracteres |
| Tokens de GitHub | `ghp_`, `gho_`, `ghu_`, `ghs_`, `ghr_` con 30 o más caracteres, y `github_pat_` |
| Tokens de GitLab | `glpat-` con 16 o más caracteres |
| IDs de clave de acceso de AWS | `AKIA` seguido de 16 letras mayúsculas y dígitos |
| Claves de API de Google | `AIza` seguido de 35 caracteres |
| Tokens de Slack | `xox[baprs]-` con 10 o más caracteres |
| Tokens de Hugging Face | `hf_` con 30 o más caracteres |
| JWT | tres segmentos base64url separados por puntos que empiezan con `eyJ` |
| Claves privadas PEM | `-----BEGIN ... PRIVATE KEY-----` hasta su línea `END` (o el fin del texto) |
| Asignaciones genéricas | `password`, `secret`, `token`, `api_key`, `authorization` y nombres similares, a los que se asigna un valor entre comillas de 16 o más caracteres |

Los formatos de token específicos ganan a la regla de asignación genérica cuando ambos coinciden con el
mismo texto.

### Cómo se filtran los valores

Pedacito no calcula la entropía de Shannon. No tiene un umbral de entropía, así que una cadena larga
aleatoria bajo un nombre no relacionado no se marca. En su lugar aplica tres
filtros baratos que reducen los falsos positivos:

- **Pisos de longitud.** Los valores de asignación genérica deben tener al menos 16 caracteres
  (en `safety.redact`, al menos 12), sin espacios.
- **Detección de marcadores de posición.** `_looks_like_placeholder` ignora valores como
  `your_key_here`, `changeme`, `<token>`, `${API_KEY}`, `os.environ[...]`,
  `process.env.X`, `xxxx` y valores que contienen `...` o `…`. Los marcadores de prueba y
  ejemplo (`test_`, `fake_`, `sample`, `dummy`) también se ignoran.
- **Detección de relleno.** `safety.redact` omite un valor entre comillas que carece de una letra
  o un dígito **y** tiene menos de seis caracteres distintos. Esto elimina
  rellenos repetidos como `aaaaaaaaaaaaaaaa`.

Si necesitas una puntuación de entropía para un formato de token personalizado, agrégala como una nueva regla en
`_TOKEN_RULES` y pruébala con las formas reales de tus claves.

### Dónde se ejecuta el escaneo

| Punto | Qué ocurre | Ajuste |
|---|---|---|
| **Antes de cada petición a un servicio alojado** | Pedacito escanea los mensajes salientes en busca de secretos no reconocidos. Los endpoints de loopback y LAN nunca se escanean, porque esas claves no salen de la máquina. | `security_outbound_mode` |
| **Escaneo del espacio de trabajo** | `scan_workspace_secrets` revisa los archivos indexados. Los resultados aparecen como un aviso posterior a la ejecución y en el estado de seguridad de la GUI (`/security` en la TUI). | Siempre activado |
| **Higiene de git** | `check_git_secret_hygiene` informa los archivos sensibles que existen en el repositorio pero que no están excluidos por `.gitignore`. `ensure_gitignore_entries` puede agregar las líneas que faltan. | `manage_gitignore` |
| **Indexación** | Los archivos sensibles nunca se indexan (ver abajo). | Siempre activado |
| **Archivos de sesión, registros, eventos** | Los valores que parecen secretos se enmascaran antes de escribirse en `.pedacito/sessions/`, `log.txt`, los eventos y los checkpoints. | `redact_secrets` (activado) |

`security_outbound_mode` admite tres valores:

- `warn` (por defecto). La GUI y la TUI preguntan si **Acknowledge and send**,
  **Redact and send** o **Cancel**. Una ejecución de CLI no interactiva redacta automáticamente.
- `redact`. Cada secreto detectado se reemplaza por `<REDACTED_SECRET>` antes de enviar la
  petición.
- `allow`. Los mensajes se envían sin cambios. Úsalo solo si aceptas el riesgo.

Los hallazgos nunca exponen el valor sin procesar. Un hallazgo incluye el archivo, la línea, el tipo, una
vista previa enmascarada (por ejemplo `sk-ab...wxyz`) y una huella SHA-256. Solo la
vista previa y la huella se serializan. Cuando reconoces un hallazgo, su
huella se guarda en `.pedacito/security_ack.json`, de modo que la misma clave no se
vuelve a señalar.

### Los archivos sensibles nunca se indexan

Estos archivos se excluyen del índice y de las lecturas del modelo a menos que apruebes una
lectura específica en el diálogo de permisos:

- `.env`, `.env.local`, `.env.production` y cualquier archivo `*.env`. Las plantillas
  (`.env.example`, `.env.sample`, `.env.template`, `.env.dist`, `.env.defaults`)
  siguen siendo legibles.
- Claves privadas y contenedores de claves: `*.pem`, `*.key`, `*.p12`, `*.pfx`, `*.jks`,
  `*.kdbx`, `id_rsa*`, `id_dsa*`, `id_ecdsa*`, `id_ed25519*`, `id_sk_ed25519*`.
  Las claves públicas (`*.pub`) siguen siendo legibles.
- Archivos de credenciales: `.npmrc`, `.pypirc`, `.netrc`, `.git-credentials`,
  `credentials.json`/`.yaml`/`.yml`, `secrets.json`/`.yaml`/`.yml`,
  `*.credentials` y `service-account*.json`.
- Cualquier cosa bajo `.ssh`, `.gnupg`, `.aws`, `.docker` o `.config/gcloud`.

Vuelve a incluir un archivo deliberadamente con `!name` en `.pedacitoignore`, o reconsidera
si debería estar en el repositorio.

### Lista de verificación

- Mantén las claves fuera del repositorio. Guarda `api_key` en tu configuración de usuario, que
  vive fuera del proyecto, o define `api_key_env` con el **nombre** de una variable de
  entorno, nunca con la clave en sí.
- Mantén `security_outbound_mode = "warn"` o `"redact"` para cualquier perfil alojado.
- Ejecuta `pedacito config --check` o `/security` y lee los hallazgos antes de la
  primera ejecución alojada en un repositorio nuevo.
- Agrega `.env` y los archivos de claves a `.gitignore`, o deja que Pedacito los agregue con
  `manage_gitignore`.
- Si alguna vez se envió una clave a un proveedor alojado, rótala. La redacción protege las peticiones
  futuras, no las ya enviadas.

---

## Pilar 2: sandbox estricto del espacio de trabajo

El objetivo es que cada archivo que toca el modelo esté dentro del proyecto que abriste,
y que un symlink, una ruta absoluta o un segmento `..` no puedan cambiar eso.

### Contención de rutas: `resolve_safe_path`

Cada ruta de herramienta pasa por `security.resolve_safe_path(root, target)`:

1. La raíz del proyecto se resuelve con `Path.resolve()`, que sigue los symlinks.
2. El texto sin procesar de la ruta se revisa en busca de cualquier componente `..`. Un `..` literal se rechaza
   incluso cuando volvería a colapsar dentro de la raíz, porque una ruta suministrada por el modelo
   nunca debería necesitar el directorio padre.
3. Una ruta relativa se une a la raíz. Una ruta absoluta se usa tal cual.
4. El candidato se resuelve con `Path.resolve()`, de modo que un symlink que apunta a
   `/etc/passwd` o `~/.ssh` se sigue hasta su destino real.
5. La ruta resuelta debe cumplir `is_relative_to(resolved_root)`. De lo contrario, la
   llamada lanza `SecurityError`.

Los llamadores incluyen las herramientas de lectura, edición, escritura y renombrado (`pedacito/tools.py`,
`pedacito/edits.py`, `pedacito/loop.py`, `pedacito/workspace.py`), el cargador de skills
(`pedacito/skills.py`), la ruta de restauración de checkpoints (`pedacito/checkpoints.py`) y
cada ruta de archivos de la GUI (`pedacito/gui/server.py`).

**Una nota sobre `os.path.commonpath`.** Pedacito usa `os.path.commonpath` en
`pedacito/index.py`, pero solo para determinar la raíz del proyecto cuando pasas varias
rutas. No decide si se permite el acceso. Es una comparación léxica y
no resuelve symlinks, así que no puede ser por sí sola la verificación de contención.
La contención usa `Path.resolve()` seguido de `is_relative_to`, que es más estricto
porque primero se resuelven los symlinks.

### La búsqueda está acotada por el índice

`grep` busca en la lista de archivos del índice, y las demás herramientas de búsqueda trabajan desde el mismo índice.
Esa lista se construye desde la raíz del proyecto y excluye las rutas ignoradas y sensibles. Una consulta
de búsqueda por lo tanto no puede leer un archivo fuera del proyecto, ni siquiera con un patrón como
`../../`. Los argumentos de alcance (`src/`, `tests/*.py`) se comparan solo contra las rutas
indexadas.

### Las escrituras están protegidas

- **Rutas protegidas.** El modelo nunca puede escribir `.git/`, `.pedacito/`,
  `.pedacito.toml`, `PLAN.md`, `todo.md` ni nada en `protected_paths`.
- **Conflictos de merge.** Un archivo con marcadores de conflicto de git sin resolver no se edita.
- **Verificación de archivo obsoleto.** La página de revisión registra el SHA-256 de cada archivo cuando se
  construye. Al aplicar se rechaza la escritura si el archivo cambió en disco entre tanto.
- **Compuerta de sintaxis.** Una edición que rompe un archivo antes limpio se rechaza antes de
  escribirse. Pedacito usa `ast.parse` para Python y tree-sitter para los demás
  lenguajes admitidos.
- **Primero la instantánea.** Cada escritura va precedida de un respaldo y un checkpoint (ver
  el Pilar 4).
- **Modo de lista permitida.** `auto_allowlist = true` restringe una tarea autónoma a los
  archivos que nombra su título o que leyó la búsqueda. Los archivos nuevos siempre se permiten.

### La configuración del proyecto no es confiable por defecto

Un repositorio puede incluir un `.pedacito.toml`. Ese archivo no puede habilitar la ejecución por sí
solo. Estos ajustes se ignoran en un archivo de proyecto hasta que lo leas y ejecutes
`pedacito trust PROJECT`:

- Comandos y hooks: `hooks`, `mcp_servers`, `auto_verify`, `test_command`,
  `allow_serve`, `allow_lsp`, `lsp_servers`, `load_cmd`, `unload_cmd`.
- Interruptores de capacidades: `allow_shell`, `allow_write`, `allow_browser`,
  `allow_screenshots`, `allow_fetch`, `allow_image_gen`, `enable_mcp`, `allow_mcp`.
- Ajustes del sandbox (`sandbox`, `sandbox_network`, `sandbox_writable`,
  `sandbox_strict`) y la depuración del entorno (`scrub_env`, `env_passthrough`).
- Redirección de endpoints: `base_url`, `api_key`, `api_key_env`. Un proyecto no puede enviar
  tus prompts a un servidor distinto.
- Interruptores de verificación del código del repositorio: `verify_tests`, `verify_dom`, `dom_entry`.
- Cualquier cosa que amplíe adónde pueden ir los datos: `fetch_allowed_hosts`,
  `browser_eval_hosts`, `shell_allowlist`.

Editar un archivo en el que confías revoca la aprobación. Los ajustes inofensivos del mismo
archivo siguen aplicándose, y estos campos se siguen respetando desde tu configuración de usuario, desde
`--config FILE` y desde las variables de entorno `PEDACITO_*`. El archivo inicial
escrito por `pedacito config --init --path <project>/.pedacito.toml` mantiene esas
líneas comentadas, de modo que confiar en él ni advierte ni revoca lo que concediste.

### Límite de la GUI y la red

El servidor de la GUI es local, pero otros sitios web aún pueden enviarle peticiones. Pedacito
lo protege de cuatro maneras:

- Se enlaza a `127.0.0.1` y rechaza las peticiones con un encabezado `Host` ajeno.
- Se rechaza una petición que cambia el estado cuyo `Origin` o `Referer` no es loopback.
  Una página abierta en el mismo navegador no puede controlar el servidor, aunque adivine el puerto.
- Solo se aceptan cuerpos `POST` en JSON, y se envía una Content-Security-Policy estricta.
- Los archivos del proyecto se sirven solo desde dentro de la raíz del proyecto. La ruta `/preview/`
  rechaza los segmentos `..` y los symlinks que escapan, y solo se enmarca desde el mismo origen.

Los iframes de vista previa (`/preview/` y `/api/runner/preview/`) además están
en sandbox con `sandbox="allow-scripts"` y deliberadamente **sin**
`allow-same-origin`. Una página en vista previa por lo tanto se ejecuta en un origen opaco: puede
ejecutar sus propios scripts, pero no puede leer el almacenamiento de la GUI ni llamar a la API del agente
como la propia aplicación. Una petición que cambia el estado desde la vista previa llega con
`Origin: null`, que la protección de POST solo para loopback rechaza. Las páginas del proyecto también se
sirven sin `'unsafe-eval'` en su CSP (la CSP de la propia página de la GUI no cambia;
los scripts en línea siguen ejecutándose, así que una página estática autocontenida se previsualiza con normalidad).
**↗ New tab** abre la página como un documento normal del mismo origen, fuera del sandbox;
esa es la salida de emergencia de confianza, así que úsala solo para proyectos que ejecutarías localmente.

La verificación de actualizaciones automáticas (`pedacito/update.py`) se mantiene dentro del mismo límite:
el navegador nunca contacta a PyPI por sí mismo (el `connect-src 'self'` de la CSP lo
prohíbe): el servidor local hace la única petición GET anónima, y la actualización se ejecuta como una
lista de argumentos con `shell=False`. `check_for_updates = false` desactiva toda la
función: ni petición, ni subproceso, ni banner.

Las herramientas `fetch` y `web_search` (`pedacito/webfetch.py`) agregan una protección contra SSRF:

- Solo se obtienen URL completas `http` y `https`.
- Las direcciones de loopback (`127.0.0.1`, `localhost`, `::1`), link-local y de metadatos de nube
  (`169.254.169.254`), y los rangos de LAN privada se rechazan a menos que el host se
  nombre en `fetch_allowed_hosts`.
- La verificación se ejecuta sobre las direcciones **resueltas**, así que un nombre público que resuelve a una
  IP privada (DNS rebinding) se rechaza. La conexión se fija a la dirección
  validada, y las redirecciones se vuelven a verificar en cada salto.

### La salida de herramientas son datos, no instrucciones

La salida de `run`, `browser`, `fetch`, `mcp`, `git`, `lsp` y `screenshot` se envuelve
como texto de observación no confiable. Se eliminan los caracteres de control y las secuencias de escape ANSI.
Las frases dirigidas al asistente, como "ignore previous instructions", "you
are now" o "reveal the system prompt", se marcan en la observación. `pedacito
eval` planta inyecciones para medir con qué frecuencia se siguen.

Esto es una mitigación. No es una garantía, razón por la cual existen los Pilares 3 y 4.

---

## Pilar 3: ejecución con mínimos privilegios

El objetivo es que un comando que elige el modelo se ejecute con no más poder del que necesita,
pueda detenerse y no pueda alcanzar tus credenciales ni la red.

### Se ejecuta como tú, nunca como root

Pedacito nunca pide privilegios elevados, nunca llama a `sudo` y nunca cambia
su ID de usuario. Cada comando, servidor y prueba se ejecuta con los permisos de la cuenta
que inició Pedacito. No lo ejecutes como root. Un proceso propiedad de root no tiene
una ruta segura sin sandbox para un comando elegido por el modelo.

### El sandbox del sistema operativo

Los comandos que el modelo hace ejecutar (`run`, y los comandos de prueba usados por
`verify_tests`, `run_test` y el selector de pruebas) se envuelven con `pedacito/sandbox.py`:

| Plataforma | Backend | Cómo se usa |
|---|---|---|
| Linux | `bwrap` (bubblewrap) | Espacios de nombres de usuario sin privilegios, sin ayudante setuid |
| macOS | `sandbox-exec` (Seatbelt) | Un perfil generado aplicado al proceso hijo |
| Windows | `win_restricted` (Job Object + token restringido) | Integrado desde v0.4.0; ver abajo |

**Sandbox de Windows (`win_restricted`, v0.4.0).** `pedacito/sandbox_win.py` ejecuta el
proceso hijo dentro de un Job Object de Windows y de un token restringido con integridad
baja (`ctypes`, cargado de forma diferida tras `os.name == "nt"`; sin paquetes de terceros).
El job está configurado para terminar todos los procesos al cerrarse, así que un comando
cancelado, con tiempo agotado o caído se lleva consigo todo su árbol de procesos, y Pedacito
espera al hijo antes de borrar cualquier directorio temporal. La etiqueta de integridad baja
se aplica solo a los directorios del proyecto y temporales, y el recorrido que la aplica se
salta los protegidos `.git`, `.pedacito` y `.pedacito.toml` (y nunca sigue uniones NTFS ni
symlinks), de modo que un comando en sandbox no puede reescribir un hook de git, el archivo
de política ni sus propias barreras de protección. Al arrancar, Pedacito prueba el backend
con un proceso hijo real que no hace nada; si fallan el Job Object, el token restringido o el
lanzamiento del hijo, el backend figura como `unavailable` y nunca se anuncia como activo.
`sandbox.status()` devuelve `active`, `unavailable` u `off`.

`win_restricted` aísla la vida del proceso (Job Object) y restringe las escrituras a los
directorios del proyecto y temporales reetiquetados (integridad baja, saltando las rutas
protegidas de arriba), pero **no** filtra el tráfico de red saliente ni oculta archivos
legibles por el usuario: el filtrado de red saliente y el ocultamiento de lecturas requieren
un AppContainer en Windows, o los backends POSIX `bwrap` / `sandbox-exec`. `sandbox = false`
desactiva por completo el envoltorio.

**Puerta dura para ejecución desatendida (D14).** Los modos `workspace-write` y `danger`
desatendidos, las tareas en segundo plano y `approval_policy = "never"` con `sandbox`
activo deben ejecutarse dentro de un sandbox del sistema operativo activo. Si no hay backend,
lanzan `SandboxRefused` y el comando no se ejecuta. Las ejecuciones interactivas `ask` no
quedan bloqueadas por la puerta, pero sí usan el sandbox cuando existe.

Los argumentos de `bwrap` en Linux ofrecen las siguientes garantías:

- `--die-with-parent`, `--new-session` y `--unshare-pid`: el hijo no puede sobrevivir a
  Pedacito, no puede tomar el control de su terminal y no puede ver otros procesos.
- El sistema de archivos raíz es de solo lectura (`--ro-bind / /`). `/tmp` es un tmpfs privado.
- El proyecto es la única ubicación escribible, junto con los directorios de
  `sandbox_writable`.
- `.git`, `.pedacito`, `.pedacito.toml`, `pedacito.toml` y `.pedacitoignore` se
  montan como solo lectura. Un comando no puede plantar un hook de git ni reescribir sus propias barreras de protección.
- Los directorios secretos (`~/.ssh`, `~/.gnupg`, `~/.aws`, `~/.azure`, `~/.kube`,
  `~/.docker`, `~/.config/gcloud`, `~/.config/gh`, `~/.config/pedacito`) se reemplazan
  con montajes tmpfs vacíos. Los archivos secretos (`~/.netrc`, `~/.git-credentials`,
  `~/.npmrc`, `~/.pypirc`) se enlazan a `/dev/null`.
- La red no se comparte (`--unshare-net`) a menos que definas `sandbox_network = true`.

En macOS el perfil Seatbelt deniega el acceso a la red, deniega las escrituras de archivos salvo al
proyecto, a un directorio temporal y a `sandbox_writable`, y deniega las lecturas de las
mismas rutas secretas.

**Sondear antes de confiar.** Pedacito inicia una vez un comando nulo dentro del sandbox y
guarda el resultado en caché. Si el backend no puede iniciar (por ejemplo, los espacios de nombres de usuario están
deshabilitados, o el proceso está dentro de un contenedor que prohíbe el anidamiento), el comando
se ejecuta sin sandbox tras un aviso. Define `sandbox_strict = true` para negarte a ejecutar
en su lugar. `pedacito config --doctor` y `/status` muestran si el sandbox está
activo. La observación de un comando fallido indica si se ejecutó en sandbox, para que el modelo
no persiga una escritura bloqueada como si fuera un error de código.

### Qué está y qué no está en sandbox

| Ruta de ejecución | En sandbox | Entorno | Tiempo límite |
|---|---|---|---|
| Acción `run` (comando elegido por el modelo) | Sí | Lista permitida depurada | `shell_timeout` (120 s), `shell_idle_seconds` (10 s) |
| Comandos de prueba de `run_test` | Sí | Depurado | Techo de 15 s por ejecución |
| `verify_tests` tras una edición | Sí | Depurado | Tiempo límite de las pruebas |
| `pedacito serve` (servidores de desarrollo en segundo plano) | **No** | Depurado | Se mantiene en ejecución hasta que se detenga |
| `browser`, `lsp`, servidores MCP | **No** | Depurado | `lsp_timeout` (10 s), descubrimiento MCP 5 s por servidor |
| Enriquecimiento del oráculo de API (`api_oracle_exec`) | Sí donde hay un backend | Depurado | El plazo restante de la tarea, con tope de 12 s |
| Hooks y `auto_verify` (tus propios comandos) | **No** (por diseño) | Depurado para los hooks (+ `PEDACITO_EVENT`); `auto_verify` también depurado | Tiempo límite del hook |
| `load_cmd` / `unload_cmd` del ciclo de vida del servidor (configuración confiable) | **No** (por diseño) | Lista permitida depurada + `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH` | 120 s |
| Pestaña **Run** de la GUI (tú haces clic) | **No** | Variables de credenciales eliminadas (ver abajo) | 15 s por defecto, o el valor que definas |

Tus propios hooks y el comando `auto_verify` son tuyos, así que no están en sandbox.
Los comandos del ciclo de vida del servidor están en el mismo grupo: solo argv, sin sandbox,
sin shell, y con las dos variables documentadas arriba.

### Control de procesos y tiempos límite

Los comandos se inician sin shell. `&` y `nohup` se rechazan, de modo que un
proceso en segundo plano no pueda escapar del tiempo límite. Los programas de larga duración pasan por `serve`,
que los registra bajo `.pedacito/procs/`.

`pedacito/proc.py` impone dos límites a cada comando:

- Un **tiempo límite duro** (de reloj).
- Un **tiempo límite por inactividad** (sin salida durante N segundos), que detiene un proceso colgado en silencio.

Cuando se alcanza cualquiera de los límites, el proceso se termina. En POSIX se mata todo el
grupo de procesos, de modo que los nietos no puedan seguir ejecutándose. Si el hijo aún no
sale, Pedacito escala a una terminación forzada.

Otros límites:

- `auto_smoke_timeout` (20 s) acota la verificación de arranque tras una ejecución autónoma.
- `dom_timeout` (45 s) acota la carga de una página headless.
- `serve_wait` (3 s) acota cuánto espera Pedacito a que un servidor de desarrollo imprima su URL.

### Política de comandos y entorno

- **El shell está desactivado por defecto.** `allow_shell = false` desactiva `run` y `diagnose`.
  Activarlo es una decisión de confianza, y un archivo de proyecto no puede activarlo.
- **Lista de denegación por patrones.** `shell_deny = ["git push*", "rm -rf *"]` rechaza los comandos
  coincidentes incluso cuando el ejecutable está permitido.
- **Entorno depurado.** Con `scrub_env = true` (el valor por defecto), los comandos elegidos por el modelo
  y los servidores de desarrollo reciben solo `PATH`, `HOME`, la configuración regional y las variables de las herramientas. Las claves de API
  y los tokens no se heredan. Agrega lo que un proyecto necesite con
  `env_passthrough = ["DATABASE_URL", "AWS_*"]`.
- **`eval` del navegador** ejecuta JavaScript escrito por el modelo, así que está restringido a los hosts de
  `browser_eval_hosts`.
- **La resolución del entorno es solo de detección.** `env_resolve` lee
  marcadores locales (`.venv`, metadatos de uv/poetry/pipenv/conda, entornos
  de tox/nox, archivos de bloqueo de pnpm/yarn/bun, manifiestos de Cargo/Go)
  y comprueba si existe un intérprete o una herramienta. Nunca carga un script
  de activación, ejecuta `uv`/`poetry`/`conda`/`tox`/`nox`/`npx`, instala un
  paquete ni hace una petición de red. Las rutas descubiertas en el proyecto
  pasan por `security.resolve_safe_path`, así que un enlace simbólico o
  junction de `.venv` que escape del proyecto se rechaza; un
  `VIRTUAL_ENV`/`CONDA_PREFIX` explícito sigue siendo estado del usuario y
  puede vivir fuera. `command_env` compone `safety.child_env`, de modo que
  anteponer los directorios bin del proyecto no puede volver a agregar una
  variable que la depuración eliminó. Los scripts descubiertos por el
  resolutor son datos: nada se ejecuta porque un manifiesto lo nombre, y
  `allow_shell`/la política/el sandbox siguen controlando cada ejecución real.
- **El oráculo de API es estático por defecto.** La acción de recolección `api`
  lee `.pyi`/`.py` y metadatos de distribución desde los `site-packages` del
  intérprete resuelto, más archivos `*.d.ts` de un `node_modules` contenido.
  Nunca importa el código del paquete, evalúa sus valores por defecto, ejecuta
  un script del paquete ni usa la red, y cada candidato pasa por
  `security.resolve_safe_path`; los enlaces que escapan de una raíz de
  dependencias aprobada se omiten. El enriquecimiento opcional
  (`api_oracle_exec = true`) también necesita `allow_shell` y una política que
  lo permita antes de ejecutar `go doc`, `cargo metadata --offline --no-deps` o
  el `inspect` del intérprete del proyecto. Esos hijos son solo argv, se
  ejecutan con el entorno depurado y el sandbox del SO cuando existe, están
  limitados por el plazo de la tarea (tope de 12 s) y su salida se acota y
  redacta. La caché en `.pedacito/cache/api/` guarda solo metadatos de símbolos
  redactados y acotados, se escribe de forma atómica, y los resultados
  derivados de ejecución no se reutilizan después de retirar el consentimiento.

La política de entorno tiene tres niveles por superficie:

| Nivel | Superficies | Política |
|---|---|---|
| Depuración con lista permitida | herramientas del agente, pruebas, servidores de desarrollo, browser/LSP/MCP, hooks, `auto_verify`, `load_cmd`/`unload_cmd` del servidor | `safety.child_env`: solo `PATH`, `HOME`, la configuración regional y las variables de las herramientas; `env_passthrough` agrega nombres o patrones `PREFIX*`; `scrub_env = false` desactiva la depuración para configuraciones confiables |
| Lista de denegación de la pestaña Run | Pestaña **Run** de la GUI/TUI | Solo se eliminan los nombres con forma de credencial; las demás variables del entorno siguen pasando (ver Límites conocidos) |
| Configuración confiable | hooks, `auto_verify`, comandos del ciclo de vida del servidor | Sin sandbox por diseño. El entorno usa la depuración con lista permitida de arriba; los hooks también reciben `PEDACITO_EVENT`, los comandos del ciclo de vida reciben `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH` |

### El intermediario de permisos

Algunas acciones pausan la ejecución hasta que decides: acceso web, instalación de paquetes, comandos de shell
que de otro modo no están permitidos, lecturas de archivos sensibles, llamadas MCP no confiables
y `run_test` cuando `enable_run_test_in_gather = false`. El diálogo (GUI),
el aviso (TUI) o el evento JSON (RPC) muestra la categoría, el objetivo exacto y la razón.

| Opción | Efecto |
|---|---|
| **Allow Once** | Aprueba solo esta solicitud. |
| **Allow for This Run** | Aprueba ese tipo de solicitud hasta que termine la ejecución. |
| **Always Allow** | La aprueba de ahora en adelante. Define `enable_web` para el acceso web, o `allow_shell` para los comandos de shell, y guarda la elección en tu configuración. |
| **Permitir '<pattern>' en este proyecto** | Solo para comandos de shell: guarda una regla estrecha en `shell_allowlist_by_project` para la raíz de este proyecto. Los comandos que coinciden omiten la solicitud solo en ese proyecto. Nunca cubre instalaciones de paquetes, shells ni comandos encadenados. |
| **Deny & Redirect** | Rechaza la solicitud. Se le indica al modelo que continúe solo con los archivos locales del proyecto. |

Una solicitud que no respondes, porque el diálogo expira o presionas Stop, se
deniega. La CLI no tiene diálogo, así que una herramienta restringida informa que está deshabilitada en la
configuración en lugar de esperar a una persona que no está.

El acceso web está desactivado por defecto (`enable_web = false`). Con él desactivado, `fetch_url` y
`web_search` se ofrecen solo cuando una tarea nombra una URL o términos web, y cada llamada pregunta
primero.

### Perfil de endurecimiento

Una configuración conservadora para un repositorio en el que no confías del todo:

```toml
[universal]
sandbox = true
sandbox_strict = true       # refuse to run instead of running unsandboxed
sandbox_network = false
scrub_env = true
allow_shell = false         # turn on per project, after reading the code
allow_write = false
enable_web = false
security_outbound_mode = "warn"
redact_secrets = true
shell_deny = ["git push*", "rm -rf *", "curl *", "wget *"]
```

---

## Pilar 4: registro de auditoría inmutable y viaje en el tiempo

El objetivo es que cada cambio que haga una ejecución autónoma pueda inspeccionarse, compararse y
revertirse, y que el historial sobreviva a una ejecución fallida o revertida.

### Qué se registra

Cada Task, paso de Auto, andamiaje de New y checkpoint manual escribe un checkpoint en
`<project>/.pedacito/checkpoints/`:

```
.pedacito/checkpoints/
  index.json              checkpoint tree: ids, parents, labels, modes, timestamps, head
  snapshots/<id>/         full copy of the tracked source files at that checkpoint
```

Cada checkpoint registra su id (por ejemplo `cp_0003_a1b2c3`), el id de su padre, su
modo, una etiqueta, una marca de tiempo y `files_changed`. Los modos son `task`, `auto`, `new`,
`manual` y `restore`. El conjunto rastreado excluye los directorios de compilación y caché como
`.git`, `node_modules`, `.venv` y `dist`, y omite los artefactos binarios de más de 2 MB.

Los checkpoints se enlazan por padre, así que el archivo es un árbol. Crear uno después de una restauración
inicia una nueva rama desde el punto restaurado en lugar de sobrescribir el historial.

Algunos registros tienen su propio rastro de auditoría. `sessions/<stamp>.md` registra qué se pidió, qué
se buscó y qué cambió. `events/<stamp>.jsonl` registra cada paso, llamada al modelo,
pliegue y verificación. `log.txt` es una traza con marcas de tiempo. Todos enmascaran los secretos.

### Inspecciona antes de deshacer

`pdc` es un alias corto de `pedacito`, y ambos aceptan los mismos comandos.

```bash
pedacito checkpoints PROJ                 # the tree: ids, parents, modes, timestamps
pedacito checkpoints PROJ --json          # the same tree for scripts
```

El botón **Checkpoints / Undo** de la GUI muestra la línea de tiempo con un diff unificado para cada
archivo, y ofrece restauración con un clic para un archivo o todo el espacio de trabajo. **⑂ Branch Prompt
From Here** restaura el espacio de trabajo a un checkpoint (los archivos actuales se guardan primero en un
checkpoint de seguridad) y prepara el cuadro de prompt para que la siguiente tarea se ramifique desde allí. La TUI tiene
`/checkpoints`, `/undo` y `/checkout ID`. Cada diff proviene de
`difflib.unified_diff` sobre la instantánea y el archivo actual, de modo que ves exactamente lo que
cambió una ejecución autónoma antes de aceptarla.

### Revertir

```bash
pedacito undo PROJ                        # state immediately before the last run
pedacito checkout cp_0003_a1b2c3 PROJ     # restore the whole workspace to a checkpoint
pedacito checkout cp_0003 PROJ --file app.py   # restore one file only
pedacito restore PROJ                     # restore the newest backup snapshot
pedacito restore PROJ --snapshot NAME     # restore a specific snapshot
pedacito restore PROJ --list              # file-level backups from before checkpoints existed
pedacito restore PROJ --dry-run           # preview a restore without writing
```

Reglas que mantienen segura la reversión:

- **Los cambios sin confirmar se capturan primero.** Antes de una restauración, cualquier cambio sin confirmar en el
  árbol de trabajo se guarda como un checkpoint `restore`. Una restauración siempre puede deshacerse.
- **Restaurar convierte el destino en la cabeza activa.** La siguiente ejecución se ramifica desde él, de modo que
  se conserva el historial.
- **Una restauración de todo el espacio de trabajo es completa.** También elimina los archivos rastreados que no
  existían en el checkpoint de destino. Una restauración de un solo archivo solo toca ese archivo.
- **Undo recurre a una alternativa con elegancia.** `pedacito undo` prefiere el padre del checkpoint activo. Un
  proyecto creado antes de que existieran los checkpoints recurre a su instantánea de respaldo más reciente.
- **Las restauraciones a nivel de archivo son conservadoras.** `pedacito restore` deja en su lugar los archivos creados después de la
  instantánea. Agrega `--delete-created` para eliminarlos también; los directorios que queden vacíos por la
  eliminación también se eliminan. Los archivos creados por la ejecución se listan en un archivo auxiliar
  `backups/<stamp>.created`.

Dentro de una ejecución, el modelo puede rebobinar su propio trabajo con la acción `undo`, que revierte
lo que hicieron hasta ahora las escrituras de esa tarea. El trinquete verde del modo autónomo revierte un paso
que aún falla tras sus rondas de reparación al último estado que pasaba, y elimina
los archivos que creó el paso.

### Inmutabilidad: qué garantiza y qué no

Los checkpoints son **solo de anexado por diseño y se escriben de forma atómica**. El índice se escribe
en un archivo temporal y se mueve a su lugar con `os.replace`, de modo que una falla no pueda dejar un
índice a medio escribir. Las instantáneas no se reescriben después de tomarse. Las restauraciones crean
nuevos checkpoints en lugar de editar los antiguos.

**No** son a prueba de manipulación evidente. El índice no tiene cadena de hashes ni firma, y
las instantáneas son archivos simples. Cualquiera que pueda escribir en `.pedacito/` puede editar el historial.
Si necesitas un registro de auditoría que resista la manipulación, confirma tu trabajo en git o copia
`.pedacito/checkpoints/` a un almacenamiento que el agente no pueda alcanzar.

### Reinicio y retención

`pedacito prune` conserva las N sesiones, respaldos y archivos de eventos más recientes. `pedacito reset`
borra el estado de ejecución, incluidos los checkpoints, en un solo paso. Deja tu código,
`.pedacito.toml` y `.pedacitoignore` intactos. `reset --dry-run` muestra lo que se iría, y
los git worktrees activos bajo `.pedacito/worktrees/` hacen que el reinicio se rehúse a menos que pases
`--force`.

---

## Commits, pull requests, undo y archivos scratch (v0.3.3)

- **Escaneo previo de secretos antes de cada commit.** `pedacito commit` (y `/git commit` en la TUI,
  y el cajón Git de la GUI) escanea el diff preparado con el mismo conjunto de patrones que
  `security.scan_text_for_secrets` antes de ejecutar `git commit`. Una coincidencia rechaza el commit
  e indica el archivo y el patrón. Solo `--allow-secrets` lo anula, y esa excepción se escribe de
  forma explícita en la línea de comandos, nunca se guarda en la configuración.
- **Escaneo previo de secretos antes de cada pull request.** `pedacito pr` escanea de la misma forma el
  diff de la rama antes de redactar título y cuerpo, así que una credencial tampoco puede llegar a la
  descripción del PR. `--dry-run` imprime el borrador y nunca llama a `gh`.
- **Sin push automático y sin force-push.** Nada en `gitops` ejecuta `git push` ni `--force`. `commit`
  deja un commit local, y `pr` abre un pull request para una rama que ya publicaste. Las llamadas a git
  usan solo argv, sin shell.
- **`undo` comprueba el HEAD de git.** Si el HEAD del repositorio se ha movido desde que se tomó el
  checkpoint (por ejemplo, hiciste commit entre medias), `pedacito undo` se niega a rebobinar el árbol
  de trabajo y pide `--force`, para que un commit no se deshaga en silencio. Antes de que una ejecución
  escriba en un árbol divergente, Pedacito toma primero un checkpoint.
- **`.pedacito/scratch/` está contenido y es de solo lectura para el modelo.** La salida grande de las
  herramientas se escribe ahí como archivos `*.log`. El modelo solo puede usar `read` y `grep` sobre esos
  archivos `*.log` directos, con la misma contención de rutas que cualquier otra lectura. No puede
  escribir, renombrar ni salir del directorio.
- **Las rutas de ACP están contenidas.** En `pedacito acp`, una mención `@file` o un bloque de recurso
  pasa por `security.resolve_safe_path`, así que se rechazan los recorridos de ruta y las escapadas por
  enlaces simbólicos. El proyecto se vincula en `session/new`, y un directorio de trabajo ajeno se
  rechaza en lugar de leerse.

## Exportación de trayectorias

La exportación de trayectorias es una función de conjuntos de datos **opcional**, desactivada por defecto
(`record_trajectories = false`). Es el único lugar donde Pedacito escribe un registro de
**prompt completo**: `pedacito/trajectories.py` captura los mensajes que el
arnés realmente envió y recibió durante una ejecución (turnos de sistema, usuario y asistente,
llamadas a herramientas del modelo y resultados de herramientas) y escribe un archivo JSON por tarea
bajo `.pedacito/trajectories/`:

```text
.pedacito/trajectories/20261120T101500Z_4f2a9c1d.json
```

Eso es deliberadamente un compromiso de privacidad distinto del de `replay.py`, que almacena
los prompts como un **hash más una vista previa breve redactada y nunca el texto completo**
(consulta `--record` / `--replay` en `pdc ask --help`). Una trayectoria está pensada para
el filtrado de conjuntos de datos SFT/GRPO, así que necesita la transcripción real. Trata un
archivo de trayectoria como el código fuente que cita.

Actívalo para una ejecución o para un proyecto:

```bash
pedacito ask . -t "fix the retry loop" --apply --save-trajectory
```

```toml
# user config, .pedacito.toml, or a profile -- a local logging flag,
# honoured like log_sessions without `pedacito trust`
record_trajectories = true
```

Qué protege el archivo:

- **La misma tubería de redacción que los registros y las sesiones.** Cada valor de cadena pasa por
  `safety.redact` antes de la serialización, sin importar `redact_secrets`; la
  transcripción serializada se escanea luego con
  `security.scan_text_for_secrets`. Un hallazgo residual se enmascara en su lugar; si
  un hallazgo sobrevive o el enmascarado rompería el JSON, la escritura se rechaza
  con una advertencia y no se crea ningún archivo. Un secreto detectado nunca se persiste.
- **Solo archivos locales.** No hay llamada de red ni telemetría; los archivos
  viven bajo el `.pedacito/` del proyecto, que `workspace.py` agrega a
  `.gitignore` en la primera ejecución.
- **Los metadatos no son una segunda copia de la tarea.** Almacenan un `task_hash` SHA-256,
  nunca el texto de la tarea, junto con `verify_passed`,
  `harness_version`, `edit_format`, `model_id`, `provider`, `started_at`,
  `elapsed_s` y los totales de tokens.
- **La captura no puede romper una ejecución.** El hook en `loop.run_task` es
  seguro ante excepciones: un disco lleno, un error de permisos o un mensaje mal formado deja
  la tarea sin afectar (la advertencia va a stderr).

Los archivos son tuyos. Son JSON simple; elimina `.pedacito/trajectories/`
cuando quieras, y no copies un conjunto de datos a un repositorio ni a un servicio compartido
sin revisarlo primero. La detección de secretos se basa en patrones, así que una
credencial sin forma conocida aún puede sobrevivir a la redacción: el mismo
[límite conocido](#límites-conocidos-y-brechas) aplica aquí.

---

## Lista de verificación resumida

- [ ] Los perfiles alojados usan `security_outbound_mode = "warn"` o `"redact"`.
- [ ] Las claves viven en tu configuración de usuario o en variables de entorno (`api_key_env`), nunca en el repositorio.
- [ ] `.env`, los archivos de claves y los archivos de credenciales están en `.gitignore`.
- [ ] `sandbox = true`, y `sandbox_strict = true` donde una alternativa silenciosa sea inaceptable.
- [ ] `sandbox_network = false` a menos que una prueba realmente necesite la red.
- [ ] `allow_shell`, `allow_write` y `enable_web` están desactivados hasta que los necesites.
- [ ] Ejecutaste `pedacito trust` solo en proyectos cuyo `.pedacito.toml` leíste.
- [ ] Antes de una ejecución autónoma grande, sabes cómo ejecutar `pedacito checkpoints` y `pedacito undo`.
- [ ] Los checkpoints que te importan también están en git u otro almacén en el que el agente no pueda escribir.
- [ ] `record_trajectories` está activado solo donde estés dispuesto a conservar localmente los prompts completos.
- [ ] Nunca ejecutas Pedacito como root.

## Higiene de artefactos sensibles, confianza de política y aprobaciones (v0.4.0)

**Higiene de artefactos sensibles (D10).** El material de credenciales nunca sale del proyecto por
los almacenes propios de Pedacito. La puerta compartida, `security.check_sensitive_write`, rechaza
variantes de `.env`, `id_rsa*`, `*.pem`, `*.key`, `*.p12`, `credentials.json` y los demás nombres de
`security.SENSITIVE_NAMES` y `SENSITIVE_PATTERNS`, además de los directorios tipo `~/.ssh`. Se aplica en
ambos motores: el motor de dos fases (`edits.apply_edits`, `tools.tool_write`) y el motor intercalado
(`workspace_overlay.VirtualWorkspace`) rechazan las mismas escrituras con el mismo mensaje. Ninguno de
estos almacenes contiene nunca los bytes de un archivo sensible:

- checkpoints, instantáneas de ratchet y copias de respaldo (`fileio.backup_copy` rechaza y no crea nada);
- espejos de ejecución en seco (`VirtualWorkspace.exec_root`);
- diffs de revisión, páginas de revisión y vistas previas JSON de la GUI. Los tokens incrustados se
  redactan como `[REDACTED]` en el HTML y las filas de revisión, pero el texto que se aplica no cambia.

Los nombres de plantilla y de clave pública siguen siendo utilizables: `.env.example`, `.env.sample`,
`.env.template` e `id_rsa.pub` no se tratan como secretos.

**Confianza del archivo de política (`.pedacito/policy.toml`).** El archivo de política de un proyecto
puede restringir lo que el agente puede hacer, pero no ampliarlo. El archivo solo se lee si el proyecto
es de confianza (`pedacito trust PROYECTO`). Un archivo no confiable se ignora por completo, así que su
`permission_mode` o `approval_policy` nunca surte efecto. El motor de políticas (`pedacito/policy.py`)
resuelve cada capacidad según el modo de permisos y la política de aprobación, y un subagente recibe un
`child_policy()` que solo puede intersectar con su padre, nunca añadir.

**Modos de permisos y aprobaciones.** `permission_mode` es uno de `plan` (solo lectura), `ask` (cada
capacidad que muta pide aprobación), `accept-edits` (las escrituras de archivos son automáticas y el resto
pide aprobación), `workspace-write` (escrituras y ejecuciones automáticas dentro del sandbox) y `danger`
(todo es automático). `approval_policy` es `ask`, `never` (una solicitud se deniega de inmediato, con un
evento `permission_decision`) o `auto-safe` (`pytest`, `ruff`, `git status` y `git diff` se aprueban; lo
demás pide aprobación). Las sesiones en segundo plano y sin interfaz se deniegan de forma determinista en
lugar de esperar a una persona.

## Límites conocidos y brechas

Estos son límites reales. Léelos antes de depender de Pedacito para código no confiable.

- **Sin puntuación de entropía.** La detección de secretos se basa en patrones, con filtros de longitud y de marcadores de posición.
  Un secreto de alta entropía que no coincide con ninguna forma conocida y está bajo un nombre
  no relacionado no se detecta.
- **La pestaña Run de la GUI no está en sandbox, y su depuración del entorno es una lista de denegación.**
  `runner.run_project_command` elimina las variables cuyos nombres coinciden con `*_API_KEY`, `*SECRET*`,
  `*TOKEN*`, `*PASSWORD*`, `*ACCESS_KEY*`, `*CREDENTIAL*`, `GITHUB_TOKEN` o `PYPI_API_KEY`,
  sin distinguir mayúsculas de minúsculas. Cualquier credencial con un nombre fuera de esos patrones aún
  llega al programa que inicias desde esa pestaña. Ejemplos son un `DATABASE_URL` que incrusta una
  contraseña, o una variable de un proveedor sin un sufijo convencional. La lista de denegación tampoco es un
  sandbox. Usa `run` con el sandbox, o mantén los proyectos no confiables fuera de la pestaña Run.
- **Los comandos del ciclo de vida del servidor no están en sandbox.** `load_cmd` / `unload_cmd` de una
  configuración confiable se ejecutan como un argv con el entorno depurado con lista permitida más
  `PEDACITO_SERVER_NAME` / `PEDACITO_MODEL_PATH`. No pasan por un shell, pero
  tampoco están en sandbox, y `scrub_env = false` les entrega el entorno ambiental.
  Mantén ese interruptor desactivado en configuraciones que no hayas leído.
- **Los servidores de desarrollo, el navegador, LSP y los servidores MCP no están en sandbox.** Reciben un
  entorno depurado, rutas protegidas, listas permitidas y avisos de confianza. Un servidor que configuras es un
  programa que elegiste ejecutar.
- **El sandbox de Windows es a nivel de proceso, no un contenedor.** `win_restricted` usa un Job Object
  y un token restringido de integridad baja. No aísla el sistema de archivos como lo haría una VM o un
  AppContainer, así que un comando aún puede leer lo que el usuario puede leer, y tampoco filtra el
  tráfico de red saliente (eso requiere un AppContainer). Cuando el backend no está
  disponible, los comandos interactivos se ejecutan sin sandbox tras un aviso, y las ejecuciones desatendidas
  se rechazan (ver la puerta dura de arriba).
- **Los entornos de contenedores y CI pueden deshabilitar el sandbox.** Si los espacios de nombres de usuario están desactivados,
  Pedacito recurre a la alternativa descrita en el Pilar 3.
- **Los checkpoints no son a prueba de manipulación evidente.** Consulta el Pilar 4.
- **La contención de rutas es tan buena como las herramientas que la llaman.** Cada herramienta de archivos integrada
  usa `resolve_safe_path`. Un hook personalizado o un servidor MCP externo que escribe archivos
  directamente queda fuera de ese límite.
- **El sandbox es defensa en profundidad, no una promesa contra un adversario decidido ni contra un
  error del kernel.**

## Reportar una vulnerabilidad

Por favor abre un aviso de seguridad privado en el repositorio de GitHub, o escribe a
contact@pedacito.dev, en lugar de un issue público. Incluye la versión de Pedacito
(`pip show pedacito`), tu plataforma, los pasos para reproducirlo y si el
sandbox estaba activo (`pedacito config --doctor`).
