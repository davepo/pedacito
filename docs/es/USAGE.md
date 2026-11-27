🌐 [English](../USAGE.md) | **Español**

# Uso

Esta guía cubre el uso diario de Pedacito: una primera ejecución, la GUI en el navegador, la línea de comandos, los
modos interactivo y autónomo, y cómo fijar contexto en una tarea. Para instalar, consulta
[INSTALLATION.md](INSTALLATION.md). Para los ajustes, consulta [CONFIG.md](CONFIG.md). Para el modelo
de seguridad y cómo deshacer una ejecución autónoma, consulta [SECURITY.md](SECURITY.md).

Para ampliar Pedacito con skills, servidores MCP, hooks y sobrescritura de prompts, consulta
[SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md). Para trabajar en Pedacito mismo, consulta
[CONTRIBUTING.md](CONTRIBUTING.md).

Los tutoriales paso a paso con la salida que deberías esperar están en
[../../QUICKSTART.es.md](../../QUICKSTART.es.md). La elección y el ajuste de modelos está en
[../../MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md).

## Inicio rápido en 60 segundos

1. **Inicia un servidor de modelos.** Carga un modelo en LM Studio, Ollama, Lemonade o
   llama.cpp. Una API alojada también sirve, si tienes una clave.
2. **Instala** desde PyPI (requiere Python 3.11+). Agrega los extras de navegador y visión
   si los quieres, o consulta [Instalar](INSTALLATION.md#instalar) para las instalaciones con pipx y desde el código fuente:

   ```bash
   pip install pedacito                     # Windows: py -m pip install pedacito
   ```

3. **Elige tu forma de entrar.**

   **En el navegador:**

   ```bash
   pedacito gui ~/code/your-project
   ```

   Abre **Settings → Model discovery**, haz clic en **Scan local servers**, marca los
   modelos que quieras e impórtalos. Luego pide un cambio en **Task/Edit**.

   **En la terminal:**

   ```bash
   pedacito models scan                       # find servers, write a profile per model, measure each one
   pedacito ask ~/code/app -t "make the retry count configurable"   # dry run: shows a diff, writes nothing
   pedacito ask ~/code/app -t "make the retry count configurable" --review   # side-by-side page, you sign off
   ```

Las versiones paso a paso, con lo que deberías ver en cada punto, están en
[../../QUICKSTART.es.md](../../QUICKSTART.es.md).

## Cambiar el idioma (EN | ES)

La interfaz de Pedacito está disponible en inglés y español. Cámbiala donde trabajes:

| Superficie | Cómo |
|---|---|
| GUI | El selector **EN \| ES** en la barra superior; la elección se guarda en `language`. |
| TUI | `F3` alterna EN/ES, o `/lang es` / `/lang en`. |
| CLI | `pedacito --lang es ...`, o `PEDACITO_LANG=es` (alias `PEDACITO_LANGUAGE`). |
| Configuración | `language = "es"` y `llm_reply_language = "match_ui"` (consulta [CONFIG.md](CONFIG.md#idioma-y-localización-language-llm_reply_language)). |

`llm_reply_language` solo cambia el idioma de las explicaciones en prosa del modelo. En la TUI,
`/profiles` lista cada perfil configurado, y `/profile NAME` cambia a uno.

## Qué obtienes

| Área | Qué hace | Más información |
|---|---|---|
| **Técnicas para modelos pequeños** | Un ciclo de recuperación dirigido por el arnés, una máquina de estados de requisitos (`UNLOCATED → LOCATED → READ → EDITED → VERIFIED`), microciclos de edición anclados a símbolos, una compuerta de sintaxis por AST antes de que corra cualquier prueba, paquetes de reparación por traceback, conversaciones amigables con la caché de prompt KV y rehidratación de tramos exactos | [../../TECHNIQUES.es.md](../../TECHNIQUES.es.md) |
| **Telemetría de Engine V2** | La GUI muestra la fase en vivo, el libro de requisitos y las búsquedas recientes mientras corre una tarea | [../../QUICKSTART.es.md](../../QUICKSTART.es.md#parte-a-la-gui) |
| **Memoria del repositorio** | Comandos confirmados en `.pedacito/micro_facts.json`, con los hechos contradichos retirados automáticamente. Los esquemas de herramientas sin usar se podan para ahorrar tokens | [../../TECHNIQUES.es.md](../../TECHNIQUES.es.md) §14 y §15 |
| **Escáner de servidores locales** | Encuentra LM Studio, Ollama, Lemonade, llama.cpp y Strata / Avalon (preset en el puerto 8080, con respaldo en `STRATA_API_KEY`) en cualquier puerto, y luego importa solo los modelos que marques. Consulta [Strata / Avalon](INSTALLATION.md#configuración-del-servidor-de-modelos) | [Configuración del servidor de modelos](INSTALLATION.md#configuración-del-servidor-de-modelos) |
| **Preajustes de proveedores alojados** | Configuración con un clic para Claude, ChatGPT, DeepSeek, Qwen, Gemini y OpenRouter. Las claves pueden venir de una variable de entorno en lugar del archivo de configuración | [Configurar](CONFIG.md#configurar) |
| **Enrutamiento de roles** | Envía el trabajo de planificador, verificador o escalamiento a un perfil distinto, por trabajo o por proyecto | [../../MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md#enrutamiento-de-roles-híbrido-local--nube) |
| **Asistente Auto-Tune** | Responde cuatro preguntas sobre tu hardware y tu base de código, previsualiza los presupuestos y luego aplícalos | [../../QUICKSTART.es.md](../../QUICKSTART.es.md#parte-a-la-gui) |
| **Razonamiento por perfil** | Activa o desactiva el razonamiento para cada perfil de modelo, desde la GUI o con `disable_thinking` | [../../MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md#razonamiento-por-perfil) |
| **Benchmarks de modelos** | Seis proyectos pequeños con una rúbrica oculta de evaluación, puntuados por perfil, incluida una prueba de dibujo SVG paramétrico cuyo resultado se muestra en línea, para que puedas comparar modelos en tu propio hardware | [../../MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md#compara-modelos-con-los-benchmarks-de-diagnóstico) |
| **Arquitecto de prompts** | `✦ Refine Prompt` convierte una idea de una línea en una especificación consciente del repositorio con archivos objetivo fijados y firmas de interfaz. `↶ Undo Refine` restaura tu texto | [../../TECHNIQUES.es.md](../../TECHNIQUES.es.md) |
| **Contratos de interfaz** | Auto fija las firmas públicas del proyecto en cada paso, y verifica las importaciones entre archivos y la aridad de las llamadas antes de que corran las pruebas | [../../TECHNIQUES.es.md](../../TECHNIQUES.es.md) |
| **Compuerta estática** | Una verificación rápida de nombres no definidos y sintaxis (`ruff` F821 y E999, con una alternativa en Python puro) se ejecuta antes de cualquier subproceso de pruebas | [../../TECHNIQUES.es.md](../../TECHNIQUES.es.md) |
| **Trinquete verde** | Auto guarda una instantánea del proyecto antes de cada paso y revierte un paso que aún falla al último estado que pasaba | [../../TECHNIQUES.es.md](../../TECHNIQUES.es.md) |
| **Intermediario de permisos** | El acceso web, la instalación de paquetes, los comandos de shell no aprobados y las lecturas de archivos sensibles pausan la ejecución para una decisión de Allow Once, Allow for This Run, Always Allow o Deny & Redirect | [Seguridad y confianza](SECURITY.md) |
| **Escrituras seguras** | Simulación por defecto, revisión lado a lado, respaldos antes de cada escritura, `pedacito restore` para deshacer | [Página de revisión](#página-de-revisión) |
| **Introspección de pruebas en vivo** | Durante la investigación el modelo puede ejecutar una verificación de prueba acotada (pytest, Node/Vitest, Go o Cargo) y leer el traceback en vivo antes de escribir una corrección. Activado por defecto (`enable_run_test_in_gather`) | [Ejecuciones de pruebas acotadas](ARCHITECTURE.md#ejecuciones-de-pruebas-acotadas-durante-la-investigación) |
| **Verificaciones de contrato del frontend** | Antes de que corran las pruebas, los archivos locales de `<script src>` y `<link href>` deben existir, y las búsquedas literales de `getElementById` y `querySelector('#id')` deben coincidir con los ids del HTML | [Verificaciones de contrato del frontend](ARCHITECTURE.md#verificaciones-de-contrato-del-frontend) |
| **Modo creativo / de diseño** | Las tareas de diseño (sitios web, rediseños, trabajo en `.html`/`.css`/componentes de UI) se ejecutan como un Senior Product Engineer y Diseñador Visual: muestreo más cálido, un presupuesto de respuesta de ≥8192 tokens y reescrituras de HTML/CSS de archivo completo. `--creative`, `/creative` en la TUI o el interruptor ✨ Creative de la GUI lo fuerzan en cada tarea | [Modo creativo / de diseño](#modo-creativo--de-diseño) |
| **Live Preview** | Un proyecto con páginas `.html` obtiene una pestaña 🌐 Preview en la GUI: una vista en vivo de solo lectura de la página que se recarga cuando termina una ejecución, o mientras escribe | [Live Preview](#live-preview) |
| **Respeto del muestreo alojado** | Las APIs alojadas no reciben `temperature`/`top_p`/`top_k`/`min_p`/`repeat_penalty` por defecto, de modo que se aplican los valores ajustados del proveedor; reactívalos con `send_temperature_to_hosted` o fija `hosted_temperature` | [Las APIs alojadas conservan su propio muestreo](CONFIG.md#las-apis-alojadas-conservan-su-propio-muestreo) |
| **Lecciones de reparación** | Una reparación que corrige una verificación fallida deja una lección sin duplicados y específica del repositorio en `.pedacito/micro_facts.json`; una lección que un directorio comprueba dos veces se promueve a una regla con alcance de ruta administrada automáticamente (`.pedacito/rules/auto_*.md`) que solo aparece cuando el modelo lee ese directorio | [Lecciones de reparación](ARCHITECTURE.md#lecciones-de-reparación-y-trampas-del-repositorio) |
| **Ediciones a nivel de símbolo y renombrado** | Reemplaza una función completa por su nombre sin repetir su texto antiguo, o renombra un símbolo en todo el proyecto sin llamada al modelo (`pedacito rename`) | [Edición](ARCHITECTURE.md#edición-buscarreemplazar-no-diffs) |
| **JSON impuesto por el motor y enrutamiento con scout** | Los pasos de búsqueda están restringidos por el soporte de esquema JSON del servidor, y `scout_profile` envía las búsquedas a un modelo rápido mientras el modelo grande planifica y edita | [Guía de modelos](../../MODEL_GUIDE.es.md#dual-gpu--dual-model-enrutamiento-scout--arquitecto-scout_profile) |
| **Enrutamiento automático por especialista** | `auto_route_specialties` envía cada tarea al perfil cuyas etiquetas de especialidad le encajan mejor (código, UI/SVG, matemáticas, ...), solo en los límites de las tareas, con una transferencia limpia de VRAM en un servidor local; `pedacito calibrate` detecta las etiquetas | [Enrutamiento por especialista](ARCHITECTURE.md#enrutamiento-automático-por-especialista) |
| **Insignias de calibración y telemetría en vivo** | Los perfiles muestran insignias ✓ Calibrated / ⚠ stale / ○ Not calibrated / ☁ Hosted; `calibrate --all` mide cada modelo local por turno; un pie de la GUI y una cinta de la CLI siguen en vivo el uso de contexto, los tokens, la velocidad y los aciertos de caché | [Medición y mantenimiento](#medición-y-mantenimiento) |

### El espacio de trabajo de la GUI

`pedacito gui` sirve una página local en `127.0.0.1` sin CDN y sin dependencias
adicionales, así que funciona sin conexión. Ejecuta el mismo motor que los comandos.

- Una barra de herramientas de dos filas con la ruta del proyecto, un selector en vivo de **perfil y modelo**,
  la luz de estado del servidor, una insignia de calibración para el perfil activo
  (✓ Calibrated, ⚠ stale, ○ Not calibrated o ☁ Hosted API), y los totales de la sesión
  de tokens, aciertos de caché y costo.
- Un **pie de telemetría** bajo el espacio de trabajo: la ventana de contexto en uso con una barra
  verde / ámbar / roja, tokens del turno y de la sesión, velocidad de generación, aciertos de caché KV
  y pliegues, y qué modelo tiene la tarjeta. Haz clic para expandir el panel
  completo.
- **Esquemas de color** del Theme Manager (el botón de paleta de la barra superior). Los nueve esquemas
  integrados y cualquier tema personalizado provienen del motor de temas compartido: el mismo conjunto que ofrece
  `/theme` en la TUI. Consulta [Esquemas de color](CONFIG.md#esquemas-de-color).
- **Cinco modos** junto al cuadro de chat: **Task/Edit** para cambios, **Ask** para
  preguntas de solo lectura, **Auto** para ejecuciones de meta a lista de verificación, **New** para iniciar
  un proyecto y **How ?** para la referencia integrada.
- **✦ Refine Prompt** sobre el compositor convierte una idea de una línea en una especificación completa
  en el lugar: objetivo, archivos a crear o modificar, contratos de interfaz,
  requisitos y casos límite, y verificación. **↶ Undo Refine** restaura tu
  texto original. Arrastra el borde superior del compositor para hacerlo más alto mientras
  lees la especificación.
- **✨ Creative** sobre el compositor fuerza la persona Builder para la siguiente
  ejecución de Task/Auto/New: el ojo de un diseñador, muestreo más cálido y reescrituras de HTML/CSS
  de archivo completo. Con él desactivado, las tareas de diseño se siguen detectando automáticamente (`creative_auto`).
  Consulta [Modo creativo / de diseño](#modo-creativo--de-diseño).
- Los interruptores del compositor **✦ Auto-route** (enrutamiento por especialista) y **Post-edit verify** (ejecuta `auto_verify` tras cada tarea Auto) viven ahora junto a **✨ Creative**, debajo del cuadro de texto, y ya no en la barra superior; la barra superior se mantiene despejada con el selector `EN | ES`.
- **Project ▾ → 🎨 Generate image…** abre un diálogo para generar imágenes con tu perfil de imagen (consulta [Generación local de imágenes](#generación-local-de-imágenes)).
- **Changes** muestra las ediciones propuestas como código antiguo y nuevo lado a lado, con
  aceptar, rechazar o editar a mano por archivo.
- Los diálogos **Permission requested** pausan una ejecución cuando el modelo pide acceso web,
  la instalación de un paquete, un comando de shell no aprobado o un archivo sensible.
  Consulta [Seguridad y confianza](SECURITY.md).
- **Settings** edita tus perfiles, cada ajuste, el asistente Auto-Tune, los servidores
  MCP con nombre y el archivo de configuración sin procesar, sin borrar tus comentarios.
- **Benchmarks** ejecuta los seis proyectos de diagnóstico contra cualquier perfil. El
  resultado del SVG del pelícano se dibuja en línea en su tarjeta de puntuación.
- **Project ▾** contiene las acciones del proyecto: **Rename symbol…** renombra un identificador
  en todo el proyecto sin llamada al modelo, y muestra los diffs por archivo antes de
  escribir. **Calibrate All Local Models…** mide cada perfil local por turno
  (los perfiles alojados se excluyen; las tarjetas vigentes se omiten a menos que marques
  re-calibrate). **Restore from a backup…**, **Check model (loads it)** y
  **Clear terminal** también están aquí, además de los diálogos de mantenimiento: **Memory & repo lessons…**,
  **Skills library & creator…** (explora las skills integradas, de usuario y de proyecto,
  previsualiza las coincidencias de disparadores de una tarea, escribe o ✦ redacta automáticamente skills personalizadas),
  **Consolidate session…** (compacta la transcripción en un punto de entrega),
  **Background servers…**, **Run project checks…**, **Prune old
  sessions/backups…** y **Reset project state…** (prune y reset muestran una vista previa
  antes de escribir). Una insignia `Scout: <profile>` aparece junto a la etiqueta del modelo
  cuando hay un perfil scout activo.

#### Disponibilidad de perfiles en el selector

El selector de perfiles valida los perfiles locales contra el sondeo de `/api/servers`, que se ejecuta
cada 15 segundos. La insignia de hardware tiene su propio sondeo más rápido de `/api/hardware`,
cada 3 segundos y también enviado en vivo por SSE, así que la RAM y la VRAM se mantienen
actualizadas por sí solas. Un perfil local cuyo servidor responde pero
cuyo modelo no está en la lista de modelos del servidor se marca **· unavailable** y
no puede seleccionarse; la API de cambio también lo rechaza con un 409. La descripción emergente
nombra el modelo y el servidor. Un perfil cuyo servidor no es accesible se marca
**· server offline** y vuelve a ser seleccionable en un sondeo posterior una vez que el
servidor inicia. Los perfiles alojados nunca se sondean y siguen siendo seleccionables, al igual que un
perfil cuyo estado de servidor aún no se conoce. Si el perfil activo deja de estar
disponible, se sigue mostrando con su estado; la GUI nunca cambia a otro por
su cuenta.

Los hilos de chat se guardan con marcas de tiempo y pueden renombrarse. Sin un proyecto
abierto se guardan en `~/.local/state/pedacito/threads/global/` (consulta la
estructura de `.pedacito/` más abajo); una vez abierto un proyecto, viven dentro de ese
proyecto. El panel **Files** lista los archivos indexados del proyecto, y **Backups**
lista las instantáneas que puedes restaurar. Cuando el proyecto tiene páginas `.html`, una pestaña
**🌐 Preview** se une a ellos: una vista en vivo de solo lectura de la página seleccionada que
se recarga cuando termina una ejecución, de modo que una tarea de sitio web muestra su resultado de inmediato
(consulta [Live Preview](#live-preview)).

### La CLI de un vistazo

El paquete instala `pedacito` y un alias más corto, `pdc`. Ambos aceptan los mismos
comandos. Ejecuta `pedacito <command> --help` para ver cada opción.

| Comando | Qué hace |
|---|---|
| `pedacito gui [PROJECT]` | Abre el panel de control en el navegador |
| `pedacito models scan` | Encuentra servidores locales, elige modelos, escribe perfiles y los mide |
| `pedacito models [--profile NAME] [--model ID]` | Muestra los números ajustados del perfil activo (o de uno indicado); `--model` nombra el id del modelo para esa vista |
| `pedacito models refresh` | Actualiza ahora la caché del catálogo models.dev (ventanas de contexto y precios de proveedores alojados); si la descarga falla, se conserva la caché anterior |
| `pedacito config --check` | Prueba la conexión y la generación. `--doctor` también sondea la salida estructurada. `--setup` ejecuta la configuración inicial; `--upgrade [--dry-run]` agrega a tu archivo las opciones recién documentadas |
| `pedacito calibrate` | Mide el modelo activo (o los perfiles locales con `--all`) y guarda una tarjeta de modelo que ajusta los valores por defecto. Los modelos alojados se rechazan |
| `pedacito profiles [--recommend]` | Lista los perfiles con sus puntuaciones y sugiere enrutamiento de roles |
| `pedacito profile [--show\|--run\|--reset]` | Muestra la instantánea de hardware, la última sonda de velocidad y el auto-ajuste activo; `--run` fuerza una sonda, `--reset` descarta las mediciones en caché |
| `pedacito ask PROJ -t "..."` | Ejecuta una tarea. Simulación por defecto. Agrega `--review`, `--apply`, `--plan`, `--refine`, `--jsonl` o `--json` |
| `pedacito explain PROJ -q "..."` | Responde una pregunta sobre el código. Nunca edita |
| `pedacito auto PROJ -t "..."` | Convierte una meta en una lista de verificación y luego la implementa tarea por tarea |
| `pedacito new DIR -t "..."` | Inicia un proyecto desde cero |
| `pedacito chat PROJ` | Sesión interactiva que mantiene caliente el índice (`--rpc` convierte la sesión en un protocolo de máquina) |
| `pedacito tui PROJ` | TUI en pantalla dividida: transcripción, barra lateral de actividad/plan en vivo y un pie de telemetría con Context Meter / consumo de tokens / tok/s. Suite completa de comandos slash; `--headless` es programable |
| `pedacito rpc [PROJ]` | JSONL-RPC por stdin/stdout para editores y otros agentes: una petición por línea, eventos en streaming, dirección, avisos de permiso, cancelación; el índice, los perfiles y la sesión permanecen calientes todo el tiempo |
| `pedacito acp [PROJ]` | Protocolo Agent Client Protocol (JSON-RPC 2.0) por stdin/stdout: `initialize` (anuncia el protocolo `0.4.0` y sus capacidades), `session/new`, `session/prompt` con bloques de texto, `@file` e imagen, notificaciones `session/update` y respuestas de `session/request_permission`; el mismo motor caliente que `rpc` |
| `pedacito daemon [start\|stop\|status\|sessions]` | Daemon headless solo en loopback: un índice caliente y leases de modelo compartidos por sesiones de espacio de trabajo aisladas, con endpoints HTTP/SSE para adjuntar, separar, enviar prompts, cancelar, permisos y revisión. Autenticado con bearer token; `status` sale con código 0 limpiamente cuando no hay daemon |
| `pedacito how "..."` | Ayuda integrada para la CLI, la GUI y los ajustes |
| `pedacito undo PROJ` | Restaura el espacio de trabajo al estado inmediatamente anterior a la última ejecución; `--list` muestra el árbol de checkpoints |
| `pedacito checkpoints` | Lista los checkpoints de viaje en el tiempo y la cabeza activa; `--limit` y `--json` dan forma a la salida |
| `pedacito checkout ID` | Restaura todo el espacio de trabajo o un `--file` desde un id de checkpoint (un prefijo único sirve) |
| `pedacito update [--run]` | Verifica en PyPI si hay un Pedacito más reciente y, opcionalmente, actualiza esta instalación |
| `pedacito restore PROJ` | Deshace la última escritura |
| `pedacito rename OLD NEW` | Renombrado de símbolos en varios archivos mediante el AST y el tokenizador, sin llamada al modelo. `--file` lo acota, `--strings` también renombra literales de cadena de palabra completa, `--include-labels` solo cadenas de etiquetas de prueba, `--apply` escribe (por defecto: vista previa del diff) |
| `pedacito image PROMPT` | Genera una imagen con un perfil de modelo de imagen configurado (`--out`, `--size WxH`, `--steps`, `--seed`, `--negative`, `--json`). Se guarda en `.pedacito/images/` |
| `pedacito diag [PATH]` | Alias de `pedacito diagnose`: ejecuta el linter o verificador de tipos detectado en el proyecto |
| `pedacito parsers prefetch [--languages ...]` | Precarga las gramáticas de tree-sitter para uso sin conexión o aislado, e informa ok o failed por lenguaje |
| `pedacito commit [--all] [-m MSG] [-y]` | Escanea el diff preparado en busca de secretos y luego crea un commit local. Nunca hace push; `--no-model` redacta el mensaje sin conexión y `--allow-secrets` es la excepción explícita |
| `pedacito pr [--base main] [--draft] [--dry-run] [-y]` | Redacta título y cuerpo a partir del diff de la rama, el registro de commits y los checkpoints recientes, y luego abre el pull request con `gh`. Nunca hace push de la rama |
| `pedacito index`, `map`, `files`, `cost` | Construyen e inspeccionan el índice. No necesitan llamada al modelo, salvo los resúmenes de `index` |
| `pedacito eval` | Puntúa el arnés sobre repositorios de prueba. Usa `--baseline` como compuerta de regresión |
| `pedacito bench [BENCHMARK\|all]` | Ejecuta la suite de benchmarks de diagnóstico de seis proyectos y la rúbrica oculta de evaluación. `--list` y `--recent` no necesitan modelo; `--json` y `--strict` encajan en CI |
| `pedacito bench compare BASELINE CURRENT` | Compara dos scorecards de eval escritas con `pedacito eval --scorecard`: salida 0 aprobado, 1 regresión, 2 inválido o no comparable. No necesita modelo; `--max-prompt-growth` vale 0 por defecto |
| `pedacito memory`, `skills`, `prune`, `reset`, `trust`, `worktree`, `serve`, `procs`, `lsp`, `mcp`, `diagnose`, `tests`, `webcheck`, `domcheck` | Herramientas de mantenimiento y de especialista. `mcp list` muestra las herramientas de los servidores MCP con nombre y sus indicadores de confianza |

Los benchmarks también se ejecutan desde la GUI mediante el botón **Benchmarks**; el comando
de la CLI construye los mismos proyectos y escribe los mismos informes.

## Ayuda integrada: `pedacito how`

Para la selección de modelos, el dimensionamiento de VRAM, los ajustes de razonamiento y el enrutamiento de roles, consulta la
[guía de modelos](../../MODEL_GUIDE.es.md). Tiene puntos de partida concretos para menos de 8 GB,
8 a 16 GB y 24 GB de VRAM, además del enrutamiento de roles híbrido local y en la nube.

`pedacito how` responde preguntas sobre el uso de Pedacito desde un **índice de referencia
estructurado** integrado: cada comando y opción de la CLI, la ruta exacta de navegación de la GUI para cada función (incluidos ✦ Refine Prompt, el diálogo de
permisos, el acceso web, las reversiones del trinquete verde y los benchmarks), y cada
ajuste de configuración con su valor por defecto, su ubicación en la GUI y su forma en `config.toml`. No necesita proyecto ni índice. La conexión con un
modelo es opcional: cuando se puede alcanzar un servidor de modelos rápido, reescribe los
hechos recuperados en una respuesta breve; de lo contrario, las tarjetas deterministas de
GUI/CLI/ajustes se imprimen de inmediato.

```bash
pedacito how "how do I disable thinking?"       # model synthesis if available
pedacito how --offline "how do I import models?" # instant, no model needed
pedacito how --json "how do I reset my config?"  # structured matches + answer
pedacito how                                      # interactive
```

Acepta `--url`/`--server`, `--model`, `--api-key`, `--vram`/`--vram-gb`,
`--config` y `--profile`, además de `--offline`/`--fast` y `--json`. La pestaña del compositor
**How ?** de la GUI expone la misma referencia con etiquetas de temas, comandos de CLI copiables
y botones **Open in Settings**.

## Flujo de trabajo diario

La ruta del proyecto es el primer argumento; el estado queda en `<project>/.pedacito/`.

```bash
pedacito index   ~/code/anki                       # build the index (cached)
pedacito ask     ~/code/anki -t "make the leech threshold configurable"
pedacito explain ~/code/anki -q "how does the scheduler pick the next card?"
```

### `index`

Analiza el proyecto en archivos, símbolos y rangos de línea, y agrega resúmenes de una línea del LLM.
`--no-summaries` construye solo el índice estático (sin llamadas al modelo);
`--resummarise` regenera los resúmenes en caché. Los resúmenes se guardan en caché por hash de contenido, así que volver a indexar cuesta solo lo que cambió. `ask` y `chat` actualizan el índice
automáticamente; las ediciones que haces en tu editor se detectan en la siguiente ejecución.

Comandos de inspección gratuitos (no se necesita modelo):

- `pedacito files PROJ`, cada archivo candidato y por qué se conservó o se omitió.
- `pedacito cost PROJ`, llamadas al LLM y tiempo estimados para un primer índice
  (`--tps` define tus tokens/segundo).
- `pedacito map PROJ`, imprime el índice (`--tier auto|flat|cards|dirs|ranked`).

### Entrada multimodal en el chat

Envía imágenes junto con una tarea a un modelo con visión: hasta **4 imágenes, 10 MB cada
una**, en PNG, JPEG, WebP o GIF.

- **CLI**: `--image RUTA` (repetible), p. ej. `pedacito ask . -t "replica esta maqueta" --image mock.png`; `new` también lo acepta.
- **GUI**: pega con `Ctrl+V` / `Cmd+V`, arrastra y suelta sobre el compositor, o pulsa `📎 Attach`.
- **TUI**: `/attach [RUTA | list | clear]` o `@image.png`.

### `ask`

Reúne el código fuente que la tarea necesita y luego propone ediciones. **Por defecto es una simulación**:
ves un diff y no se escribe nada.

```bash
pedacito ask PROJ -t "..." --apply      # write immediately
pedacito ask PROJ -t "..." --review     # side-by-side sign-off page
pedacito ask PROJ -t "..." --reindex    # force a fresh index first
pedacito ask PROJ -t "..." --plan       # split a multi-part task into steps
pedacito ask PROJ -t "..." --jsonl      # one JSON event per line on stdout
```

`--refine` ejecuta primero una pasada del Arquitecto de Prompt/Especificación: una idea de una línea se convierte en una
especificación de cinco secciones (objetivo, archivos, contratos de interfaz, requisitos y
verificación) basada en el árbol de archivos real, y la ejecución usa esa especificación. Si ningún
modelo responde, una alternativa determinista escribe las mismas secciones, así que la opción
nunca detiene la ejecución. Consulta [../../TECHNIQUES.es.md §17](../../TECHNIQUES.es.md).

Otras opciones útiles: `--steps N` (límite de búsquedas), `--strict` (sale con 1 cuando una tarea
aplicó ediciones pero dejó un bloque de edición fallido o pruebas afectadas que siguen fallando
tras las rondas de reparación), `--model`, `--url`, `--profile`,
`--role planner=big` (enruta un trabajo a otro perfil en esta ejecución),
`--auto-route` / `--no-auto-route` (enrutamiento por especialista por ejecución; consulta
[Enrutamiento automático por especialista](ARCHITECTURE.md#enrutamiento-automático-por-especialista)),
`--creative` (modo Builder creativo/de diseño en esta ejecución; consulta
[Modo creativo / de diseño](#modo-creativo--de-diseño)),
`--config`, `--root`, `--exclude`, `--include`, `--ext`.

**Modo de plan.** Una tarea que cambia tres cosas en una ventana de contexto normalmente
termina la primera, la segunda a medias y olvida la tercera. `--plan`
(o `plan_threshold`, que una tarjeta de modelo define para modelos más débiles) le da a cada
cambio su propia ventana, búsquedas, verificaciones de evidencia y reversión. Una tarea que
nombra cinco o más archivos se planifica incluso cuando `plan_threshold` es 0, el
valor por defecto, y lo que elige la tarjeta de un modelo fuerte, porque el ancho es un
problema de capacidad de la ventana, no de habilidad; un `plan_threshold` explícito en
tu configuración sigue mandando, y `--no-plan` excluye una tarea. Una tarea escrita como
viñetas o varios requisitos claros se divide de forma determinista, sin llamada al modelo;
cualquier otra va al rol `planner` (que puede declarar
`depends_on`). El plan es una lista de verificación en markdown en `.pedacito/plans/` que puedes
leer y editar (`--review-plan` pausa para eso). Tras un paso fallido se le pide una vez al planificador
que revise los pasos restantes (`max_replans`).

**Ejecuciones legibles por máquina.** `--jsonl` imprime cada evento (pasos, llamadas al modelo con
uso de tokens, la respuesta final y los diffs) como un objeto JSON por línea en
stdout y mueve el texto para humanos a stderr, de modo que un plugin de editor o un script pueda
manejar Pedacito. Para un solo objeto de veredicto final en lugar de un flujo de eventos, usa
`--json` (consulta [Modo headless y automatización](#modo-headless-y-automatización)). Las ejecuciones con escritura
(`--apply`) toman un bloqueo por proyecto
(`.pedacito/run.lock`); un segundo escritor se rechaza mientras el primero esté vivo.

**Dirección y preguntas.** En `chat`, todo lo que escribas mientras el modelo está
trabajando se entrega en su siguiente paso ("stop reading tests, it's in
`sync.py`") sin perturbar el prefijo de prompt en caché (`allow_steering`). En
un `ask`/`chat` interactivo el modelo también obtiene una acción `question` para ambigüedades genuinas,
como máximo dos veces por tarea; en `auto`, `--jsonl` y `--json` nunca
pregunta y en su lugar declara su suposición.

Cada escritura guarda primero una instantánea de los originales; `pedacito restore PROJ` deshace la
última.

### `explain`

Responde preguntas sobre el proyecto indexado en prosa, sin ediciones. Nunca
construye un índice; ejecuta `pedacito index` primero.

```bash
pedacito explain PROJ -q "trace the flow from answer to schedule"
pedacito explain PROJ                   # interactive
```

### Ver una GUI o una página web

Un modelo de texto no puede ver una ventana. `screenshot` y `browser` la convierten en
evidencia sobre la que puede actuar:

```toml
allow_screenshots = true
allow_browser = true
allow_shell = true                  # only needed to launch the app or server
vision_model = "your-vl-model-id"   # optional: multimodal model on the same server
```

`browser` toma una URL o un comando de servidor que imprima una, más acciones
opcionales:

```text
browser  command="python main.py"  pattern="press Enter; wait 3"  start=4
```

La gramática de `pattern` también cubre `hold ArrowUp 1.5` (una tecla mantenida PRESIONADA durante N
segundos; un `press` instantáneo puede caer entre dos muestras de un ciclo de juego de
paso de tiempo fijo), `click #start`, `fill #name Ada`, `type hello` y `eval` (JS en
la página). Esto detecta de forma confiable pantallas negras, páginas en blanco, texto faltante y
errores de consola; no juzga la calidad visual. Los PNG se guardan bajo
`.pedacito/screenshots/`. Para tus propios ojos, la pestaña 🌐 Preview de la GUI renderiza
el HTML del proyecto en vivo a medida que una ejecución lo escribe (consulta [Live Preview](#live-preview)).

### Consolidación de sesión

Las sesiones largas llenan la ventana de contexto. La consolidación destila el historial
reciente en una entrega estructurada de menos de 400 tokens —**Goal & Context**,
**Files Modified & Key Decisions**, **Verified Test Status**, **Open Next
Steps**— construida a partir de hechos del arnés (tareas, ediciones aplicadas/fallidas, resultados de pruebas,
decisiones), nunca con prosa inventada. Guarda `.pedacito/consolidated_session.md`,
fija las decisiones duraderas en `memory.md` y reinicia el búfer en vivo para que el siguiente
turno empiece solo desde la entrega:

- **GUI**: el botón **Consolidate Session** en el pie de telemetría (o
  **Project ▾ → Consolidate session…**). La transcripción se reinicia desde la
  tarjeta de entrega y el Context Meter se actualiza de inmediato.
- **Chat de CLI**: `/consolidate` (alias `/compact`); la siguiente tarea se siembra
  con la entrega. `pedacito memory PROJ --consolidate` ejecuta la misma
  destilación sobre las transcripciones de sesión más recientes e imprime el checkpoint.
- **RPC**: `{"id": 1, "op": "consolidate"}` (o `"method": "consolidate"`,
  alias `compact`) responde con `before_tokens`, `after_tokens` y la
  entrega; se rechaza mientras corre una tarea.

## Modo headless y automatización

`pdc ask` ya es el punto de entrada no interactivo de una tarea: no necesita TTY,
toma la tarea en la línea de comandos y deja el estado bajo `.pedacito/`.
`--json` convierte la ejecución en un solo veredicto legible por máquina para CI y arneses de evaluación;
`--jsonl` sigue siendo la alternativa en streaming (un evento por línea).
`--json` y `--jsonl` son mutuamente excluyentes y se rechazan juntos con un
error claro.

```bash
# One summary object on stdout; every human line goes to stderr.
pdc ask . -t "add retries to the sync client" --apply --json

# Same, with a one-run verification command; verify_passed reports its result.
pdc ask . -t "fix the failing parser tests" --apply --json --verify-cmd "pytest -q"
```

La forma de stdout es exactamente este objeto (sin claves adicionales, sin prosa, un objeto al
final de la ejecución):

```json
{
  "ok": true,
  "model": "qwen2.5-coder-14b",
  "provider": "LM Studio",
  "turns": 7,
  "elapsed_sec": 12.345,
  "tokens": {"prompt": 8123, "completion": 1456, "tok_per_sec": 23.4},
  "files_changed": ["src/sync.py"],
  "verify_passed": null,
  "diff": "--- a/src/sync.py\n+++ b/src/sync.py\n@@ -1,3 +1,4 @@\n...",
  "diff_truncated": false
}
```

| Clave | Tipo | Significado |
|---|---|---|
| `ok` | bool | Éxito de la tarea. No es el código de salida: una ejecución sin `--strict` puede salir con 0 mientras una verificación en rojo o un bloque de edición fallido hacen que `ok` sea false. El código de salida del proceso conserva su semántica existente. |
| `model`, `provider` | string | El modelo y la etiqueta del endpoint que respondió, de la telemetría de la ejecución. |
| `turns` | int | **Llamadas al modelo** en esta ejecución, exactamente `events.RunTelemetry.summary()["calls"]`: cada paso de recolección, respuesta, planificación y petición de reparación cuenta una vez. |
| `elapsed_sec` | float | Tiempo de llamadas al modelo acumulado por la telemetría de la ejecución (`elapsed_s`), no tiempo de reloj. |
| `tokens.prompt` / `tokens.completion` | int | Tokens de prompt y de completado en esas llamadas. |
| `tokens.tok_per_sec` | float | Tokens de completado por segundo de tiempo del modelo. |
| `files_changed` | list[string] | Rutas relativas al proyecto cuyos bytes cambió la ejecución aplicada (vacía en una simulación). |
| `verify_passed` | bool o null | El resultado de verificación de la ejecución. `null` significa que no corrió ninguna verificación. Con `--verify-cmd` (o un `auto_verify` configurado) es el resultado de ese comando; de lo contrario es el veredicto de las pruebas afectadas posterior a la edición cuando esas pruebas corrieron. |
| `diff` | string | Diff unificado combinado de los cambios aplicados, limitado a 200,000 caracteres. |
| `diff_truncated` | bool | `true` cuando el diff se cortó en ese límite. |

`--json` implica `--no-review`: nunca abre la página de revisión ni un paginador y
nunca pide un plan, una pregunta o una decisión de permiso. El texto de la tarea
nunca se repite en el objeto, y el `diff` y los campos de cadena pasan por el mismo
enmascarado `redact_secrets` que los registros, de modo que un secreto plantado en la tarea no pueda
volver a salir por stdout (define `redact_secrets = false` solo si quieres deliberadamente
una salida sin enmascarar).

`--verify-cmd "COMMAND"` anula `auto_verify` para esta única ejecución **solo
en memoria**: la configuración en vivo recibe el comando más una entrada `sources` de
`--verify-cmd`, y no se escribe ni reescribe ningún archivo de configuración. Se ejecuta una vez después de
una tarea aplicada; una simulación omitida por lo tanto deja `verify_passed` en null.

### Guardar trayectorias para SFT/GRPO

`--save-trajectory` escribe una transcripción JSON de prompt completo por cada tarea ejecutada en
`.pedacito/trajectories/`, para el filtrado de conjuntos de datos y el ajuste fino. Es la
contraparte opcional de `replay` (que almacena solo un hash y una vista previa breve redactada),
así que contiene los prompts reales; cada cadena se redacta y el
archivo se vuelve a escanear en busca de secretos antes de escribirse. La misma captura está
disponible en todas partes mediante `record_trajectories = true`; la opción la
activa solo para esta ejecución de `ask` sin tocar el archivo de configuración.

```bash
pdc ask . -t "fix the retry loop" --apply --save-trajectory
```

Cada archivo es `{"metadata": {...}, "messages": [...]}`, donde `messages` es la
transcripción con forma ChatML/OpenAI realmente intercambiada (turnos de sistema/usuario/asistente,
llamadas a herramientas y resultados de herramientas). Conserva solo las ejecuciones cuya verificación
pasó antes de construir un conjunto de datos:

```python
import json
from pathlib import Path

kept = []
for path in Path(".pedacito/trajectories").glob("*.json"):
    record = json.loads(path.read_text(encoding="utf-8"))
    meta = record["metadata"]
    if meta["verify_passed"] is True:          # null = no check ran
        kept.append({"messages": record["messages"], **{
            key: meta[key] for key in (
                "task_hash", "model_id", "provider", "edit_format",
                "harness_version", "started_at", "elapsed_s", "tokens")
        }})
Path("dataset.jsonl").write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in kept),
    encoding="utf-8")
```

`verify_passed` es el veredicto de las pruebas afectadas posterior a la edición de la propia tarea
(`true`/`false`/`null`); `null` significa que no corrió ninguna verificación dentro de la tarea, así que `is True`
excluye deliberadamente las ejecuciones sin verificar. Un `--verify-cmd` se ejecuta una vez después de la
tarea, así que su resultado lo informa `--json`; la trayectoria registra la
verificación dentro del ciclo que existía cuando terminó la tarea. El contrato completo de metadatos
y el compromiso de privacidad están en
[SECURITY.md — Exportación de trayectorias](SECURITY.md#exportación-de-trayectorias).

## Fijar contexto con `@`

Menciona un archivo o un símbolo con `@` en tu prompt, y Pedacito fija su código fuente exacto en la
ejecución. El contexto fijado se lleva a cada fase de la tarea (búsqueda, respuesta y edición), de modo que el
modelo no pueda perderlo ni trabajar en silencio a partir de otra versión.

```text
Make @src/app.py retry on timeout, and keep the signature that @Store.save uses.
```

### Qué acepta `@`

| Forma | Se resuelve a |
|---|---|
| `@path/to/file.py` | Una ruta relativa exacta |
| `@file.py` | Un nombre base único, cuando solo un archivo tiene ese nombre |
| `@app` | Un prefijo sin distinción de mayúsculas inequívoco o una coincidencia aproximada con el nombre de un archivo |
| `@Class.method` | Un nombre calificado exacto |
| `@file.py:symbol` | Un símbolo en un archivo específico |
| `@symbol` | Un nombre de miembro indexado único |

Dos cosas se ignoran a propósito: las direcciones de correo (`name@domain.com`) y las líneas de decoradores en
bloques de Python delimitados. Una mención que no coincide con ningún archivo o símbolo único no se fija.

### Administrar anclajes en la TUI

`/pin` muestra los anclajes de esta sesión. `/pin add REF` agrega uno explícitamente, `/pin remove REF`
elimina uno y `/pin clear` los elimina todos.

### Skills: catálogo bajo demanda e inyección manual

Las skills son listas de verificación en markdown con frontmatter YAML (`name`, `description` y, opcionalmente,
`triggers`). Pedacito mantiene el catálogo en el prompt (nombres y descripciones de una línea) y
carga un cuerpo completo solo cuando el modelo lo pide con la acción `skill`. Hasta dos coincidencias de
disparadores se seleccionan automáticamente para una tarea. Las skills de proyecto anulan a las del usuario, que
anulan a la biblioteca incluida; `pedacito skills list|show|add|draft|rm` administra los archivos.

Para forzar una en la siguiente tarea por tu cuenta, sin esperar una coincidencia de disparador:

```text
/skill add-a-test          # TUI: load the full body into the next task
```

El cuerpo se valida ahora y luego se lleva por el mismo bloque **Explicitly Pinned Context**
que los anclajes `@file`/`@symbol`, sobre el mapa, y nunca se expulsa por la poda de tokens.
`/skills` lista los nombres, `/pin remove skill:add-a-test` desancla uno y `/pin clear` desancla
todo. En la GUI abre **Skills library & creator** y usa **Use in next task** en cualquier
skill; la lista de fijadas en el diálogo (y la fila 📌 Pinned sobre el compositor) muestra qué
se inyectará, con las acciones Remove y Clear skills. Un anclaje de skill es un anclaje
de sesión normal (`skill:add-a-test`), así que acompaña a cada tarea hasta que lo elimines.

## Más comandos

### `chat`: sesión interactiva

```bash
pedacito chat ~/code/anki [--apply | --review]
```

Mantiene caliente el índice entre tareas. Comandos slash: `/profile NAME`,
`/model ID`, `/apply on|off`, `/steps N`, `/status` (paquete, modo de respuesta,
verificador, ventana, tarjeta de modelo), `/cost` (tokens, aciertos de caché y gasto por sitio de
llamada), `/memory`, `/skills`, `/consolidate` (alias `/compact`: compacta la
sesión en una entrega; consulta [Consolidación de sesión](#consolidación-de-sesión)),
`/profiles`, `/help`, `/quit`. **Ctrl-C cancela la
tarea en ejecución pero mantiene la sesión abierta.** Cambiar de perfil a mitad de sesión es la forma
más barata de comparar dos modelos en una tarea; un cambio fallido deja la
sesión en el modelo anterior.

Con `--rpc` la misma sesión caliente habla un protocolo de máquina en lugar de prosa:
stdin toma una petición JSON por línea y stdout responde con eventos JSONL;
consulta `pedacito rpc` más abajo.

### `tui`: sesión de terminal en pantalla dividida

```bash
pedacito tui ~/code/anki                 # apply-on by default, with undo backup
pedacito tui . --no-apply                # preview edits only
pedacito tui . -t "add retries to the sync client"
pedacito tui . --headless < commands.txt  # scriptable CI mode, no PTY
```

La TUI mantiene caliente el índice como `chat`, pero pinta cuatro zonas en lugar de un
REPL con desplazamiento: un encabezado (proyecto, perfil/modelo, insignia de calibración,
Auto-Route, Scout, rama/estado sucio de git), un panel principal de transcripción junto a una barra lateral plegable de
**Activity & Plan**, un pie de telemetría de dos filas y la barra de prompt
`pdc tui (profile) >`. El pie de telemetría se repinta en vivo: un observador de 150 ms
sigue el registro de eventos del proyecto mientras cada tarea corre en un hilo de trabajo.
Al texto de archivos del repositorio, registros y resultados de herramientas se le eliminan las secuencias
de escape de terminal antes de pintarse, de modo que un archivo no pueda cambiar el título de tu terminal ni escribir
en tu portapapeles. La barra lateral aparece con 100+ columnas y
se alterna con `/sidebar`; muestra las casillas de `PLAN.md`, los pasos en vivo de `read`/`grep`/
`mcp_call`/`task`, el enrutamiento `model_swap` por especialista y los archivos que esta
sesión cambió.

La entrada interactiva es un lector crudo cbreak/`msvcrt`, de modo que las teclas de navegación editan el
prompt en lugar de mostrar `^[[5~`: `PgUp`/`PgDn` desplazan la transcripción media
página, `Shift+Up`/`Shift+Down` desplazan tres líneas, `Alt+PgUp`/`Alt+PgDn` desplazan
la barra lateral de Activity/Plan, y la rueda del mouse desplaza tres líneas. La rueda
usa Alternate Scroll de xterm (`?1007h`) por defecto, de modo que la selección con clic izquierdo y arrastre
y el copiar/pegar del sistema operativo siguen funcionando; `/mouse on` (o `F2`) activa la
captura SGR opcional de la rueda y `/mouse off` restaura la selección nativa. `Up`/`Down` recuperan el
historial de comandos, `Left`/`Right`/`Home`/`End` editan el prompt, `Tab` completa
comandos slash y nombres de perfil, y `Esc`/`End` saltan de vuelta al final en vivo.
Una transcripción desplazada hacia atrás ([ ↑ Scrolled back N lines ... ]) permanece fija
mientras llegan líneas de registro en segundo plano. Los repintados usan la salida sincronizada DEC y
colocan el cursor de hardware dentro de `pdc tui (profile) > ...`.

**Entrada multilínea.** `Shift+Enter`, `Alt+Enter` y `Ctrl+J` insertan un salto de línea en
el prompt; una `\` final seguida de `Enter` hace lo mismo en terminales que no distinguen
`Shift+Enter`. El texto pegado llega mediante pegado entre corchetes (`?2004h`), así que un
pegado de varias líneas queda en el prompt como un solo bloque y no se ejecuta línea por
línea. `Enter` solo envía.

**Imágenes adjuntas.** `/attach [RUTA | list | clear]` (o `@image.png` en el prompt) prepara
imágenes PNG/JPEG/WebP/GIF para la siguiente tarea; `list` muestra lo preparado y `clear` lo
descarta. Consulta [Entrada multimodal](#entrada-multimodal-en-el-chat).

**Vista enriquecida (v0.3.3).** Sigue sin dependencias. Las ediciones se muestran como diffs
unificados con resaltado de sintaxis en línea, y cada llamada a una herramienta es una tarjeta
plegable de una línea: `Ctrl+O` alterna la última tarjeta, y `/expand [N|all]` y
`/collapse [N|all]` actúan sobre cualquier tarjeta. Escribe `@` en el prompt para abrir un
selector difuso interactivo de seis filas sobre archivos, símbolos e imágenes del proyecto: con las
flechas Arriba y Abajo te mueves, Tab o Enter inserta la ruta y Esc lo cierra. Las zonas de ratón SGR
también son clicables: `[EN|es]` cambia el idioma, las pestañas de la barra lateral cambian de panel,
una tarjeta de herramienta se expande o contrae, y la barra de desplazamiento salta. La captura del
ratón sigue siendo opcional (`/mouse on`, o `F2`).

El pie se actualiza después de cada llamada al modelo y evento: un
**Context Meter** verde/ámbar/rojo (`Ctx: [████████░░░░░░░░] 6,420 / 32,768 (19.6%)`), el
**Burn Rate** (tasa de consumo de tokens) del turno y de la sesión, `Speed: 44.2 tok/s (avg 41.8 tok/s)`,
la tasa de aciertos de caché KV, la cuenta de pliegues y `Active: <profile> [<role>/<specialty>]` con `$cost`
cuando hay precios. `pedacito tui` registra el intermediario de permisos en vivo, de modo que una herramienta MCP
no confiable, la lectura de un archivo sensible o un comando de shell preguntan en la barra de prompt:
`y` permite una vez, `a` permite esa clave durante la sesión, `n` deniega.

Comandos slash: `/help`, `/keys`, `/status` (`/telemetry`), `/new`, `/auto GOAL`,
`/auto! GOAL`, `/plan [GOAL]`, `/consolidate` (`/compact`), `/profile [NAME]`,
`/profiles`, `/model ID`, `/auto-route [on|off]`, `/think [on|off]`,
`/effort [low|medium|high]`, `/creative [on|off|auto]`, `/servers`,
`/unload [NAME]`, `/apply [on|off]`, `/verify on|off|status`, `/skills`,
`/skill NAME`, `/memory`, `/procs`, `/security`, `/theme`, `/pin`, `/rules`,
`/hybrid`, `/arena`, `/diff`, `/expand [N|all]`, `/collapse [N|all]`, `/checkpoints`, `/undo`, `/checkout ID`, `/run`,
`/fix`, `/image PROMPT`, `/update`, `/reset`, `/sidebar`, `/scroll up|down|top|bottom|N`,
`/mouse [on|off]`, `/clear`, `/quit` (`/exit`). Una línea simple es una
tarea y corre por el ciclo normal de recolección/edición/verificación. `/creative on` fuerza
la persona Builder en cada tarea, `auto` (el valor por defecto) detecta las tareas de diseño,
`off` mantiene ediciones quirúrgicas en todas partes; `/verify` controla el comando de verificación posterior a la edición
(consulta [los tres niveles](#los-tres-niveles-de-verificación-tal-como-se-entregan));
`/status` muestra el estado actual.

### `rpc`: maneja pedacito desde un editor u otro agente

```bash
pedacito rpc ~/code/anki          # or: pedacito chat ~/code/anki --rpc
echo '{"id":1,"method":"status"}' | pedacito rpc ~/code/anki
```

JSONL-RPC bidireccional por stdio, el mismo motor que `ask` y `chat`: cada
línea de stdin es `{"id": ..., "method": ..., "params": {...}}`; cada línea de stdout
es un marco JSON, de modo que un cliente nunca analiza prosa. `ask` transmite los
eventos de la tarea etiquetados con su `req_id` y termina con un marco `result` (`applied`,
`changed`, `ok`, costo y `blockers`: los hallazgos de las compuertas con los que terminó la ejecución;
`applied: true` junto a `blockers` no vacío significa que el código aterrizó y no se
verificó limpio, que es la diferencia que necesita un agente que llama); `steer` empuja
una nota a la cola de la tarea en ejecución; los eventos `permission_request` llegan cuando
una herramienta restringida quiere actuar y se responden con `permission_reply` (`allow_once`,
`allow_run`, `allow_always`, `deny`); `cancel` detiene la tarea y mantiene el
proceso caliente; `consolidate` (alias `compact`, también aceptado como `"op"`)
compacta la sesión en un punto de entrega y reinicia el búfer de reanudación,
de modo que el siguiente `ask` empiece solo desde la entrega; `switch_profile` y `status`
también funcionan a mitad de sesión. El índice,
los perfiles y el registro de sesión permanecen calientes entre peticiones: ese es todo
el punto. Los diagnósticos para humanos van a stderr; stdout permanece como JSONL puro.
Los detalles del protocolo están en `pedacito/rpc.py`, la justificación en
[Técnicas §31](../../TECHNIQUES.es.md).

### `acp`: puente de Agent Client Protocol

```bash
pedacito acp ~/code/anki                                  # editor o host de agente por stdio
pedacito acp ~/code/anki --profile fast --execution-mode interleaved
```

`pedacito acp` habla Agent Client Protocol (JSON-RPC 2.0) por stdin y stdout, para editores
y hosts de agentes como Zed. Envuelve el mismo motor caliente que `rpc`, así que el arnés no
se duplica. `initialize` anuncia las capacidades, y `session/new` vincula un proyecto (un
directorio de trabajo ajeno se rechaza en lugar de leerse) con un perfil y un modo de ejecución
opcionales. `session/prompt` acepta bloques de texto, recursos `@file` e imágenes; las rutas
`@file` pasan por la misma comprobación de contención que cualquier otra lectura, así que se
rechazan los recorridos de ruta y las escapadas por enlaces simbólicos. Las tareas transmiten
notificaciones `session/update` (`agent_message_chunk`, `agent_thought_chunk`, `tool_call`,
`tool_call_update`, `plan`), y las herramientas restringidas hacen el viaje de ida y vuelta por
`session/request_permission`. `session/cancel` equivale a la cancelación de rpc y mantiene vivo el
proceso. stdout es JSON-RPC puro; cada línea humana va a stderr, y el fin de stdin vacía la tarea
activa y sale con 0.

### `new`: inicia un proyecto desde cero

```bash
pedacito new my-app -t "a CLI todo manager with a sqlite backend" --apply
pedacito new my-api --spec docs/spec.md --review
pedacito new my-app -t "..." --plan-only --apply     # skeleton only
```

El modelo planifica un árbol de archivos, se escribe el esqueleto y luego se completa cada archivo.
Simulación por defecto. El directorio destino debe estar vacío o ser nuevo, o contener
solo los archivos propios de Pedacito (`.git`, `.gitignore`, `.pedacito/`,
`.pedacito.toml`, `.pedacitoignore`), a menos que uses `--force`. `--complex` entrega la
implementación al ciclo `auto`, una tarea por archivo planificado: los archivos de pruebas se
implementan después de las fuentes, de modo que cada uno se escribe contra la API real.

Cada archivo lo escribe su propia llamada, así que el árbol terminado se verifica como
un todo antes de que la compilación informe éxito: un andamiaje de navegador cuyos scripts no pueden
enlazar, o cuyo `getElementById` nombra un id que ninguna plantilla renderiza, recibe rondas de reparación
y luego imprime `BUILD INCOMPLETE` y sale con 1. El `README` generado de un proyecto web
también indica cómo iniciarlo realmente: los módulos ES se rechazan sobre
`file://`, así que nombra `python -m http.server 8000` y la URL a abrir en lugar
de una línea `python main.py` para un archivo que el proyecto no contiene. Para un
sitio web o un andamiaje con mucha UI, `--creative` ejecuta cada tarea en modo
creativo/de diseño (consulta [Modo creativo / de diseño](#modo-creativo--de-diseño)).

`--no-complex` (alias `--single`) desactiva la escalada automática a andamiaje Complex para
prompts que nombran varios módulos. `--complex` ahora activa `verify_final` por defecto, y un
andamiaje cuya ejecución de pytest no recolecta pruebas (código de salida 5) se trata como
andamiaje válido y no como fallo. El `pyproject.toml` inicial incluye
`[tool.pytest.ini_options] pythonpath = ["."]`, así que `pytest` encuentra el paquete desde la raíz.

### `auto`: ciclo autónomo sobre una lista de tareas

Divide una meta en una lista de verificación ordenada e implementa cada tarea en un contexto
nuevo, con verificación y reparación opcionales:

```bash
pedacito auto . -t "add retry logic to the sync client"      # plan only
pedacito auto . -t "..." --apply                             # plan, then implement
pedacito auto . --apply --verify "pytest -q"                 # verify and repair each task
pedacito auto . --apply --verify "pytest -q" --verify-final  # one whole-goal check at the end
pedacito auto . --apply --diagnostics                        # lint/typecheck each task
pedacito auto . --apply --worktree refactor                  # run in an isolated git worktree
pedacito auto . --apply --plan TODO.md                       # use an existing checklist
pedacito auto . -t "restyle the buttons in index.html" --force-auto  # skip the Task-mode nudge
```

**Aviso de idoneidad Task-vs-Auto.** Antes de levantar un `PLAN.md` de varios pasos,
`auto -t "..."` ejecuta una verificación rápida y determinista (`evaluate_mode_suitability`,
cero llamadas al modelo): un prompt que apunta a 1–2 archivos existentes, usa verbos
de edición localizados (`fix`, `restyle`, `rename`, `remove`, `add a link`, …), o ajusta
contenido/estilos en un proyecto diminuto (≤5 archivos fuente no ignorados) parece
trabajo de **Task** de una sola pasada. En una terminal, Pedacito imprime la recomendación
y pregunta `Run as a single-pass Task instead? [Y/n]`: Enter o `y` ejecuta todo
el prompt como una sola pasada de `ask` (~20 s, sin lista de verificación); `n` mantiene el ciclo Auto.
Un stdin que no es TTY (una tubería de CI) nunca se bloquea: registra una nota de una línea y
continúa con Auto. `--force-auto` (o `-y`/`--yes`) omite la verificación, y
los prompts que dicen "multi-step plan", "create a plan", "from scratch", o nombran
≥4 archivos/módulos nuevos distintos siempre mantienen Auto. La TUI lo replica en
`/auto GOAL` (`/auto! GOAL` fuerza el ciclo), y la GUI muestra un banner en vivo
"💡 Looks like a single-pass edit" sobre el compositor más una opción
"Run as a single-pass Task instead?" al ejecutar, con los botones **Run as Task
(Recommended)** y **Run Auto Anyway**.

El plan es una lista de verificación en markdown (`PLAN.md` por defecto), de modo que una ejecución puede reanudarse
y editarse; el archivo también conserva la meta completa (una línea `Goal:` con líneas de
continuación `> `), que una ejecución reanudada devuelve a cada tarea. No
hay un tope por defecto de tareas, pasos o tiempo; agrégalo con `--max-tasks N`,
`--max-minutes N` o `--steps N`. `--max-minutes` limita el reloj propio del ciclo:
la planificación, el andamiaje y los subprocesos de compuerta por paso corren fuera de él, así que un
llamador que necesita un techo de reloj debería mantener su propio margen.
`--repair N` define las rondas de reparación
por verificación fallida, `--keep-going` continúa tras las tareas fallidas y
`--auto-install` instala un paquete de Python faltante en lugar de informarlo.
`--verify-final` ejecuta el comando de `--verify` una vez, después de la última tarea sin
marcar, en lugar de después de cada tarea: una verificación de toda la meta no puede pasar a mitad del plan, así que las
ejecuciones por tarea revertirían correcciones parciales válidas (`auto_verify_final`,
desactivado por defecto). Después de la última tarea, `--apply` también ejercita el proyecto
una vez (`auto_smoke`, activado por defecto): los puntos de entrada de Python se ejecutan con exec y
el recibo dice `verify: smoke passed`; eso demuestra que el proyecto ARRANCA, no
que la suite esté en verde, así que un plan debería terminar con una tarea de ejecutar las pruebas o
deberías pasar `--verify "pytest -q"`. Sin `--verify` ni `main.py`
que ejecutar como humo, el recibo dice `verify: not run` en lugar de afirmar un éxito. `--strict` sale con 1 cuando una tarea aplicó ediciones pero dejó un
bloque de edición fallido o pruebas afectadas que siguen fallando tras las rondas de reparación,
para CI y tuberías desatendidas (sin él ambos se informan y la ejecución
sale con 0; "informado" significa que el veredicto en rojo se nombra donde lees el
resultado: un `ask` cierra con una nota `[verification: RED ...]` que lista los
bloqueadores que la reparación no despejó, y la línea final de `auto` cuenta las
tareas marcadas como hechas mientras la última ronda de compuerta estaba en rojo, igual que
cuenta las aplicaciones parciales; ../../TECHNIQUES.es.md §33).

Protecciones para ejecuciones desatendidas: una ronda de reparación que empeora la verificación (más
fallos de pruebas, más hallazgos) se revierte desde la instantánea de esa misma ronda
y la tarea se marca como fallida, en lugar de agravar el daño. Los títulos del plan
se revisan con linter al iniciar: un título debería ser un cambio, y un título que agrupa
varias ediciones recibe una advertencia. `auto_allowlist = true` restringe las escrituras de cada
tarea: los archivos existentes solo pueden modificarse, moverse o eliminarse cuando el título de
la tarea los nombra o la fase de búsqueda los leyó; crear archivos nuevos siempre
se permite.

Tres protecciones más corren sin llamada al modelo:

- **Bloqueo de interfaz** (`interface_lock`, activado por defecto). Antes de cada tarea, las
  firmas públicas del proyecto (`def parse_line(text, sep=',')`, constructores de clases,
  funciones JS/TS) se inyectan en el prompt bajo un encabezado de
  "do not break signatures", junto con una subsección de fixtures de `conftest.py`
  (firmas y alcances exactos de las fixtures) siempre que el repositorio
  tenga fixtures o la tarea toque pruebas. Después de cada tarea aplicada, cada
  importación y llamada local —tanto en Python como en las exportaciones de módulos ES de JS/TS— se
  verifica contra esas firmas. Una discrepancia hace fallar la tarea
  antes de que corra cualquier prueba, y el mensaje exacto va al prompt de reparación.
- **Compuerta estática** (`static_gate`, activada por defecto). Los archivos de Python modificados reciben una
  verificación rápida de nombres no definidos y errores de sintaxis (reglas `ruff` F821, F822,
  F823, F402, F404, con E999 para la sintaxis). Ruff tiene un techo de 3 segundos
  (`static_gate_timeout`), y un verificador integrado toma el relevo si ruff falta o es
  lento. Solo cuentan los errores que el proyecto no tenía ya. Y cuando un nuevo
  nombre no definido demostrablemente tiene exactamente una importación (un módulo de la biblioteca estándar,
  o un símbolo que solo exporta un archivo local), Pedacito inserta esa importación
  por sí mismo antes de preguntarle siquiera al modelo (`auto_heal_imports`, activado por defecto).
- **Trinquete verde** (`auto_green_ratchet`, activado por defecto). Antes de cada paso,
  Pedacito guarda una instantánea de los archivos fuente en `.pedacito/checkpoints/`. Un paso que
  aún falla tras sus rondas de reparación se revierte a esa instantánea, y los
  archivos que creó se eliminan. El siguiente paso comienza desde el último estado que pasaba,
  y el registro de Activity dice `ROLLED BACK`. Consulta
  [../../TECHNIQUES.es.md §18–§20](../../TECHNIQUES.es.md).

#### Los tres niveles de verificación, tal como se entregan

Las ediciones pasan por tres niveles, del más barato al más caro:

1. **Compuerta de sintaxis y estática (en proceso, sin llamada al modelo).** Los archivos
   de Python editados se analizan y luego se verifican con las reglas `ruff` `F821`, `F822`, `F823`,
   `F402`, `F404` y `E999`; un resolvedor de AST integrado toma el relevo cuando ruff
   falta, falla o excede su techo. El techo es
   **`static_gate_timeout = 3.0` segundos** por cada ejecución de ruff. Solo se informan los hallazgos que
   introdujo la edición: el estado previo a la edición del proyecto es la línea base,
   así que los defectos antiguos nunca vetan una tarea nueva. `auto_heal_imports = true` inserta
   la única importación que demostrablemente corrige un nuevo nombre no definido antes de que se
   le pregunte al modelo.
2. **Pruebas afectadas acotadas.** Cuando `verify_tests = true` y el subproceso de pruebas
   está permitido (`allow_shell`, o un comando de pruebas concedido por el usuario como
   `auto_verify`/`test_command`), el grafo de importaciones selecciona las pruebas que un cambio
   puede romper —las pruebas con el nombre del archivo, las pruebas de sus importadores/llamadores y
   los importadores directos— y ejecuta un `pytest FILE...`, `npm test --
   FILE...` o `go test ./dir/...` acotado. Una ejecución está limitada a **`test_timeout =
   300` segundos**. La suite completa deliberadamente no se ejecuta aquí.
3. **La suite completa es tu comando de verificación.** `auto_verify` (o `--verify` en
   una ejecución) es el comando elegido por el usuario que corre después de cada tarea de Auto; con
   `auto_verify_final = true` corre una vez después de la última tarea sin marcar en su lugar. Un fallo envía su salida a
   **`auto_repair_rounds = 2`** rondas de corrección. Después de una ejecución `auto --apply`, **`auto_verify_rounds = 2`** limita
   la verificación en tiempo de ejecución. La clave es una cadena de comando: vacía significa desactivado.

El `/verify on|off|status` de la TUI y la casilla **Post-edit verify** de la GUI
(más el campo de comando de verificación de Auto Options) controlan el nivel 3. `on` deja en paz un
comando existente; con un `auto_verify` vacío primero le pregunta
a `testsel.detect_test_command` por el ejecutor del proyecto e informa lo que
encontró. Cuando no se puede detectar nada, el comando permanece vacío, mientras que los niveles 1 y 2
y el bloqueo de interfaz siguen corriendo. El trinquete verde
(`auto_green_ratchet = true`) revierte un paso de Auto fallido a la última instantánea
verificada en verde, e `interface_lock = true` verifica las importaciones entre archivos y la aridad de las llamadas
después de cada edición.

`--worktree NAME` se ejecuta en `.pedacito/worktrees/NAME` en una rama `pedacito/NAME`,
dejando intacto tu checkout. Administra los worktrees directamente con
`pedacito worktree create|list|remove NAME`.

### git: commit y pull requests

`pedacito commit` es opcional: nunca prepara ni confirma cambios sin que lo pidas, y se
niega cuando `security.scan_text_for_secrets` encuentra una credencial en el diff preparado.
Revisa el mensaje y confirma (o usa `-y`). `--all` prepara antes todos los cambios;
`--no-model` redacta el mensaje de Conventional Commit de forma determinista, sin llamar al servidor;
`--allow-secrets` es la excepción explícita para un falso positivo confirmado. El commit es local:
Pedacito nunca ejecuta `git push`.

```bash
pedacito commit --all                       # redacta un mensaje a partir del diff preparado
pedacito commit -m "feat: add login" -y     # confirma; antes escanea igualmente
```

`pedacito pr` usa el mismo escaneo de secretos sobre el diff de la rama, reúne el registro de
commits y las etiquetas de checkpoints recientes, y redacta un título y cuerpo fundamentados.
`--dry-run` los imprime sin llamar a `gh`; `--draft` abre un PR en borrador. Requiere la CLI de
GitHub (`gh`) y nunca hace push de la rama.

```bash
pedacito pr --base main --dry-run
pedacito pr --base main --draft -y
```

La TUI expone la mitad local como `/git [status|stage PATH|unstage PATH|commit [MSG]|branch NAME]`,
y el botón **Git** de la GUI (o **Proyecto → Git: commit y pull request…**) ofrece filas para
preparar y quitar cambios, un cuadro de mensaje con **Draft message** y **Commit**, y un banner de
advertencia del escaneo de secretos. Ambas usan las mismas funciones de `gitops` y se niegan ante hallazgos.
El cajón Git de la GUI tampoco hace push nunca.

### Analizadores sin conexión: `parsers prefetch` y `diag`

```bash
pedacito parsers prefetch                          # el conjunto común de gramáticas, con conexión
pedacito parsers prefetch --languages python go    # solo estas gramáticas
pedacito diag .                                    # alias de `pedacito diagnose`
```

Las gramáticas de tree-sitter se cargan al primer uso. `parsers prefetch` las precarga e imprime
`ok` o `failed` para cada lenguaje. Una gramática que no puede descargarse se reintenta y luego
recurre al analizador integrado, así que una máquina aislada sigue funcionando. `diag` es el nombre
corto de `diagnose`, el linter o verificador de tipos detectado en el proyecto, y acepta el mismo `--file`.

### Windows PowerShell 5.1: capturar stderr (`2>&1`)

Pedacito escribe su texto para humanos en stderr, y con `--jsonl`, `rpc` y `acp` stdout debe
seguir siendo JSON puro. En Windows PowerShell 5.1, `2>&1` envuelve cada línea de stderr como un
registro de error, así que una ejecución correcta puede imprimir texto rojo `NativeCommandError`, y un
flujo JSON encadenado puede corromperse. Para una salida limpia apta para máquinas, descarta stderr
(`pedacito ask . -t "..." --json 2>$null`), o ejecuta el comando en PowerShell 7 o `cmd.exe`, que
combinan stderr como texto plano.

### `gui`: panel de control en el navegador

```bash
pedacito gui [PROJECT] [--profile NAME] [--port N] [--no-open]
```

Una página local (enlazada a `127.0.0.1`, sin CDN ni dependencias adicionales, así que funciona
sin conexión) que ejecuta el mismo motor que los comandos (`ask`, `explain`, `auto`,
`new`), de modo que la planificación, las verificaciones previas, el escalamiento y las pruebas posteriores a la edición se comportan
de forma idéntica.

| Área | Qué hace |
|---|---|
| **Encabezado** | Abre un proyecto (escribe una ruta, explora con **Browse…** las carpetas de la máquina o elige uno reciente), elige su perfil de modelo, mira la luz de estado del servidor de modelos y los totales de tokens/caché/costo de la sesión. |
| **Chat** (izquierda) | Cinco modos junto al cuadro: **Task/Edit** (editar código; *review* o *write directly*, planificación auto/always/never, máximo de búsquedas), **Ask** (preguntas de solo lectura, la respuesta llega en streaming), **Auto** (meta → plan → implementar; solo plan o implementar, comando de verificación, rondas de reparación, topes de tareas/minutos, worktree, keep-going, pip-install; aplicar Auto acepta el límite de confianza de herramientas y habilita las herramientas de escritura/ejecución para la ejecución), **New** (inicia un proyecto desde cero en una carpeta: vista previa, compilar, compilar y luego revisar, solo esqueleto, y *complex* para el ciclo autónomo) y **How ?** (la referencia integrada de CLI/GUI/ajustes con etiquetas de temas, comandos copiables y enlaces *Open in Settings*; las respuestas quedan fuera del hilo de programación guardado). **Attach** (o arrastra) un archivo de texto como tarea, pregunta o especificación. **✨ Creative** sobre el compositor fuerza la persona Builder para ejecuciones de Task/Auto/New; con él desactivado, las tareas de diseño se siguen detectando automáticamente. Mientras corre una tarea, escribe para **steer** al modelo, responde sus preguntas aclaratorias en línea o **Stop**. |
| **Changes** (centro) | Las ediciones propuestas como *Old code \| Updated code* lado a lado con resaltado a nivel de palabra. Acepta o rechaza todo el conjunto, desmarca archivos o bloques sueltos para aceptar solo algunos, o **Edit updated** para editar a mano un archivo antes de aceptar. Los archivos llevan insignias de cláusula (`R1`, `R2`, ...) que muestran a qué parte de la tarea sirven; una aceptación parcial señala las cláusulas sin aceptar. Primero se hacen respaldos y un archivo que cambió en disco desde la vista previa se rechaza. |
| **Activity** y **Plan** (derecha, arriba) | Fase y actividad en vivo de la tarea en ejecución. La lista de verificación de `auto`/plan con progreso; haz clic en **Edit** para cambiarla y luego en **Run** para implementarla. |
| **Files**, **Backups**, **Logs** (derecha, pestañas inferiores) | Files: un árbol filtrable de los archivos indexados, con los archivos de la revisión pendiente resaltados. Backups: instantáneas restaurables. Logs: la terminal, todo lo que imprime pedacito, en vivo, con colores ANSI. |
| **🌐 Preview** (derecha, pestaña inferior) | Aparece cuando el proyecto tiene páginas `.html`: un iframe de solo lectura de la página seleccionada (por defecto `index.html`), servido desde dentro de la raíz del proyecto. ⟳ Refresh la recarga, ↗ New tab la abre en una pestaña completa del navegador, y se recarga automáticamente cuando una ejecución termina o cambia archivos. Consulta [Live Preview](#live-preview). |
| **File preview** | Hacer clic en un archivo de la pestaña Files abre una vista previa modal de solo lectura. |
| **Settings** | Consulta abajo. |

Los paneles se redimensionan (arrastra los divisores), el diseño y el esquema de color se
recuerdan, y en una ventana estrecha los paneles se convierten en pestañas.

**Settings** edita la configuración de usuario o el `.pedacito.toml` del proyecto abierto
sin tocar el resto del archivo, los comentarios sobreviven, cada edición se verifica
con las mismas reglas que usa el cargador antes de escribirse y el archivo anterior
se conserva como `config.toml.bak`. Pestañas: **Profiles** (crear, duplicar, eliminar,
hacer predeterminado; URL del servidor, clave, modelo con un *Test connection* que lista los modelos
del servidor, nivel, VRAM, modo JSON y los interruptores comunes; un formulario **Dual-GPU /
Scout + Architect Pairing** escribe `default_profile` + `scout_profile` en
una sola edición que conserva los comentarios y puede crear un perfil de scout inicial para la
segunda GPU), **Model discovery**, **Auto-Tune** (hardware, clase de modelo,
tamaño de la base de código, prioridad y una pregunta de dual-GPU; una respuesta dual-GPU recomienda
un scout y ofrece *Create starter Scout & pair…*), **All settings** (cada
ajuste, con búsqueda, con su texto de ayuda y de dónde viene su valor,
aplicado globalmente o a un perfil), **MCP Servers** (agrega, edita, prueba, habilita,
confía y elimina servidores `[mcp_servers.<name>]` con nombre) y **Raw file /
reset**. Sin ninguna configuración, un formulario de primer uso (con *Find servers* para LM
Studio/Ollama) construye una. Los ajustes de ejecución de la configuración de un proyecto siguen ignorados
hasta que confíes en ella (*Review & trust…*, lo mismo que `pedacito trust`).

Cómo se mantiene seguro: rechaza las peticiones con `Host`/`Origin` ajenos, acepta
solo `POST` en JSON, envía una Content-Security-Policy estricta y sirve los archivos de un proyecto
solo desde dentro de ese proyecto. Las claves de API nunca se muestran en los formularios de perfil
(deja el cuadro en blanco para conservar una), y la pestaña **Raw file** enmascara cada
secreto almacenado como `***REDACTED***`: guardar la máscara sin cambios restaura el
valor almacenado, mientras que una máscara para una clave que el archivo no tiene se rechaza.
Los archivos sensibles (`.env*`, claves, credenciales) también se rechazan en la vista previa de archivos.
Solo corre un trabajo a la vez, y los trabajos que escriben directamente toman el mismo
bloqueo del proyecto que la línea de comandos.

### Live Preview

Cuando el proyecto abierto tiene al menos un archivo `.html`, el panel del espacio de trabajo
suma una pestaña **🌐 Preview** junto a Files / Backups / Logs: la página misma, renderizada
en un iframe de solo lectura (`#live-preview-frame`) mientras trabajas en ella.

- Elige la página en el menú desplegable (`index.html` es la predeterminada). Los enlaces relativos de CSS,
  JS e imágenes se resuelven por la misma ruta de solo lectura, así que la página
  se ve como se verá cuando se sirva.
- **⟳ Refresh** recarga la página; **↗ New tab** la abre en una pestaña completa del navegador.
- El iframe de vista previa está en sandbox (`allow-scripts`, sin acceso de mismo origen): los scripts
  se ejecutan, pero la página no puede leer el almacenamiento de la GUI ni manejar la API del agente. Usa
  **↗ New tab** cuando quieras que la página se comporte como un documento normal del mismo origen.
- La vista previa se recarga automáticamente cuando termina una ejecución de Ask/Auto/New/Apply, y
  mientras una ejecución escribe, cuando la pestaña de vista previa está abierta, de modo que una tarea de sitio web
  muestra su resultado de inmediato.

La ruta es `GET /preview/<path>` y nunca sale de la raíz del proyecto: los segmentos `..`
y los symlinks que escapan se rechazan, los archivos sensibles se rechazan igual que en la
vista previa de archivos, y solo el origen propio de la aplicación puede enmarcar la página
(`X-Frame-Options: SAMEORIGIN` más una CSP `frame-ancestors 'self'`). Es de solo lectura:
nada de lo que haga la página puede escribir de vuelta a través de ella.

### Medición y mantenimiento

```bash
pedacito calibrate [--quick] [--repeats N]   # measure the model; save a model card
pedacito calibrate --all [-y] [--recalibrate]  # every local profile, one after another
pedacito eval [--task ID] [--out report.json]   # score the harness on fixtures
pedacito map PROJ -t "fix X"       # see the map a task would get (ranked)
pedacito tests PROJ --files src/a.py   # detected test command + impacted tests
pedacito webcheck PROJ             # static contract checks for web files
pedacito domcheck PROJ             # headless DOM smoke check (needs Playwright)
pedacito memory PROJ [--add T | --forget T | --consolidate]
pedacito skills [list | show NAME | add NAME | draft "PROMPT" | rm NAME] [PROJ] [--global]
pedacito prune PROJ --keep-sessions 50 --keep-backups 20 [--dry-run]
pedacito reset PROJ [--dry-run] [--keep-memory] [--keep-config] [--reindex]
pedacito trust PROJ                # approve a project's .pedacito.toml
pedacito ask PROJ -t "..." --record          # save the model's responses
pedacito ask PROJ -t "..." --replay FILE     # re-run the harness, no model needed
pedacito ask PROJ -t "..." --replay FILE --strict-replay  # fail on prompt drift
pedacito ask PROJ -t "..." --continue        # seed the task from the newest ask checkpoint
pedacito eval --out new.json --baseline base.json         # CI A/B regression gate
pedacito eval --seed 1 --scorecard card-before.json       # scorecard schema v2 (tareas completas, semilla explícita)
pedacito bench compare card-before.json card-after.json   # puerta de scorecards: 0 aprobado, 1 regresión, 2 inválido/no comparable
pedacito eval --repeats 3 --seed 1 --out new.json   # 3 seeded runs: pass counts, mean ± spread
pedacito new app -t "..." --design           # interfaces/data/tests before code
pedacito serve --root PROJ --name web -- npm run dev
pedacito procs --root PROJ [logs|stop NAME] [--all]
pedacito lsp --root PROJ --file src/a.py --line 12 --op definition
```


`webcheck` ejecuta verificaciones estáticas de contrato entre archivos en un proyecto web, sin necesidad de navegador:
ids del DOM que no existen en ningún archivo HTML, sintaxis ESM cargada mediante un `<script src>` clásico
(un SyntaxError silencioso), globales duplicados entre scripts clásicos, archivos de `<script src>` que no existen,
importaciones simples que package.json nunca declaró, manejadores en línea que llaman a miembros que nada define,
un nombre de nivel de módulo enlazado dos veces (`Identifier 'X' has already been declared`, que mata todo el
grafo de módulos) y una llamada cuyo destinatario ningún módulo del grafo enlaza. Es a
nivel de regex a propósito, así que sus hallazgos son advertencias (impresas por `auto` tras
las tareas que tocaron archivos web, y en el resumen del andamiaje de `new`), nunca fallos
duros. Esas dos últimas formas también las impone la compuerta de examen del frontend,
que lee los mismos nombres mediante la gramática de tree-sitter en lugar de regex.

`domcheck` es la mitad en tiempo de ejecución que `webcheck` no puede ser: sirve el proyecto en un
servidor HTTP de loopback y carga la página de entrada en Chromium headless (Playwright),
demostrando que el grafo de módulos realmente se MONTA. Informa errores de página no capturados,
cargas del mismo origen fallidas y controles sin enlazar (de forma conservadora: cualquier manejador delegado
o en línea suprime el hallazgo), y puede comparar la superficie montada
con una instantánea anterior (`--snapshot before.json`, luego `--baseline
before.json`) para detectar la forma BUG-216 en la que un módulo de entrada reemplazado deja de
renderizar en silencio. También detecta los fallos SILENCIOSOS de Web Audio: instrumenta la
API (contextos, llamadas a resume, el grafo de conexiones, inicios de fuentes, envolventes de ganancia,
automatización inválida de AudioParam), luego activa un paso si la cuadrícula está vacía, hace clic en
Play y recorre los controles deslizantes, marcando un grafo que nunca llega a
`ctx.destination`, un contexto que nunca se reanudó dentro de un gesto del usuario, un botón Play
que inicia cero fuentes, ganancias fijas en cero y automatización NaN/no positiva.
Las páginas que nunca tocan Web Audio obtienen cero hallazgos de audio. Necesita `pip install
playwright && playwright install chromium`; cuando falta lo dice
en lugar de fingir que la página está bien. Ejecutar el código del proyecto es una
decisión de ejecución, así que la compuerta dentro de la ejecución necesita `verify_dom = true` Y `allow_shell = true`
(o `ask --verify-dom`); una invocación explícita de `pedacito domcheck` es en sí misma el
consentimiento. Consulta ../../TECHNIQUES.es.md #45.

`calibrate` también mide las **llamadas nativas a herramientas** contra la gramática JSON en tu
modelo (los mismos ocho escenarios, ambos modos) y la tarjeta cambia `tool_mode` a
`native` solo cuando elige la acción correcta al menos 0.1 más a menudo sin
pérdida de pasos bien formados. El modo nativo envía el esquema del paso como una sola
función forzada y lee los argumentos de la llamada a herramienta del modelo, lo que le sirve a los modelos entrenados
en llamadas a herramientas; un servidor que rechaza `tools` recurre a la gramática una vez
y lo recuerda. Sus sondas cortas (seguimiento de instrucciones, aguja de contexto largo)
reciben un presupuesto de tokens generoso y reintentan una vez con el razonamiento desactivado, de modo que un modelo
de razonamiento que gasta un presupuesto diminuto en `<think>` no se puntúe como si fallara la habilidad.
El A/B de razonamiento luego hace la misma pregunta corta de edición con el presupuesto
de respuesta real, una vez con el razonamiento activado y otra desactivado: un modelo que solo es utilizable con
el razonamiento desactivado obtiene `disable_thinking = true` persistido en la tarjeta, de modo que las
ejecuciones posteriores comiencen en el modo ganador en lugar de redescubrirlo con una respuesta vacía
(un viaje de ida y vuelta desperdiciado) a la vez.

### Telemetría en vivo

Mientras corre una tarea, el **pie de telemetría** de la GUI muestra la ventana de contexto en
uso —`Context: 6,420 / 32,768 (19.6%)` con una barra que es verde bajo 70%,
ámbar de 70–85%, roja por encima— más los tokens del turno y de la sesión, la velocidad de generación
(`Speed: 44.2 tok/s (avg 41.8 tok/s)`), la tasa de aciertos de caché KV y la cuenta de pliegues, y
el modelo activo con su especialidad (`Active: deepseek-coder (coding)`) y
el perfil orquestador. Haz clic en el pie para expandir un panel con los números completos
por llamada. Mientras un especialista ocupa la tarjeta, los números de la última llamada
son los del especialista; los totales de sesión siguen siendo los del registro compartido. El pie
también incluye el botón **Consolidate Session**: un clic destila la
transcripción en un punto de entrega y reinicia el chat desde él, y la
barra de contexto baja de inmediato al tamaño de la entrega (consulta
[Consolidación de sesión](#consolidación-de-sesión)).

La CLI imprime la misma imagen como una cinta de una línea en stderr después de cada
tarea:

```
[telemetry | ctx: ████░░░░░░ 38% (6,210/16,384) | turn: 6,210 in / 185 out | 43.5 tok/s | cache: 82%]
```

`show_telemetry = false` la oculta; `--jsonl` y los modos rpc nunca la imprimen, de modo que
los consumidores de máquina vean flujos limpios. En `chat`, `/status` y `/cost` incluyen
la barra de contexto y las velocidades, y la respuesta `status` de rpc lleva
`context_utilization_pct`, `last_tok_per_s` y `avg_tok_per_s`.

`--record` escribe las respuestas ordenadas del modelo en `.pedacito/replay/<stamp>.jsonl`
(los prompts como un hash y una vista previa breve redactada, nunca completos); `--replay`
las reproduce sin servidor, ejecutando todo el arnés (ciclo de recolección, pliegue,
verificación previa, análisis de ediciones, reparación) de forma determinista y gratuita. Informa qué
llamadas recibieron un prompt *distinto* al de cuando se grabó: así se prueba un
cambio en el arnés en lugar de en el modelo.

`pedacito eval` ejecuta 15 tareas sobre cuatro repositorios de prueba: una pequeña tienda en
Python, una aplicación TypeScript, una CLI en Go y un servicio Python mediano de varios archivos
con módulos señuelo, cada uno con un conjunto de archivos de referencia, una verificación de subcadena, una prueba real para
Python e inyecciones de prompt plantadas, e imprime la tasa de localización, el éxito de primera
aplicación, llamadas de reparación, tokens, proporción de aciertos de caché, tiempo de reloj y una
taxonomía de fallos; para ediciones: archivo equivocado, edición sin leer, error de formato, truncamiento,
rechazo, alucinación, regresión de pruebas, inyección seguida; para preguntas:
fallo de recuperación (nunca leyó el archivo de referencia), cita sin fundamento (un archivo o línea
que el proyecto no tiene), comportamiento inventado (lo leyó y aun así lo hizo
mal), rechazo excesivo (lo leyó y se negó). El resumen también informa cada
tipo (edición, pregunta) por separado, y la compuerta los verifica por separado. Úsalo para hacer A/B de cualquier ajuste:
ejecútalo, cambia `prompt_pack` o `answer_mode`, ejecútalo de nuevo. Con
`--baseline base.json`, el nuevo informe se compara con el anterior y el
comando sale con un código distinto de cero ante una regresión de calidad, más inyecciones seguidas o
un crecimiento de tokens/tiempo más allá de las tolerancias configuradas: una compuerta de CI sobre una repetición
o una ejecución en vivo. Las tasas se comparan por tarea sobre las tareas que ambos informes ejecutaron: una
caída de hasta un equivalente de tarea (una tarea, incluso en todas las repeticiones) es ruido
y solo se anota, una caída mayor falla, y una etiqueta de fallo que la taxonomía de la línea base no podía emitir se informa
como información en lugar de una regresión. `--repeats N` ejecuta el conjunto de tareas N veces, la repetición *i*
con `sampler.seed = S + i` (`--seed S`; por defecto tu propia semilla del sampler, o si no 0),
y el informe gana conteos de aprobación por tarea y la media ± desviación estándar de
llamadas, tokens, segundos y aciertos de caché entre repeticiones; la compuerta amplía su margen
al doble del error estándar cuando las propias repeticiones son ruidosas. Las repeticiones
reutilizan la caché de prompt caliente del servidor, así que `cache_hit` se lee un poco más alto que
en una sola ejecución. Dos informes que se ejecutaron con ajustes distintos de `disable_thinking`
se marcan en lugar de compararse en crecimiento: activar/desactivar el razonamiento mueve los tokens más
que cualquier cambio en el arnés, y la compuerta no debe culpar (ni bendecir) al arnés
por la configuración. `ask --replay FILE --strict-replay` hace lo mismo para la deriva
de prompts: sale con 1 cuando el arnés hace un prompt distinto al de la grabación.

### Pestaña Run de la GUI y Fix This Crash

La pestaña **Run** de la GUI previsualiza `index.html` / `public/index.html` /
`dist/index.html` en un iframe en vivo y ejecuta los puntos de entrada detectados (`python
main.py`, `pytest -q`, un lanzador de shell o un `run_cmd` personalizado en
`.pedacito/config.json`) sin shell. Una salida distinta de cero o un traceback de Python
muestra un botón **Fix This Crash** que coloca el comando exacto, el código de salida y el
traceback en el compositor de Task. En la TUI, `/run [cmd]` ejecuta los mismos
objetivos y `/fix` envía el último fallo a una tarea de reparación.

## Modo creativo / de diseño

La disciplina de edición quirúrgica (el diff más pequeño posible, muestreo frío, un presupuesto de
respuesta ajustado) es correcta para una corrección de errores e incorrecta para una landing page. Las tareas de
diseño por lo tanto corren en un modo distinto: el modelo responde como un **Senior
Product Engineer & Visual Designer**, con

- **muestreo más cálido** en servidores locales: la temperatura de respuesta es al menos
  `0.6` (`max(temperature_answer, 0.6)`), para que el modelo explore diseños y
  textos en lugar de repetir la primera idea segura;
- **espacio para terminar**: el presupuesto de respuesta tiene un piso de **8192 tokens** (tu
  `request_max_tokens_cap` sigue imponiendo el tope), para que una página completa no se corte
  a mitad de un `<div>`;
- **reescrituras de archivo completo**: se permite un reemplazo completo de `.html`/`.htm`/`.css`/`.md`,
  y el freno de reescritura excesiva en ventanas pequeñas y el veto de fragmentos
  se desactivan para las tareas creativas y los prompts de reestructuración (redesign, restyle,
  reorganize, ...). El rechazo de borrado total y la protección de exportaciones JS siguen aplicando: una
  reescritura puede reemplazar una página, nunca eliminar en silencio su sustancia.

**La detección es automática.** Con `creative_auto = true` (el valor por defecto), una tarea
es creativa cuando apunta a archivos de UI (`.html`, `.htm`, `.css`, `.svg`, `.jsx`,
`.tsx`, `.vue`, `.svelte`) o suena a trabajo de diseño (website, portfolio,
landing page, hero, dashboard, redesign, restyle, typography, color scheme)
o pide construir/crear un artefacto web o de diseño. Una tarea de backend simple mantiene
la disciplina quirúrgica incluso con la detección activada.

**Fuérzalo o desactívalo:**

| Dónde | Control |
|---|---|
| GUI | el botón **✨ Creative** sobre el compositor (ejecuciones de Task/Auto/New); apagado = detección automática |
| TUI | `/creative on` (cada tarea), `/creative auto` (detectar, el valor por defecto), `/creative off` (quirúrgico en todas partes); `/status` muestra el estado |
| CLI | `--creative` en `ask`, `auto`, `new`, `chat`, `tui` |
| Configuración | `creative_mode = true` fuerza cada tarea; `creative_mode = false` + `creative_auto = false` desactiva la detección |

En una API alojada la persona y el presupuesto de respuesta siguen aplicando, pero el muestreo
queda en manos del proveedor: no se envía ninguna clave de temperatura (consulta
[Las APIs alojadas conservan su propio muestreo](CONFIG.md#las-apis-alojadas-conservan-su-propio-muestreo)).
Combina el modo con el [Live Preview](#live-preview) de la GUI para ver cómo cambia
la página a medida que la ejecución la escribe.

## Generación local de imágenes

Si tienes un servidor de imágenes local (receta `sd-cpp` de Lemonade Server, Stable Diffusion, Flux),
`pedacito models scan` lo etiqueta con `modality = "image"` y lo muestra como **🎨 Image**. Configura
`image_profile` (o déjalo vacío para usar el primero) y genera desde donde prefieras:

```bash
pedacito image "un faro al atardecer, acuarela" --size 1024x1024 --out assets/faro.png
```

En la TUI usa `/image PROMPT [--out FILE] [--size WxH] [--steps N] [--seed N] [--profile NAME]`; en la GUI,
**Project ▾ → 🎨 Generate image…**. Sin `--out`, el archivo se escribe en `.pedacito/images/`. Para que el
agente pueda llamar a la acción `generate_image` por su cuenta, activa `allow_image_gen`
([detalles](CONFIG.md#generación-local-de-imágenes-image_profile-allow_image_gen)).

## Página de revisión

`--review` (o `review = true`) abre una página sin conexión, lado a lado: el archivo actual a la
izquierda, la versión propuesta a la derecha, con las palabras modificadas resaltadas. Para escribir, marca la
casilla de atestación, haz clic en **Sign off & write** y confirma. Enter nunca escribe.

La página registra el SHA-256 de cada archivo cuando se construye, y al aplicar se vuelve a verificar: si
editas un archivo entre tanto, la escritura se rechaza en lugar de sobrescribir
tu cambio. El servidor se enlaza solo a `127.0.0.1` y rechaza los encabezados `Host`
y `Origin` ajenos.

Cuando la ejecución rastreó cláusulas de requisitos, cada archivo lleva una insignia (`R1`,
`R2`, ...) que nombra las cláusulas a las que sirvieron sus ediciones; el título de la insignia es el texto de la
cláusula. Si aceptas solo algunos archivos, cualquier cláusula cuyos archivos hayan sido todos rechazados
se marca como sin aceptar en el detalle de la firma, de modo que una aceptación parcial no pueda
descartar en silencio la mitad de la tarea.

**Firma por bloque.** En el panel Changes de la GUI, cada bloque de cambio de un archivo aparece como **Bloque N** (Hunk N), con una casilla *aceptar solo este bloque*. Al aceptar algunos bloques solo se escriben esos; el resto del archivo conserva su texto actual. La verificación SHA-256 sigue cubriendo todo el archivo antes de escribir nada.
