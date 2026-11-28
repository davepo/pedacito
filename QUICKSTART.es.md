🌐 [English](QUICKSTART.md) | **Español**

# Inicio rápido de Pedacito

Pedacito permite que un modelo local pequeño trabaje sobre una base de código mucho más grande que su
ventana de contexto. Construye un índice de tu proyecto y luego deja que el modelo obtenga
solo las piezas que necesita para responder una pregunta o proponer una edición. Para el
contexto general, consulta el [README](README.es.md). Para saber cómo funcionan las técnicas para modelos pequeños,
consulta [TECHNIQUES.es.md](TECHNIQUES.es.md). Para elegir un modelo, consulta la
[guía de modelos](MODEL_GUIDE.es.md). La referencia canónica de ajustes, comandos y
seguridad está en [docs/es/](docs/es/), empezando por [docs/es/CONFIG.md](docs/es/CONFIG.md). Las skills personalizadas, los servidores MCP
y los hooks están en [docs/es/SKILLS_AND_TOOLS.md](docs/es/SKILLS_AND_TOOLS.md).

Hay dos formas de entrar, y ambas ejecutan el mismo motor y leen la misma configuración:

- **[Parte A: la GUI](#parte-a-la-gui).** Un panel de control en tu navegador. Ideal si
  quieres ver diffs, planes y ajustes lado a lado.
- **[Parte B: la CLI](#parte-b-la-cli).** Comandos en una terminal. Ideal para
  scripts, editores y tareas rápidas y puntuales.

Todos los comandos funcionan igual en Windows, Linux y macOS. Usa `pedacito`, el
alias más corto `pdc`, o `python -m pedacito` (`py -m pedacito` en Windows) si
ninguno está en tu `PATH`.

---

## Cambiar el idioma (EN | ES)

Todo lo que sigue funciona en inglés o en español. En la GUI usa el selector **EN | ES** de la
barra superior. En la TUI presiona `F3` o escribe `/lang es` (y `/profiles` lista tus perfiles). En la
línea de comandos pasa `--lang es` (por ejemplo `pedacito --lang es gui`) o define
`PEDACITO_LANG=es`. Los ajustes `language` y `llm_reply_language` hacen permanente la elección;
consulta [docs/es/CONFIG.md](docs/es/CONFIG.md#idioma-y-localización-language-llm_reply_language) y
[docs/es/USAGE.md](docs/es/USAGE.md#cambiar-el-idioma-en--es).

## Antes de empezar

### 1. Un servidor de modelos

Pedacito se comunica con cualquier servidor que tenga una API `/v1` compatible con OpenAI.

- **LM Studio:** carga un modelo y luego inicia el servidor desde la pestaña Developer. El
  puerto por defecto es `1234`. Para usarlo desde otra máquina, activa **Serve on Local
  Network**.
- **Ollama:** ejecuta `ollama serve` y luego `ollama pull <model>`. El puerto por defecto es
  `11434`. Para otra máquina, define `OLLAMA_HOST=0.0.0.0` antes de iniciarlo.
- **Lemonade Server:** inicia el servidor y carga un modelo. Usa la URL base que
  muestra.
- **llama.cpp:** inicia `llama-server` con tu modelo. Usa la URL que imprime.
- **Una API alojada** (Claude, ChatGPT, DeepSeek, Qwen, Gemini, OpenRouter): necesitas
  una clave de API. Colócala en una variable de entorno, como `DEEPSEEK_API_KEY`, y
  Pedacito la lee desde ahí para que la clave no quede en tu archivo de configuración.

Los puertos son valores por defecto, no reglas. El escáner identifica cada servidor por sus respuestas,
así que dos servidores pueden ejecutarse en una misma máquina.

¿No sabes qué tamaño de modelo cargar? La [referencia rápida de VRAM](docs/es/INSTALLATION.md#qué-modelo-se-ajusta-a-tu-gpu)
relaciona las tarjetas de 6–8 GB, 12–16 GB y 24 GB+ con los tamaños de modelo que mejor funcionan en
cada una. Empieza con un modelo local; agrega un planificador o verificador alojado más tarde si
lo deseas, y es opcional.

### 2. Instalar

Requiere Python 3.11 o superior.

```bash
cd /path/to/pedacito
pip install -e .            # Windows: py -m pip install -e .
```

Ese único comando instala todo lo que necesita el núcleo: el fragmentador AST de tree-sitter
y sus gramáticas (la verificación exacta de sintaxis antes de cada edición), además de `pytest` y
`ruff` para la verificación, y `requests` para la conexión con el modelo. No se necesita nada más
para la configuración por defecto. `pip install -r requirements.txt` instala el
mismo conjunto núcleo sin instalar Pedacito.

`-e` significa "editable": un `git pull` posterior surte efecto sin reinstalar.
Extras opcionales: `pip install -e ".[full]"` agrega capturas de pantalla y el navegador
headless, o elige uno con `pip install -e ".[vision]"` (capturas de pantalla) o
`pip install -e ".[browser]"` para el navegador headless (luego ejecuta
`python -m playwright install chromium`).

---

## Parte A: la GUI

### A1. Abre el panel de control

```bash
pedacito gui ~/code/your-project
```

Pedacito inicia un servidor local en `127.0.0.1` y lo abre en tu navegador. La
dirección que imprime es la que debes usar si la pestaña no se abre sola. Pasa
`--port 8787` para fijar el puerto, `--no-open` para no lanzar un navegador y
`--profile NAME` para elegir un perfil de modelo para el proyecto.

Si aún no tienes un archivo de configuración, aparece un formulario de primer uso. Su botón
**Find servers** busca LM Studio y Ollama en esta máquina.

### A2. Conecta un modelo

Puedes agregar modelos de dos maneras. Ambas escriben perfiles en tu configuración, así que solo haces
esto una vez por modelo.

**Desde un servidor local.** Abre **Settings** (el engranaje) y ve a **Model
discovery**. Haz clic en **Scan local servers**. El escaneo está limitado a 20 segundos y
devuelve lo que haya encontrado cuando se agota el presupuesto, así que un servidor lento no puede
bloquearlo. Pedacito pregunta cuánta memoria de GPU tiene la
máquina que ejecuta el servidor, en GB. Luego marca solo los modelos que quieras. El
cuadro de nombre te permite renombrar un perfil. Haz clic en **Import Selected**. Cada modelo se convierte en un
perfil con su nombre, y uno queda como predeterminado. Cada perfil de la lista
tiene un botón **Calibrate** que ejecuta una medición rápida de ese modelo, y una
insignia de calibración: ✓ Calibrated (con la fecha), ⚠ stale u ○ Not calibrated.
Los perfiles de proveedores alojados muestran ☁ Hosted en su lugar, y su botón Calibrate está
deshabilitado: la calibración es solo para modelos locales; los modelos alojados usan
automáticamente los valores por defecto de la tabla del proveedor.

**Desde un proveedor alojado.** En la misma pestaña, haz clic en **+ Add Hosted Provider**.
Elige un preajuste: **Claude (Anthropic)**, **ChatGPT (OpenAI)**, **DeepSeek**,
**Qwen (DashScope)**, **Google Gemini** u **OpenRouter**. Proporciona la clave, o
indica la variable de entorno que la contiene. Pedacito se conecta al proveedor,
lista sus modelos e importa los que marques, un perfil por cada uno.

De vuelta en la barra superior, el selector de perfil cambia el modelo para la siguiente tarea.
Su luz de estado muestra si el servidor es accesible. La etiqueta **no-think** o
**thinking** junto a él indica si se permite el razonamiento para ese perfil,
y una insignia de calibración muestra si el modelo activo tiene una tarjeta de modelo vigente.

La lista de modelos escaneados se puede contraer por servidor (haz clic en el encabezado de un servidor) y cerrar con **Close**, así un escaneo largo nunca empuja el resto de **Model discovery** fuera de la pantalla. Los generadores de imágenes muestran una insignia **🎨 Image** y un botón **Probe image**, que ejecuta una comprobación rápida de 512×512.

Para medir todos los modelos locales a la vez, abre **Project ▾ → Calibrate All Local
Models…** (también en **Settings → Model discovery**). Una confirmación lista los
perfiles locales (los alojados se excluyen) y luego un trabajo en segundo plano carga
y mide cada uno por turno, marcándolos a medida que se guardan sus tarjetas. Marca
**Re-calibrate models that already have a fresh card** para rehacerlos todos.

### A3. Ejecuta el asistente Auto-Tune

Los ajustes son los números que deciden cuánto del contexto de un modelo usa Pedacito
para lecturas, pasos y verificación. El asistente los define a partir de tu
hardware, no a base de suposiciones.

1. Abre **Settings → Auto-Tune**.
2. Responde cuatro preguntas: tu hardware, la clase de modelo, el tamaño de tu
   base de código y qué te importa más (velocidad o exhaustividad).
3. Lee la vista previa. Lista cada presupuesto recomendado y la razón. Aún no
   se escribe nada, y la vista previa no usa tokens.
4. Haz clic en **Apply Recommended Universal Settings**. El archivo anterior se conserva como
   `config.toml.bak`.

Puedes ejecutar el asistente de nuevo cuando cambies tu GPU o tu modelo.

### A4. Activa o desactiva el razonamiento para un perfil

Algunos modelos razonan antes de responder. Eso ayuda en problemas difíciles pero puede consumir
el presupuesto de respuesta de un modelo pequeño. Defínelo por perfil:

1. Abre **Settings → Profiles** y selecciona el perfil.
2. En **Reasoning & Execution Overrides**, define `disable_thinking`.
3. Guarda. La etiqueta de la barra superior cambia de **thinking** a **no-think**.

Para un modelo que necesita un interruptor en el prompt, define `thinking_suffix`, por
ejemplo `/no_think`. [MODEL_GUIDE.es.md](MODEL_GUIDE.es.md#razonamiento-por-perfil)
explica cuándo ayuda cada ajuste.

### A5. Pide un cambio (Task/Edit)

1. Elige tu perfil en la barra superior y selecciona el proyecto si no está abierto.
2. Elige **Task/Edit** y describe el cambio. Sé específico y nombra un archivo o
   función cuando conozcas uno. Por ejemplo: `make the retry count in
   client.py configurable`.
3. Revisa las opciones junto al cuadro:
   - **Writing:** *Review before writing* (el valor por defecto) o *Write directly
     (backup first)*.
   - **Planning:** *Automatic* divide una petición de varias partes en pasos. *Always
     plan* y *Never* lo anulan.
   - **Max lookups:** un tope de cuántas lecturas puede hacer el modelo. `0` usa
     el valor de la configuración.
4. Haz clic en **Run**. El panel **Engine V2** muestra la fase actual, el libro de
   requisitos (una línea por cláusula de tu petición, desde sin ubicar hasta verificada) y
   las búsquedas recientes.
5. Cuando termina la ejecución, el panel **Changes** muestra el código antiguo y el
   nuevo de cada archivo lado a lado, con las palabras modificadas resaltadas. Puedes aceptar o
   rechazar todo el conjunto, desmarcar archivos individuales o hacer clic en **Edit updated** para
   cambiar la versión nueva a mano antes de aceptar. Cada archivo muestra insignias de cláusula
   (`R1`, `R2`, ...) para las partes de tu petición a las que sirve; si aceptas
   solo algunos archivos, se señala cualquier cláusula que quede sin aceptar. Cada bloque de cambio dentro de un archivo es un bloque numerado que puedes aceptar por separado en el panel Changes.

Pedacito respalda cada archivo antes de escribir. Si un archivo cambió en disco después de
la vista previa, la escritura se rechaza, así que tus propias ediciones nunca se sobrescriben.

**¿Poco detalle?** Escribe una idea de una línea y haz clic en **✦ Refine Prompt** sobre el
compositor. Pedacito reemplaza la línea con una especificación completa: un objetivo, los archivos a
crear o modificar (con rutas reales de tu proyecto), contratos de interfaz,
requisitos y casos límite, y verificación. Léela, edítala si quieres y luego
haz clic en **Run**. **↶ Undo Refine** restaura tu texto original. Arrastra el borde superior del
compositor para hacerlo más alto mientras lees.

**¿Construyes algo visual?** Haz clic en **✨ Creative** sobre el compositor para forzar el
modo creativo/de diseño (Builder) en la siguiente ejecución de Task/Auto/New: una persona de Senior Product
Engineer y Diseñador Visual, muestreo más cálido y reescrituras de HTML/CSS de archivo completo
en lugar de la disciplina del diff más pequeño. Con el interruptor apagado, las tareas de diseño
(sitios web, rediseños, trabajo en `.html`/`.css`/componentes de UI) se siguen
detectando automáticamente: eso es `creative_auto`; `creative_mode = true` en la configuración
lo fuerza para cada tarea (A21).

**Interruptores de enrutamiento y verificación.** **✦ Auto-route** (enviar cada tarea al modelo especialista cuyas etiquetas encajen) y **Post-edit verify** (ejecutar tu comando `auto_verify` tras cada tarea Auto) son casillas en la fila del compositor, junto a **✨ Creative**. La barra superior solo conserva el selector de perfil y el selector `EN | ES`.

**Genera una imagen.** Con un perfil de imagen configurado, abre **Project ▾ → 🎨 Generate image…**, escribe un prompt, define opcionalmente tamaño, pasos, semilla o una ruta de guardado y haz clic en **Generate**. El archivo se guarda en `.pedacito/images/` y se previsualiza en el diálogo.

### A6. Haz una pregunta (Ask)

Elige **Ask** para preguntar cómo funciona el código. La respuesta llega en streaming y no se
escribe nada. Úsalo para entender un módulo antes de cambiarlo.

### A7. Ejecuta un plan autónomo (Auto)

Elige **Auto** y luego describe una meta, como `add CSV export to the report
page, with tests`.

- Marca **Plan only** en las opciones para recibir primero una lista de verificación. Puedes
  editarla en el panel **Plan** (**Edit**, luego **Save**) y después hacer clic en **Run**.
- Sin **Plan only**, Pedacito planifica y luego aplica y verifica cada tarea.
  Las opciones incluyen un comando de verificación (por ejemplo `pytest -q`), rondas de reparación, un
  tope de tareas y minutos, un git worktree separado y continuar tras
  una tarea fallida.

Auto escribe archivos y ejecuta comandos sin preguntar por cada uno. Iniciar una
ejecución Auto es tu aprobación de esas herramientas para esa ejecución, así que usa **Plan only** hasta que
confíes en el proyecto. Una ronda de reparación que empeora las cosas se revierte
automáticamente.

Cada paso también se verifica antes de que comience el siguiente. Pedacito compara cada
firma de función y clase del proyecto con las llamadas e importaciones entre
archivos, y ejecuta una verificación rápida de nombres no definidos, de modo que un paso que rompe un nombre se
detecta antes de que corran sus pruebas. En un proyecto web también verifica que cada script
y hoja de estilos local exista y que cada id del DOM que busca un script esté en la página. Si un
paso aún falla después de sus rondas de reparación, el
trinquete verde lo revierte al último estado que pasaba. El registro de Activity entonces
muestra `ROLLED BACK`, con los archivos que se restauraron o eliminaron.

### A8. Inicia un proyecto nuevo (New)

Elige **New**, describe el proyecto y elige una carpeta. Puedes previsualizar el
árbol de archivos, construirlo, construirlo y revisarlo, o detenerte tras el esqueleto. La opción
*complex* entrega la implementación al ciclo autónomo.

### A9. Pregunta cómo hacer algo (How ?)

Elige **How ?** para preguntar dónde vive un ajuste o función, o cómo ejecutar algo
desde la CLI. La respuesta viene con comandos copiables y botones **Open in Settings**.
Las respuestas de How se mantienen fuera de tu hilo de programación, y funcionan incluso antes de que se
configure un modelo.

### A10. Compara modelos con los benchmarks

Haz clic en **▦ Benchmarks** en la barra superior. Elige un perfil y, opcionalmente, una carpeta
para la salida, luego haz clic en **Run Benchmark**. Cada ejecución construye uno de seis proyectos pequeños
con el ciclo normal de Pedacito y luego lo califica con una rúbrica oculta de evaluación. El informe completo es
`BENCHMARK_REPORT.md` en la carpeta del nuevo proyecto. Una
ejecución puede tardar varios minutos. [MODEL_GUIDE.es.md](MODEL_GUIDE.es.md#compara-modelos-con-los-benchmarks-de-diagnóstico)
describe los seis proyectos y cómo leer los resultados. El benchmark del pelícano
también dibuja el SVG que produjo el modelo, en línea dentro de la tarjeta de puntuación.

### A11. Deshaz una escritura

En el panel derecho, abre la pestaña **Backups**. Lista las instantáneas de
las escrituras de Task/Edit y Auto. Elige una y haz clic en **Restore**. Los archivos que cambiaron
desde la instantánea se sobrescriben, y los archivos creados desde entonces permanecen en disco.

### A12. Aprueba o deniega una solicitud de permiso

A veces el modelo pide algo que aún no has permitido: acceso web, la instalación de un
paquete, un comando de shell que no está en la lista permitida o un archivo sensible
como un `.env`. La ejecución se pausa y un diálogo **Permission requested**
muestra qué quiere, por qué y el objetivo exacto.

- **Allow Once** deja pasar solo esta solicitud.
- **Allow for This Run** detiene los avisos de ese tipo de solicitud hasta que la ejecución
  termine.
- **Always Allow** los detiene de ahora en adelante. Para el acceso web activa
  `enable_web`, y para los comandos de shell activa `allow_shell`.
- **Deny & Redirect** rechaza la solicitud, y el modelo continúa con los archivos
  locales del proyecto.

Si no respondes, la solicitud se deniega cuando el diálogo expira o presionas
Stop. Una denegación nunca hace fallar la ejecución. La CLI no tiene diálogo, así que allí una herramienta
restringida informa que está deshabilitada en la configuración.

### A13. Activa el acceso web (opcional)

El acceso web está desactivado por defecto. Dos herramientas integradas lo usan: `fetch_url` lee una página
web y `web_search` busca documentación. Para permitirlas sin preguntar cada
vez, abre **⚙ Settings → All settings**, busca `enable_web`, actívalo
para cada perfil y guarda. O déjalo desactivado y aprueba cada llamada en el
diálogo de permisos. Las direcciones de loopback, link-local y de red privada se rechazan
a menos que las incluyas en `fetch_allowed_hosts`. Un nombre público que *resuelve* a una
dirección privada también se rechaza (DNS rebinding), y la conexión se fija a
la IP verificada para que una segunda respuesta DNS no pueda cambiarla a mitad de la petición.

### A14. Otras cosas que debes saber

- **Settings** edita tu configuración de usuario o el `.pedacito.toml` del proyecto. Tus
  comentarios sobreviven, y cada edición se verifica antes de guardarse. La pestaña **Raw
  file / reset** muestra el archivo con cada secreto almacenado enmascarado como
  `***REDACTED***`; guarda la máscara sin cambios y se conserva la clave almacenada.
- **Servidores MCP:** **⚙ Settings → MCP Servers** agrega y administra servidores de herramientas
  `[mcp_servers.<name>]` con nombre (probar, habilitar, confiar, eliminar) sin
  editar TOML a mano.
- **Project ▾** también contiene **Calibrate All Local Models…** y los diálogos de
  mantenimiento: **Memory & repo lessons…**, **Skills library & creator…** (A19),
  **Consolidate session…** (A20), **Background servers…**, **Run project
  checks…**, **Prune old sessions/backups…** y **Reset project state…**
  (prune y reset muestran una vista previa antes de escribir).
- **Pie de telemetría:** bajo el espacio de trabajo, una barra en vivo muestra la ventana
  de contexto en uso (verde bajo 70%, ámbar hasta 85%, rojo por encima), los tokens del turno y de la sesión,
  la velocidad de generación, los aciertos de caché KV y los pliegues, y el modelo activo.
  Haz clic para expandir el panel completo. Su botón **Consolidate Session**
  compacta un chat largo en un punto de entrega (A20). La CLI imprime los
  mismos números como una cinta de una línea después de cada tarea; `show_telemetry = false`
  la oculta.
- **Temas:** abre el Theme Manager con el botón de paleta de la barra superior. Ofrece
  nueve esquemas integrados (`slate`, `light`, `daylight`, `nord`, `midnight`,
  `jazz_solo`, `trapper_keeper`, `memphis_arcade`, `windbreaker`) y cualquier tema
  personalizado que guardes: el mismo conjunto que ofrece `/theme` en la TUI. La elección se guarda
  en `themes.json` junto a tu configuración.
- **Adjunta** un archivo de texto, o arrástralo al cuadro del chat, para usarlo como tarea, como
  pregunta o como especificación.
- **Steer:** mientras corre una tarea, escribe una nota para redirigir al modelo. También puedes
  responder sus preguntas aclaratorias en línea, o hacer clic en **Stop**.
- **Terminal:** el panel inferior muestra todo lo que imprime Pedacito, en vivo.
- **🌐 Preview:** cuando el proyecto tiene páginas `.html`, una pestaña Live Preview se une a
  Files / Backups / Logs y renderiza la página en un iframe de solo lectura. Se recarga
  cuando termina una ejecución —y mientras una ejecución escribe— para que veas el resultado
  de inmediato (A22).

### A15. Renombra un símbolo en todo el proyecto

Renombrar una función o variable en todos los lugares donde se usa no requiere un modelo.

1. Abre **Project ▾** en la barra superior y elige **Rename symbol…**.
2. Escribe el nombre actual en **Rename**, el nuevo nombre en **To** y, opcionalmente, un
   archivo en **Scope file** para limitar el renombrado a ese archivo y a los archivos que lo
   importan.
3. Haz clic en **Preview**. Aparecen los diffs por archivo y aún no se escribe nada.
4. Haz clic en **Rename** para escribir el cambio. Los originales se respaldan primero, así que
   puedes deshacerlo desde **Project ▾ → Restore from a backup…**.

Las cadenas y los comentarios que casualmente contienen el nombre antiguo se dejan intactos. Un parámetro
de una función no relacionada que tiene el mismo nombre también se renombra, así que lee la vista previa.
En la CLI, `pedacito rename OLD NEW --strings` también renombra las apariciones de palabra completa
dentro de literales de cadena (los comentarios no se tocan), y `--include-labels` mueve solo las
cadenas de etiquetas de prueba (`test`/`it`/`describe`/`suite`); el diálogo de la GUI no tiene esa opción.

### A16. Deja que el modelo ejecute una prueba acotada

Mientras investiga, el modelo puede ejecutar una sola prueba acotada y leer el traceback
en vivo antes de escribir una corrección. Los objetivos de Python se ejecutan con pytest; los archivos `*.test.*` /
`*.spec.*` se ejecutan con Vitest (o el ejecutor integrado de Node), `*_test.go` con
`go test` y `tests/*.rs` con `cargo test`. Está activado por defecto. Nada se
ejecuta a través de un shell, y cada ejecución está limitada a 15 segundos.

Si prefieres aprobar cada ejecución, abre **⚙ Settings → All settings**, busca
`enable_run_test_in_gather`, desactívalo y guarda. Cada ejecución de prueba aparece entonces en el
diálogo **Permission requested** bajo **Test Execution**.

### A17. Usa un modelo scout para las búsquedas (dos GPU)

Si tienes un modelo pequeño en una segunda GPU, o un servidor ligero, deja que haga las
búsquedas mientras el modelo grande planifica y edita.

El camino rápido: **⚙ Settings → Profiles → Dual-GPU / Scout + Architect Pairing**.
Elige el arquitecto (tu perfil de planificador/editor) y un scout, o elige
**➕ Create a starter Scout profile…** e indica la URL del segundo servidor, el id del modelo, la
VRAM y el nivel. **Save pairing** escribe `default_profile` + `scout_profile` en una sola
edición que conserva los comentarios. **⚙ Settings → Auto-Tune** también hace una pregunta sobre dual-GPU
y, cuando una segunda tarjeta lo permite, ofrece **Create starter Scout & pair…**.

El camino manual:

1. Agrega el modelo pequeño como su propio perfil en **⚙ Settings → Profiles**, con su propia
   dirección de servidor y modelo.
2. Abre **⚙ Settings → All settings**, busca `scout_profile` y elige ese
   perfil. Elige **None (use active profile)** para desactivar el scout.
3. Mantén el modelo grande como perfil activo. La barra superior mostrará entonces una
   insignia `Scout: <profile>` junto a la etiqueta del modelo.

Si no se puede alcanzar al scout, Pedacito recurre al modelo activo y muestra una
advertencia. Los tamaños a combinar, y la salvedad de una sola GPU, están en la
[guía de modelos](MODEL_GUIDE.es.md#dual-gpu--dual-model-enrutamiento-scout--arquitecto-scout_profile).

### A18. Enruta cada tarea a un modelo especialista

Un scout divide una tarea por *trabajo*; el enrutamiento por especialista divide una ejecución por *tarea*: el
perfil de código toma "refactor the parser", el perfil de visión toma "fix the button
layout", el razonador toma "optimize the sorting algorithm".

1. Etiqueta tus perfiles: **⚙ Settings → Profiles → [selecciona uno] → Connection &
   Capabilities → specialties**: activa las etiquetas (Coding, Frontend / UI,
   3D / graphics / SVG, Math / algorithms, Testing / QA, SQL / data,
   Translation / i18n), o escribe una etiqueta personalizada en minúsculas y presiona Enter. Déjalo
   vacío para que lo decida la tarjeta del modelo: `pedacito calibrate` detecta etiquetas a partir del
   nombre del modelo, la biblioteca de arquitecturas y seis microsondas (SVG, matemáticas,
   traducción, SQL, testing/QA, planificación), y el formulario muestra lo que detectó.
2. Activa el enrutamiento con la casilla **✦ Auto-route** de la barra superior (o
   `auto_route_specialties = true` en la configuración).

Los cambios ocurren entre tareas, nunca a mitad de una investigación. En una sola GPU el
modelo saliente se descarga antes de que cargue el especialista, así que ambos caben en la VRAM uno después del
otro; con dos GPU (o con un especialista alojado) ambos permanecen cargados. El panel de Activity
muestra qué especialista está en ejecución, y uno inalcanzable recurre al
modelo activo con una advertencia. En la CLI: `--auto-route` por ejecución, `--no-auto-route`
para anular la configuración.

### A19. Explora y crea skills

Las skills son listas de verificación cortas que el modelo sigue para un tipo de tarea. Vienen
diecinueve iniciales incluidas (testing, refactorización, endpoints de FastAPI, canvas/SVG,
migraciones de SQLite, seguridad asíncrona, endurecimiento de seguridad, i18n, parsers,
profiling, diseño web, GUIs de escritorio y juegos, …); las 2 mejores coincidencias de disparadores
para tu tarea se inyectan automáticamente, con un tope de 1,200 caracteres.

1. Abre **Project ▾ → Skills library & creator…** (también enlazado en
   **⚙ Settings → Quick links**). Cada skill muestra una insignia de alcance:
   **bundled**, **user** (todos tus proyectos) o **project** (este repositorio).
2. Escribe una tarea en el cuadro de vista previa y presiona Enter para ver qué skills
   inyectaría; haz clic en cualquier skill para leer sus disparadores y su lista de verificación.
3. Haz clic en **+ New Custom Skill**, completa el nombre (un slug), la descripción,
   los disparadores separados por comas y el cuerpo de la lista de verificación; o simplemente escribe la
   descripción y haz clic en **✦ Auto-Draft from Description** para que el modelo activo
   redacte toda la skill. Revisa el borrador y luego **Save to Project**
   (`.pedacito/skills/<name>.md`) o **Save to Global**
   (`~/.config/pedacito/skills/<name>.md`).
4. Las skills personalizadas se pueden editar o eliminar desde el mismo diálogo; las integradas
   son inmutables. Una skill de proyecto puede eclipsar una integrada del mismo nombre.

Los equivalentes en la CLI son `pedacito skills list|show|add|draft|rm` (B13).

### A20. Consolida una sesión larga

Cuando un chat se alarga, consolídalo: la transcripción se destila en una
entrega de menos de 400 tokens con cuatro secciones —Goal & Context, Files
Modified & Key Decisions, Verified Test Status, Open Next Steps— guardada en
`.pedacito/consolidated_session.md`. Las decisiones duraderas se fijan en
`memory.md`.

1. Haz clic en **Consolidate Session** en el pie de telemetría (o en
   **Project ▾ → Consolidate session…**) y confirma. Primero hay que
   detener un trabajo en ejecución.
2. El chat se reinicia desde la tarjeta única de entrega, y el Context Meter
   baja de inmediato al tamaño de la entrega: el siguiente turno empieza pequeño.

En la CLI: `/consolidate` (alias `/compact`) dentro de `pedacito chat`, o
`pedacito memory PROJ --consolidate` sobre las transcripciones de sesión más recientes.
Mediante RPC: `{"id": 1, "op": "consolidate"}`.

### A21. Modo creativo / de diseño (la persona Builder)

Una corrección de errores quiere el diff más pequeño posible; una landing page quiere un diseñador.
Pedacito los distingue automáticamente: con `creative_auto = true` (el valor
por defecto), una tarea que apunta a archivos de UI (`.html`, `.htm`, `.css`, `.svg`,
`.jsx`, `.tsx`, `.vue`, `.svelte`) o que suena a trabajo de diseño (sitio web,
portafolio, landing page, hero, dashboard, rediseño, restyle) se ejecuta como un
**Senior Product Engineer & Visual Designer**:

- la temperatura de respuesta sube a al menos `0.6` en servidores locales, para que el
  modelo explore en lugar de repetir la primera idea segura;
- el presupuesto de respuesta tiene un piso de 8192 tokens (tu `request_max_tokens_cap`
  sigue imponiendo el tope), para que una página completa no se corte a mitad de un `<div>`;
- se permiten reescrituras de `.html`/`.css` de archivo completo: el veto de fragmentos y el
  freno de reescritura excesiva en ventanas pequeñas se desactivan (el rechazo de borrado total y la
  protección de exportaciones JS siguen vigentes).

Para forzarlo en **cada** tarea, haz clic en **✨ Creative** sobre el compositor, pasa
`--creative` en la CLI, escribe `/creative on` en la TUI o define
`creative_mode = true` en la configuración. `/creative auto` restaura la detección y
`/creative off` (ambos ajustes en false) mantiene ediciones quirúrgicas en todas partes;
`/status` muestra el estado actual. En una API alojada la persona y el presupuesto
siguen aplicando, pero el proveedor conserva su propio muestreo (consulta "Varios modelos:
perfiles" más abajo).

### A22. Live Preview: mira la página mientras la construyes

Cuando el proyecto abierto tiene al menos un archivo `.html`, el panel del espacio de trabajo
suma una pestaña **🌐 Preview** junto a Files / Backups / Logs.

1. Abre la pestaña y elige la página en el menú desplegable (`index.html` es la
   predeterminada). Los enlaces relativos de CSS/JS/imágenes se resuelven por la misma ruta
   de solo lectura, así que la página se ve como se verá cuando se sirva.
2. Ejecuta una tarea de sitio web. La vista previa se recarga automáticamente cuando la ejecución
   termina, y mientras escribe, si la pestaña está abierta.
3. **⟳ Refresh** recarga la página a mano; **↗ New tab** la abre en una pestaña
   completa del navegador.

La vista previa es de solo lectura y nunca sale de la raíz del proyecto: los segmentos `..` y
los symlinks que escapan se rechazan, los archivos sensibles se rechazan y solo el origen
propio de la aplicación puede enmarcar la página. Combínala con ✨ Creative (A21) para el ciclo
de diseño completo.

---

## Parte B: la CLI

### B1. Verifica la conexión

Configuración inicial. Esto encuentra servidores locales, lista sus modelos y escribe un
perfil para cada uno:

```bash
pedacito config --setup
```

Luego verifica la conexión:

```bash
pedacito config --check
```

```
Connecting to http://localhost:1234/v1 ...
  reachable. Models available: qwen3.5-9b
  'qwen3.5-9b' is listed. Testing that it actually generates...
  generation works. You're good.
```

Si falla, hay dos casos. **Not reachable** significa que el servidor no está en
ejecución, el puerto es incorrecto o (en un servidor remoto) no está escuchando en la
red. **Reachable, but the model is not listed or will not generate** significa que
`model =` no coincide con el nombre que usa el servidor (cópialo de la salida de
`--check`), o que el servidor no pudo cargar el modelo, normalmente porque no hay
suficiente VRAM libre.

Una vez por servidor, ejecuta también:

```bash
pedacito config --doctor
```

Repite la verificación y luego sondea qué modos de salida estructurada admite el servidor,
recomienda un `json_mode`, informa la ventana de contexto real y verifica
si tree-sitter está activo. Las respuestas se recuerdan por una semana.

Después de actualizar Pedacito, `pedacito config --upgrade` agrega a tu configuración las
opciones recién documentadas, comentadas, sin tocar tus
ajustes ni comentarios existentes. `pedacito config --upgrade --dry-run` lista
lo que se agregaría sin escribir.

Pedacito también te avisa cuando existe una versión más reciente: una verificación anónima a PyPI
al iniciar (en caché 24 h, silenciosa sin conexión). `pedacito update` muestra
la versión instalada frente a la última, `pedacito update --run` actualiza con el comando de
tu método de instalación (pipx, venv o pip --user; las copias editables reciben instrucciones de
`git pull`), la GUI muestra un banner descartable con actualización en streaming
y Restart, y la TUI tiene `/update`. Define `check_for_updates = false`
para desactivarlo por completo: cero llamadas de red, todo lo demás sin cambios.

Inspecciona la máquina y, si quieres, deja que Pedacito ajuste el perfil a partir de ella:

```bash
pedacito profile            # RAM/VRAM with a source per number, last probe, tuning
pedacito profile --run      # force a prefill/generation speed probe
pedacito profile --reset    # drop cached measurements and the auto_tune opt-in
```

`auto_tune` está desactivado por defecto. Define `auto_tune = true` en un perfil para ajustar
`context_tokens`, los presupuestos de recuperación y el tiempo límite a la VRAM/RAM libre; los valores
explícitos siempre ganan y, si falta telemetría, auto-tune se abstiene. Consulta
[Perfilado avanzado del sistema y auto-ajuste](docs/es/CONFIG.md#perfilado-avanzado-del-sistema-y-auto-ajuste).

### B2. Encuentra modelos y mídelos

```bash
pedacito models scan
```

Esto encuentra servidores locales en cualquier puerto, te deja elegir modelos, escribe un perfil
por modelo y mide cada uno con una calibración rápida. La calibración
sondea la elección de herramientas, la salida JSON, el formato de edición, la resistencia a inyecciones de
prompt plantadas, la honestidad sobre lo que se leyó, la recuperación en contexto largo y si el
razonamiento oculto consume el presupuesto de respuesta. También detecta las
**especialidades** del modelo (heurísticas de nombre, la biblioteca de arquitecturas y seis
microsondas: SVG/UI, matemáticas, traducción, SQL, testing/QA, planificación) para
el enrutamiento por especialista; consulta A18. El resultado se guarda como una tarjeta de modelo. Las
importaciones alojadas nunca se calibran: usan automáticamente los valores por defecto de la tabla del proveedor.

Opciones útiles:

- `--match 'qwen*'` considera solo los modelos cuyo id coincide con el patrón.
- `--loaded` considera solo los modelos que el servidor tiene cargados.
- `--calibrate none|quick|full` elige cuánta medición hacer.
- `-y` selecciona cada modelo coincidente sin preguntar.

Para un proveedor alojado, indica la URL y la variable de entorno que contiene la
clave. `--match` es obligatorio, porque los catálogos alojados pueden contener cientos de modelos:

```bash
pedacito models scan --url https://api.deepseek.com/v1 \
  --api-key-env DEEPSEEK_API_KEY --match 'deepseek-chat'
```

Para medir un modelo de nuevo más tarde, ejecuta `pedacito calibrate`. Para medir cada perfil
local de una vez, ejecuta `pedacito calibrate --all`: pregunta primero (`-y` omite la
pregunta), carga cada modelo por turno, lo descarga antes del siguiente y omite
los perfiles con una tarjeta vigente a menos que agregues `--recalibrate`. Los perfiles alojados se
rechazan: la calibración es solo para modelos locales. Para ver los resultados lado
a lado, ejecuta `pedacito profiles`, o `pedacito profiles --recommend` para obtener
roles sugeridos.

Después de cada tarea la CLI imprime en stderr una cinta de telemetría de una línea: barra de
contexto, tokens del turno, velocidad, aciertos de caché. Define `show_telemetry = false` para ocultarla;
`--jsonl` y los modos rpc nunca la imprimen.

Dos cosas más ayudan cuando hay un modelo alojado en juego. `--max-cost USD` en un
comando de tarea se detiene antes de la siguiente llamada al modelo una vez que el gasto estimado alcanza
ese tope (requiere `price_*_per_mtok` definido para el modelo). En el pie de la GUI,
**KV Cache Hit %** muestra cuánto del prompt reutilizó tu servidor local, que
es el número a vigilar si una sesión larga empieza a sentirse lenta.

### B2b. Genera una imagen (`pedacito image`)

```bash
pedacito image "un faro al atardecer, acuarela"
pedacito image "icono de app, plano" --size 512x512 --steps 4 --seed 7 --out assets/icono.png
```

Necesita un perfil con `modality = "image"` (se detecta solo en servidores Lemonade `sd-cpp`, Stable Diffusion y Flux;
consulta la [guía de modelos](MODEL_GUIDE.es.md#modelos-de-imagen-modality--image)). El archivo se guarda en `.pedacito/images/` salvo que
pases `--out`. Dentro de `pedacito tui`, `/image PROMPT` hace lo mismo.

### B3. Mira qué se indexaría

```bash
pedacito files ~/code/anki
```

```
  created ~/code/anki/.pedacitoignore
  created ~/code/anki/.gitignore with .pedacito/
  ref   data/intervals.csv
  code  src/scheduler.py
  skip  deprecated/   [ignored directory, 1 file(s)]
```

Esto no hace ninguna llamada al modelo. Si algo no debe indexarse (scripts antiguos,
código de terceros, salida de compilación), agrégalo a `.pedacitoignore` con sintaxis de gitignore,
usando barras diagonales en todos los sistemas operativos:

```
deprecated/
*_old.py
```

Ejecuta `pedacito files` de nuevo para confirmar. Dejar fuera lo que no necesitas le da al
modelo más espacio para el código que te importa.

### B4. Revisa el costo y luego indexa

```bash
pedacito cost ~/code/anki        # free: estimated model calls and time
pedacito index ~/code/anki
```

`cost` te dice si el primer índice tarda dos minutos o veinte. Ocurre una
sola vez. Después, solo se vuelve a resumir el código que cambió. No tienes que ejecutar
`index` tú mismo, porque `ask` y `chat` lo actualizan.

Para ver el mapa que recibiría una tarea, sin llamada al modelo:

```bash
pedacito map ~/code/anki -t "make the leech threshold configurable"
```

En un proyecto grande, los archivos con más probabilidad de importar van primero, cada uno con una
línea `why listed:`, sobre un esqueleto del resto.

### B5. Pide un cambio (no se escribe nada)

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable"
```

```
--- looking things up ---
  step 1: read src/scheduler.py L11-21   (see the scheduler)
  step 2: done                           (have enough)
--- answering ---
...
[ok ] src/scheduler.py: 2 block(s) would apply
--- a/src/scheduler.py
+++ b/src/scheduler.py
```

**Sin `--apply` ni `--review`, no se escribe nada.** Obtienes un diff y tú
decides. Mantente en este modo durante tus primeras doce tareas con un modelo o proyecto nuevo,
para que aprendas qué tan confiablemente aterrizan sus ediciones.

Antes de escribir cualquier edición, Pedacito la verifica contra errores comunes de modelos pequeños:
editar código que nunca leyó, dejar un `TODO` en lugar de código funcional o eliminar
una función que aún tiene llamadores. Los errores disparan un reintento corregido. Cada tarea
termina con una línea de uso que muestra llamadas, tokens de entrada y salida, la proporción de aciertos
de caché de prompt del servidor y el costo si definiste precios.

Para trabajo de diseño, agrega `--creative` (modo Builder creativo/de diseño: persona de
diseñador, muestreo más cálido, reescrituras de HTML/CSS de archivo completo; A21):

```bash
pedacito ask ~/code/site -t "redesign the landing page hero" --creative
```

Las tareas de diseño se detectan automáticamente incluso sin la opción (`creative_auto`);
`--creative` fuerza el modo para cada tarea de la ejecución.

### B6. Revisa y luego aplica

Revisa primero. Esto abre una página sin conexión con tu archivo actual a la izquierda y
la versión propuesta a la derecha:

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --review
```

Para escribir, marca la casilla de atestación, haz clic en **Sign off & write** y confirma.
Presionar Enter nunca escribe. Si editas el archivo mientras la revisión está abierta, la
escritura se rechaza en lugar de sobrescribir tu cambio. **Discard** no cambia nada.
Define `review = true` en tu configuración para que la revisión sea lo predeterminado, y usa
`--no-review` para omitirla en una ejecución.

Para escribir de inmediato:

```bash
pedacito ask ~/code/anki -t "make the leech threshold configurable" --apply
```

En ambos casos, los originales se guardan primero en una instantánea.

### B7. Deshacer

```bash
pedacito restore ~/code/anki --list
pedacito restore ~/code/anki --dry-run      # what the newest snapshot would restore
pedacito restore ~/code/anki                # restore it, byte for byte
pedacito restore ~/code/anki --snapshot 20260830-005535
pedacito restore ~/code/anki --delete-created   # also remove files the run created
```

Durante una tarea, el modelo puede deshacer sus propias escrituras con la acción `undo` (esto
requiere `allow_write`, y los respaldos están activados por defecto). Eso revierte las
instantáneas de esta tarea y elimina los archivos que creó.

Para un retroceso más fino, cada Task, paso de Auto y andamiaje de New también registra un
checkpoint de viaje en el tiempo (`.pedacito/checkpoints/`). Inspecciona el árbol, restaura un
solo archivo o todo el espacio de trabajo, y crea una rama de una nueva ejecución desde cualquier nodo:

```bash
pedacito undo ~/code/anki                    # state immediately before the last run
pedacito checkpoints ~/code/anki             # ids, parents, modes, changed files
pedacito checkout cp_0003_a1b2c3 ~/code/anki # restore the whole workspace
pedacito checkout cp_0003 ~/code/anki --file app.py
```

El botón **Checkpoints / Undo** del encabezado de la GUI previsualiza el diff unificado de cada
checkpoint y ofrece **Restore Entire Workspace**, **Restore Selected File**
y **Branch Prompt From Here**. La TUI expone `/checkpoints`, `/undo` y
`/checkout ID`.

La pestaña **Run** de la GUI previsualiza en vivo un `index.html` estático y ejecuta los
puntos de entrada detectados (`python main.py`, `pytest -q`, ...) o un comando personalizado. Un fallo
ofrece **Fix This Crash**, que llena el compositor de Task con el comando, el
código de salida y el traceback. En la TUI usa `/run [cmd]` y luego `/fix`.

### B8. Haz una pregunta (sin ediciones)

```bash
pedacito explain ~/code/anki -q "how does the scheduler pick the next card?"
```

`explain` nunca construye un índice, así que ejecuta `pedacito index` primero.

### B9. Ejecuta un plan autónomo

```bash
pedacito auto . -t "add retry logic to the sync client"          # plan only
pedacito auto . -t "add retry logic to the sync client" --apply  # plan, then implement
```

El plan es una lista de verificación en markdown en `PLAN.md`. Puedes editarlo antes de aplicarlo,
y una ejecución puede reanudarse; el archivo también guarda la meta completa, que una
ejecución reanudada devuelve a cada tarea. Opciones útiles:

- `--verify "pytest -q"` verifica y repara cada tarea.
- `--verify-final` ejecuta esa verificación una vez, después de la última tarea sin marcar, en lugar
  de después de cada una.
- `--diagnostics` ejecuta el linter y el verificador de tipos del proyecto en cada tarea.
- `--worktree NAME` trabaja en un git worktree aislado y deja tu checkout
  intacto.
- `--plan FILE` usa tu propia lista de verificación en lugar de generar una.
- `--max-tasks N`, `--max-minutes N` o `--steps N` definen límites. No hay ninguno por
  defecto.
- `--strict` sale con estado 1 si una tarea aplicó ediciones pero dejó una edición fallida o
  pruebas fallidas, lo que es útil en CI.
- `--refine` expande una meta de una línea en una especificación completa antes de planificar.
- `--force-auto` omite el aviso de idoneidad Task-vs-Auto (abajo); `-y`/`--yes`
  hace lo mismo y acepta otras confirmaciones.

**Las ediciones de una sola pasada reciben un aviso.** Cuando la meta parece trabajo de Task de una pasada
(apunta a 1–2 archivos existentes, usa verbos de edición localizados como `fix`, `restyle`
o `rename`, o ajusta contenido/estilos en un proyecto diminuto), `auto` pregunta
`Run as a single-pass Task instead? [Y/n]` antes de construir un `PLAN.md` de varios pasos.
Enter o `y` ejecuta todo el prompt como una pasada rápida de `ask`; `n` mantiene
el ciclo Auto. Las tuberías y el CI nunca se bloquean con la pregunta: reciben una
nota de una línea y Auto continúa. Los prompts que dicen "multi-step plan",
"create a plan", "from scratch" o nombran ≥4 archivos/módulos nuevos siempre mantienen Auto.
En la TUI aparece la misma opción en `/auto GOAL` (`/auto! GOAL` fuerza el
ciclo); en la GUI un banner "💡 Looks like a single-pass edit" ofrece
**Switch to Task mode (Recommended)**, y al hacer clic en Run sobre ese prompt se pregunta
**Run as a single-pass Task instead?** con los botones **Run as Task (Recommended)** /
**Run Auto Anyway**.

Una ronda de reparación que empeora la verificación se revierte automáticamente, y esa
tarea se marca como fallida. Una ejecución desatendida no puede agravar el daño. Una reparación
tampoco puede "arreglar" una verificación fallida editando un archivo de prueba preexistente: esas
ediciones se rechazan y se redirigen al código bajo prueba (una tarea que pide cambios
en las pruebas está exenta). Un paso que
aún falla después de sus rondas de reparación se revierte al último checkpoint que pasaba
(`auto_green_ratchet`, activado por defecto), de modo que los pasos posteriores nunca se construyan sobre él.

### B10. Sesión interactiva

```bash
pedacito chat ~/code/anki            # add --apply or --review for the whole session
```

El índice se mantiene caliente entre tareas. Escribe `/help` para ver los comandos. Los más
útiles son `/profile NAME` y `/model ID`. Cuando la sesión se alarga,
`/consolidate` (alias `/compact`) la compacta en un punto de entrega y
la siguiente tarea comienza solo desde esa entrega (A20). Ctrl-C cancela la tarea
en ejecución y mantiene la sesión abierta. Cambiar de perfil y ejecutar la misma tarea
de nuevo es la forma más barata de comparar dos modelos.

### B10b. Sesión en pantalla dividida (`pedacito tui`)

```bash
pedacito tui ~/code/anki                  # apply-on, with an undo backup
pedacito tui . --no-apply                 # preview edits only
pedacito tui . --headless < commands.txt  # scripted/CI mode, no PTY
```

`pedacito tui` mantiene el índice caliente y divide la terminal en cuatro zonas:
un encabezado con el proyecto, el perfil/modelo, la insignia de calibración, Auto-Route,
Scout y el estado de git; un panel de transcripción junto a una barra lateral plegable de **Activity &
Plan** (100+ columnas, `/sidebar` la alterna); un pie de telemetría de dos filas;
y el prompt `pdc tui (profile) >`.

El pie está en vivo: el **Context Meter** se pone verde bajo 70%, ámbar de
70–85% y rojo por encima de 85%, junto a `Turn: +in/+out`, los totales de sesión,
`Speed: 44.2 tok/s (avg 41.8 tok/s)`, la caché KV, los pliegues y el
modelo/rol activo. El **Burn Rate** (tasa de consumo de tokens) y la barra lateral siguen cada paso de `read`/`grep`/`mcp_call`,
`model_swap`, recibo de verificación y casilla de `PLAN.md` mientras corre una tarea.

Las teclas funcionan dentro de la TUI en lugar de mostrar códigos de escape: **PgUp/PgDn** desplazan
la transcripción media página, **Shift+Up/Down** desplazan tres líneas,
**Alt+PgUp/PgDn** desplazan la barra lateral de Activity/Plan, y la **rueda del mouse**
desplaza tres líneas. El modo de rueda por defecto (`?1007h` Alternate Scroll) conserva
la selección nativa con clic izquierdo y el copiar/pegar del sistema operativo; **F2** o `/mouse on` activa
la captura SGR opcional de la rueda, y `/mouse off` restaura la selección nativa.
**Up/Down** recuperan el historial de comandos, **Left/Right/Home/End** editan el prompt,
**Tab** completa comandos slash y nombres de perfil, y **Esc/End** saltan al
final en vivo. Una vista desplazada hacia atrás muestra `[ ↑ Scrolled back N lines ... ]` y
permanece fija mientras llegan nuevas líneas de registro.

**Entrada multilínea:** `Shift+Enter`, `Alt+Enter` o `Ctrl+J` añaden un salto de línea, igual
que una `\` final seguida de `Enter`; el pegado entre corchetes (`?2004h`) mantiene un pegado
de varias líneas en el prompt como un solo bloque. `Enter` envía.

**Imágenes:** `/attach [RUTA | list | clear]` o `@image.png` prepara archivos
PNG/JPEG/WebP/GIF para la siguiente tarea. Lo mismo vale en otros sitios: `--image RUTA` en la
CLI (repetible), y en la GUI pegar con `Ctrl+V` / `Cmd+V`, arrastrar y soltar o `📎 Attach`.
Límites: 4 imágenes, 10 MB cada una.

Escribe una tarea como de costumbre, o usa `/help`, `/keys`, `/status`, `/auto GOAL`
(las ediciones de una sola pasada reciben primero una sugerencia de modo Task; `/auto! GOAL` fuerza el
ciclo de varios pasos), `/plan`, `/consolidate`, `/profile`, `/auto-route`, `/think`,
`/creative [on|off|auto]` (modo de diseño: on = cada tarea, auto = detectar tareas de diseño,
off = ediciones quirúrgicas), `/apply`,
`/image PROMPT` (genera una imagen con tu perfil de imagen), `/skills`, `/memory`, `/procs`, `/diff`, `/undo`, `/reset`, `/sidebar`,
`/scroll up|down|top|bottom|N`, `/mouse [on|off]`, `/clear` y `/quit`.
Los avisos de permiso aparecen en la barra del prompt (`y` una vez, `a` para la sesión,
`n` para denegar).

### B11. Benchmarks

La suite de benchmarks de seis proyectos se ejecuta desde la GUI (el botón **▦ Benchmarks** en
la Parte A10) o desde la CLI:

```bash
pedacito bench --list                                  # ids, checks, deliverables
pedacito bench --recent                                # past runs, no model needed
pedacito bench rpn_expression_calc                     # one benchmark
pedacito bench all --json --strict                     # all six, CI-friendly
pedacito bench rpn_expression_calc --dir ~/benchmarks  # choose the output root
```

Cada ejecución construye el proyecto con el ciclo normal de Pedacito y luego califica las
pruebas del propio modelo y la rúbrica oculta de evaluación. El progreso se transmite a stderr; la
tarjeta de puntuación (pruebas aprobadas, verificaciones de rúbrica, tokens, duración y ruta del informe) va a
stdout. `--json` imprime un resultado estructurado para scripts, y `--strict` sale con 1
a menos que pasen tanto las pruebas del modelo como la rúbrica de evaluación. `--list` y `--recent`
no necesitan servidor de modelos ni `vram_gb`.

La CLI también mide un modelo con `pedacito calibrate` y `pedacito eval`,
que se describen en la [guía de modelos](MODEL_GUIDE.es.md#cómo-medir).

### B12. Renombra un símbolo (sin llamada al modelo)

```bash
pedacito rename apply_discount discounted_price                        # preview the diffs
pedacito rename apply_discount discounted_price --file pkg/core.py     # that file and its importers
pedacito rename apply_discount discounted_price --strings              # also whole-word string literals
pedacito rename toggleStep flipStep --include-labels                   # also test-label strings only
pedacito rename apply_discount discounted_price --apply                # write it
pedacito restore ~/code/anki                                           # undo it, if needed
```

La vista previa es lo predeterminado. Con `--apply`, los originales se respaldan primero, de modo que
`pedacito restore` puede deshacer el renombrado. Se renombran identificadores de Python, JavaScript, TypeScript, Go y Rust;
una cadena o comentario que contenga el nombre antiguo se deja intacto
a menos que pases `--strings`, que también renombra las apariciones de palabra completa dentro de
literales de cadena (los comentarios siguen intactos), o `--include-labels`, que mueve solo las
cadenas de etiquetas de prueba (el primer argumento de `test`/`it`/`describe`/`suite`). Cada
archivo modificado debe pasar la verificación de sintaxis, o no se escribe nada.

### B13. Administra skills

```bash
pedacito skills .                          # every skill with its scope badge
pedacito skills show add-a-test            # one skill's triggers + checklist
pedacito skills add deploy-checklist \
  --description "Ship a release safely" \
  --triggers "deploy,release" --body "1. Run the suite. 2. Tag. 3. Publish."
pedacito skills draft "Skill for Alpine.js dark-mode components"   # model writes it
pedacito skills rm deploy-checklist        # delete a custom skill
pedacito skills list --global              # the user-level library instead
```

`add` y `draft` escriben `.pedacito/skills/<name>.md` en el proyecto, o
`~/.config/pedacito/skills/<name>.md` con `--global`. Las skills integradas son
inmutables: `rm` las rechaza, pero una skill personalizada que eclipsa una integrada puede
eliminarse para que la integrada vuelva a verse. El equivalente en la GUI es
**Project ▾ → Skills library & creator…** (A19).

### B14. Consolida el historial de sesiones

```bash
pedacito memory ~/code/anki --consolidate
```

Destila las transcripciones de sesión más recientes en un punto de entrega de menos de 400
tokens (Goal & Context, Files Modified & Key Decisions, Verified Test Status,
Open Next Steps), escribe `.pedacito/consolidated_session.md`, fija las decisiones duraderas
en `memory.md` e imprime los conteos de tokens antes/después. Dentro de
`pedacito chat` la misma operación es `/consolidate` (alias `/compact`); mediante
RPC es el método `consolidate` (alias `compact`). El botón de la GUI es
**Consolidate Session** en el pie de telemetría (A20).

---

## Varios modelos: perfiles

Cada modelo que usas es un perfil. `pedacito models scan` y la GUI los crean
por ti, pero también puedes escribirlos tú mismo:

```toml
[profiles.tiny]
model = "qwen2.5-coder:1.5b"
tier  = "tiny"           # ~4k window budgets for models of 4B and under

[profiles.small]
model = "qwen2.5-coder:7b"
tier  = "small"          # tight token budgets for an 8k window

[profiles.big]
model = "some-27b-model"
tier  = "large"

[profiles.api]
base_url = "https://api.deepseek.com/v1"
api_key  = "sk-..."
model    = "deepseek-chat"      # remote hosts are detected automatically

[profiles.reasoning]
model = "qwen3:32b"
disable_thinking = true         # family-aware: faster, less reasoning
```

```bash
pedacito ask anki -t "..." --profile big
pedacito config --profile big        # show what it resolves to, and why
```

Elige un perfil con `default_profile =` en tu configuración, la variable de entorno
`PEDACITO_PROFILE`, `--profile` o `/profile` en `chat`. Las opciones posteriores
ganan. `tier` normalmente puede quedar sin definir: Pedacito lo deduce del nombre del modelo (4B
o menos es `tiny`, hasta 10B es `small`, hasta 20B es `medium` y todo lo
más grande es `large`).

Los nombres de modelo difieren según el servidor. LM Studio y Lemonade usan el id que
imprime `pedacito config --check`. Ollama usa el nombre de `ollama list`, como
`qwen2.5-coder:14b`. Una discrepancia es el error de configuración más común.

Cambiar de modelo hace que el servidor cargue el nuevo, lo que puede tardar uno o
dos minutos. Pedacito espera (`load_wait_seconds`, por defecto 240).

**Las APIs alojadas conservan su propio muestreo.** Un perfil con un `base_url` no local
(el `[profiles.api]` de arriba) no recibe `temperature`, `top_p`, `top_k`,
`min_p` ni `repeat_penalty` por defecto, incluidas las claves de la tabla
`sampler`, de modo que se aplican los valores por defecto ajustados del proveedor. Para enviar de todos modos la
temperatura configurada, define `send_temperature_to_hosted = true`; para fijar un valor
en los endpoints alojados (p. ej. `1.0` para Claude), define `hosted_temperature = 1.0`
globalmente o en el perfil. En cualquier caso solo se envía `temperature`; las otras
cuatro claves siguen omitidas. Los servidores locales (loopback o LAN) conservan cada valor
configurado: `temperature_answer` es `0.3` por defecto (fría, no voraz), las tareas creativas
se ejecutan a ≥ 0.6, y una respuesta que se repite recibe un suave empujón de
`presence_penalty = 0.15` en el reintento en lugar de una temperatura fría.

Para enviar solo los pasos de búsqueda a un perfil más pequeño, define `scout_profile` con su nombre,
por ejemplo `scout_profile = "scout"` en `[universal]`. El perfil activo sigue
planificando, respondiendo y editando. La [guía de modelos](MODEL_GUIDE.es.md#dual-gpu--dual-model-enrutamiento-scout--arquitecto-scout_profile)
explica cómo combinar los tamaños.

Para enviar *tareas* completas a perfiles especialistas, etiqueta cada perfil con
`specialties = ["coding", ...]` (o deja que `pedacito calibrate` detecte las etiquetas) y
define `auto_route_specialties = true`, o pasa `--auto-route` a `ask`, `auto` o
`new --complex`. Cada tarea se ejecuta entonces en el perfil que mejor coincide, cambiando solo
en los límites de tareas; en un servidor local el modelo anterior se descarga primero para que
el siguiente quepa en la VRAM. `--no-auto-route` mantiene todo en el perfil
activo. Un título de tarea con el prefijo `[specialty: TAG]` fuerza esa especialidad.

---

## Cómo escribir buenas tareas

El andamiaje lleva a un modelo pequeño a las 300 líneas correctas de 12,000. No
hace al modelo más inteligente una vez que está allí.

- **Funciona bien (acotado, local):** "add retry-with-backoff to the sync client",
  "`is_leech` has an off-by-one, fix it", "add type hints to `Scheduler`".
- **Funciona mal (amplio, estructural):** "refactor the scheduler", "modernise this
  codebase", "add tests" (nada específico en qué anclarse).

Nombra el archivo o símbolo cuando lo conozcas. "Fix the off-by-one in `is_leech`" supera a
"fix the leech bug". Las preguntas también funcionan. Si no se necesita ningún cambio de código, recibes una
respuesta y ninguna edición.

Varios cambios en una tarea funcionan mejor si se dividen. Usa `--plan`, o viñetas en
el texto de la tarea. Cada requisito obtiene entonces su propia ventana de contexto nueva, y
Pedacito sigue cada uno hasta tener evidencia de que se hizo. Las tareas que nombran
cinco o más archivos se planifican automáticamente.

Una petición de una línea también puede expandirse primero. En la GUI, haz clic en **✦ Refine
Prompt**. En la línea de comandos, agrega `--refine` a `ask`, `new` o `auto`. La
especificación expandida nombra los archivos reales y las firmas de las funciones, de modo que cada paso tenga
los mismos nombres con los que trabajar.

---

## Cuando algo parece estar mal

| Síntoma | Causa y solución |
|---|---|
| `Cannot reach the server` | El servidor no está en ejecución, el puerto es incorrecto o (remoto) no está escuchando en la red. Consulta "Un servidor de modelos" más arriba. |
| El modelo aparece en la lista pero no carga ni genera | Normalmente es VRAM. Hay otro modelo cargado. Descárgalo o reduce la longitud de contexto. |
| `search block not found` | El modelo copió los prefijos de número de línea, o el archivo cambió debajo de él, o su texto lleva bytes que el archivo no tiene (una comilla tipográfica, mojibake, un tabulador, CRLF). El fallo muestra el texto coincidente más cercano con los números de línea actuales, y en el reintento se ofrece el ancla de rango (`--- FILE: path LINES 40-46`), que ignora por completo el texto citado y se aplica por número. Si sigue fallando, acota la tarea. |
| El modelo dice que una función no existe | No está indexada. Revisa `pedacito files`. Una regla de exclusión puede ser demasiado amplia. |
| `EXIT: 5` / `no tests ran` de `run_test` | No se recolectó nada: no es una prueba fallida, y el código puede estar bien. El objetivo probablemente es un script independiente (`def main()`, `if __name__ == "__main__"`). Ejecútalo con `run` en su lugar, o escribe funciones `def test_...` reales si debía contener pruebas. Renombrar funciones que funcionan para perseguir el código de salida es la corrección equivocada, y la herramienta lo dice. |
| La página de un proyecto de navegador nuevo aparece en blanco al abrirla con doble clic | Los módulos ES se rechazan sobre `file://` (la consola muestra un error de CORS). Sirve la carpeta: `python -m http.server 8000` y luego abre `http://localhost:8000/index.html`. El bloque `## Run` del README generado lo indica. |
| `Model would not produce valid JSON` | Normalmente es una respuesta truncada. Pedacito reintenta con más espacio. Si persiste, aumenta `max_tokens_step`. Un modelo de razonamiento que gasta todo su presupuesto razonando no devuelve JSON, así que desactiva el razonamiento para ese perfil. |
| Cada petición falla, o "kept hitting limits" | La petición es mayor que la ventana de contexto del servidor. Revisa la línea `[auto-fit]` impresa al inicio. Si el servidor informa una ventana desactualizada, define `context_tokens` (y `k_cache_quant` / `v_cache_quant` para una caché cuantizada). |
| La respuesta no tiene sentido en un servidor local | Algunos servidores (Ollama) truncan silenciosamente un prompt demasiado grande. Aumenta la longitud de contexto del servidor. |
| Una API alojada parece ignorar `temperature` | Es a propósito: los endpoints alojados no reciben claves de muestreo por defecto, de modo que se aplican los valores ajustados del proveedor. Define `send_temperature_to_hosted = true` para enviar la temperatura configurada, o fija `hosted_temperature` (p. ej. `1.0` para Claude). Los servidores locales conservan cada valor configurado. |
| Un modelo local se repite textualmente | La fase de respuesta se ejecuta con `temperature_answer = 0.3` (no voraz) para evitar justamente esto, y un reintento por repetición recibe un suave empujón de `presence_penalty = 0.15`. Si persiste, aumenta `temperature_answer` ligeramente o cambia de perfil; una tarea de diseño también puede usar `--creative` (≥ 0.6). |
| El modelo parece perdido o lee las cosas equivocadas | Nombra el archivo en la tarea. Si definiste `--steps` o `max_steps`, auméntalo. |
| Las respuestas son vagas | Puede que el mapa haya pasado de un listado completo a tarjetas por archivo. `pedacito map PROJ --tier auto` muestra cuál. |
| Resúmenes obsoletos o engañosos | Un docstring reemplaza un resumen generado. Define `summarise_documented = true` y vuelve a indexar. |
| No apareció una página de revisión | Nada se ancló, así que no había nada que revisar. Revisa los fallos impresos. |
| `FrontendContractError` tras una edición | La página no puede cargarse tal como está escrita: una ruta de script u hoja de estilos que no existe, un `getElementById` o `querySelector('#id')` para un id que ningún HTML define, JavaScript que no se analiza, un nombre de nivel de módulo enlazado dos veces (un segundo `import { X }` en un archivo que ya lo importa) o una llamada a un nombre que ningún módulo declara ni importa. El mensaje nombra el archivo, la línea y la corrección exacta. `pedacito webcheck PROJ` lista todos esos problemas. |
| La página pasa `webcheck` pero sigue en blanco tras una edición | Las verificaciones estáticas no ven los fallos en tiempo de ejecución (un módulo que lanza una excepción al cargar, un montaje que nunca ocurre). `pedacito domcheck PROJ` carga la página de entrada en Chromium headless e informa errores de página, regresiones de montaje y controles sin enlazar; necesita la instalación opcional de Playwright (`pip install playwright && playwright install chromium`). Para ejecutarlo tras cada edición web aplicada, define `verify_dom = true` y `allow_shell = true`, o pasa `ask --verify-dom`. |
| Un secuenciador/sintetizador pasa todas las verificaciones pero no emite sonido | Web Audio falla en silencio: un contexto que nunca se reanudó dentro de un gesto del usuario, un grafo que nunca llega a `ctx.destination`, osciladores sin `start()`, ganancias fijas en 0, automatización con NaN. `pedacito domcheck PROJ` instrumenta la API de audio, hace clic en Play y recorre los controles deslizantes, e informa estos casos como errores (las páginas sin audio no se ven afectadas). |
| El servidor rechaza la salida estructurada, o los pasos siguen fallando al analizarse | Pedacito baja por sí mismo a un modo estructurado más simple. Ejecuta `pedacito config --doctor` para ver qué modos admite el servidor. Define `enforce_json_schema = false` solo si el servidor se comporta mal con muestreo restringido. |
| Todo va lento | Es probable que el modelo se haya desbordado de la VRAM. |

Para forzar una reconstrucción limpia, ejecuta `pedacito reset PROJ --reindex`, o pasa `--reindex`
a `ask` o `chat`. `reset` borra el estado `.pedacito/` del proyecto sin
tocar tu código ni `.pedacito.toml`. Usa `--dry-run` para previsualizarlo,
`--keep-memory` para conservar los hechos recordados, y `--keep-config` para conservar tus reglas,
skills, prompts e instrucciones. Un simple `--reindex` es la opción más ligera cuando
solo el mapa está desactualizado.

---

## Novedades de la v0.3.3

- **TUI enriquecida, sin paquetes extra.** Las ediciones se muestran como diffs en línea, y las llamadas
  a herramientas son tarjetas de una línea que puedes abrir con `Ctrl+O` (o con `/expand` y
  `/collapse`). Escribe `@` para abrir un selector difuso de archivos e imágenes. Los clics del ratón
  funcionan en `[EN|es]`, en las pestañas de la barra lateral y en la barra de desplazamiento.
- **Git guiado.** `pedacito commit` y `pedacito pr` escanean secretos antes de confirmar o abrir un pull
  request. Ninguno de los dos hace push nunca.
- **Editores y hosts de agentes.** `pedacito acp` es un puente Agent Client Protocol por stdio, construido
  sobre el mismo motor que `rpc`.
- **Servidores Tailscale y LAN.** Una dirección `100.64.x.x` (Tailscale) cuenta como endpoint local. Una
  tarjeta de modelo local sin temperatura calibrada se marca como obsoleta; ejecuta
  `pedacito calibrate --recalibrate` una vez.
- **Windows PowerShell 5.1.** `2>&1` envuelve las líneas de stderr como errores. Para una salida JSON
  limpia usa `2>$null`, o ejecuta el comando en PowerShell 7 o `cmd.exe`.

## Comandos de un vistazo

| Comando | Qué hace |
|---|---|
| `pedacito gui [PROJECT]` | Panel de control en el navegador: Task/Edit, Ask, Auto, New, How ?, Benchmarks, Settings |
| `pedacito config --setup` | Configuración inicial: encuentra servidores y escribe perfiles |
| `pedacito config --check` | ¿El servidor está activo y el modelo funciona? |
| `pedacito config --doctor` | También sondea la salida estructurada, la ventana de contexto, el tokenizador y el sandbox |
| `pedacito config --upgrade [--dry-run]` | Agrega a tu configuración las opciones recién documentadas, comentadas; `--dry-run` solo las lista |
| `pedacito models [--profile NAME] [--model ID]` | Muestra los números ajustados del perfil activo (o de uno indicado); `--json` los imprime como JSON |
| `pedacito models scan` | Encuentra servidores locales o una API alojada, elige modelos, escribe perfiles y los mide |
| `pedacito calibrate` | Mide el modelo y guarda una tarjeta de modelo que ajusta los valores por defecto; `--all` recorre cada perfil local (los modelos alojados se rechazan) |
| `pedacito profiles [--recommend]` | Lista los perfiles y sus puntuaciones; sugiere roles |
| `pedacito how "..."` | Pregunta cómo usar Pedacito. No se necesita proyecto |
| `pedacito image PROMPT` | Genera una imagen con un modelo de imagen configurado (`--out`, `--size`, `--steps`, `--seed`) |
| `pedacito files PROJ` | Qué se indexaría, y por qué. Gratis |
| `pedacito cost PROJ` | Cuánto costará indexar. Gratis |
| `pedacito index PROJ` | Construye o actualiza el índice |
| `pedacito map PROJ [-t "task"]` | Imprime el índice, ordenado para una tarea. No se necesita modelo |
| `pedacito ask PROJ -t "..."` | Una tarea. Simulación por defecto. Agrega `--review`, `--apply`, `--refine` o `--creative` (modo de diseño) |
| `pedacito explain PROJ -q "..."` | Responde una pregunta sobre el código. Sin ediciones |
| `pedacito chat PROJ` | Sesión interactiva; el índice se mantiene caliente |
| `pedacito tui PROJ` | TUI en pantalla dividida: transcripción + barra lateral de actividad/plan + Context Meter, consumo de tokens y telemetría de tok/s. `/help` lista los comandos slash; `--headless` es programable |
| `pedacito rpc [PROJ]` | JSONL-RPC por stdin/stdout para editores y agentes; el índice se mantiene caliente |
| `pedacito mcp list` | Muestra las herramientas de los servidores MCP con nombre y sus indicadores de confianza |
| `pedacito new DIR -t "..."` | Inicia un proyecto desde cero (o `--spec FILE`); `--no-complex` (alias `--single`) desactiva la escalada automática a Complex |
| `pedacito auto PROJ -t "..."` | Planifica una meta y luego la implementa tarea por tarea |
| `pedacito restore PROJ` | Deshace la última escritura |
| `pedacito recover list [PROJ]` | Muestra los diarios de tareas inacabadas por UUID. `recover rollback ID PROJ` previsualiza una reversión vinculada de forma segura; añade `--apply` para realizarla |
| `pedacito rename OLD NEW [--file F] [--strings] [--include-labels] [--apply]` | Renombra un símbolo en todo el proyecto. Sin llamada al modelo; vista previa por defecto; `--strings` también renombra literales de cadena de palabra completa, `--include-labels` solo cadenas de etiquetas de prueba |
| `pedacito eval` | Puntúa el arnés sobre repositorios de prueba. Usa `--baseline` como compuerta |
| `pedacito bench [BENCHMARK\|all]` | Ejecuta los seis benchmarks de diagnóstico y la rúbrica de evaluación. `--list`/`--recent` son gratis; `--json`/`--strict` para CI |
| `pedacito memory PROJ` | Lista, agrega u olvida hechos recordados; `--consolidate` compacta el historial de sesiones en una entrega |
| `pedacito skills [list\|show\|add\|draft\|rm] [PROJ]` | Explora y administra la biblioteca de skills (19 integradas + personalizadas); `--global` apunta a la biblioteca del usuario |
| `pedacito tests PROJ --files F` | Comando de pruebas detectado, y las pruebas que una edición en F puede romper |
| `pedacito webcheck PROJ` | Verificaciones estáticas de archivos web: ids del DOM, carga de scripts, scripts faltantes, enlaces de módulos ES y vínculos |
| `pedacito domcheck PROJ` | Verificación de humo del DOM headless: carga la página de entrada en Chromium headless, comprueba que el grafo de módulos se monta, informa errores de consola, controles sin enlazar y fallos silenciosos de Web Audio (requiere Playwright) |
| `pedacito diagnose` | Ejecuta el linter o verificador de tipos del proyecto |
| `pedacito diag [PROJ]` | Alias corto de `diagnose` |
| `pedacito parsers prefetch [--languages ...]` | Precarga las gramáticas de tree-sitter para que la herramienta funcione sin conexión o aislada; informa ok o failed por lenguaje |
| `pedacito acp [PROJ]` | Protocolo Agent Client Protocol (JSON-RPC 2.0) por stdin/stdout para editores y hosts de agentes; texto, recursos `@file`, imágenes, actualizaciones en streaming y avisos de permiso |
| `pedacito daemon [start\|stop\|status\|sessions]` | Daemon de sesiones headless solo en loopback con un índice compartido caliente; `status` funciona sin conexión y sale con 0 cuando no hay daemon |
| `pedacito commit [--all] [-m MSG] [-y]` | Escanea el diff preparado en busca de secretos y luego crea un commit local. Nunca hace push; `--no-model` redacta sin conexión y `--allow-secrets` es la excepción explícita |
| `pedacito pr [--base main] [--draft] [--dry-run] [-y]` | Redacta título y cuerpo del PR a partir del diff y los commits de la rama, y luego lo abre con `gh`. Nunca hace push de la rama |
| `pedacito prune PROJ` | Elimina sesiones, respaldos y eventos antiguos |
| `pedacito reset PROJ [--reindex]` | Borra el estado de ejecución de `.pedacito/`. El código y la configuración permanecen |
| `pedacito trust PROJ` | Aprueba el `.pedacito.toml` de un proyecto para que se apliquen sus ajustes de ejecución y sus interruptores de capacidades |
| `pedacito worktree create\|list\|remove NAME` | Administra git worktrees aislados |
| `pedacito serve --root PROJ --name web -- npm run dev` | Inicia un servidor en segundo plano con nombre para el proyecto |
| `pedacito procs --root PROJ [logs\|stop NAME] [--all]` | Lista, sigue o detiene los servidores en segundo plano del proyecto. La lista oculta las entradas detenidas reemplazadas; `--all` las muestra |
| `pedacito lsp --root PROJ --file F --line N --op definition` | Pregunta al servidor de lenguaje del proyecto por definiciones, referencias o información al pasar el cursor |

Opciones disponibles en la mayoría de los comandos: `--exclude 'pattern'`, `--include 'pattern'`,
`--ext .kt`, `--steps N`, `--model ID`, `--url URL`, `--profile NAME`,
`--config FILE`, `--root PATH`, `--creative` (modo Builder creativo/de diseño).

---

## Dónde vive cada cosa

Por proyecto, bajo `<project>/.pedacito/`:

- `index.json`: el mapa.
- `summaries.json`: la caché de resúmenes.
- `backups/<stamp>/`: instantáneas previas a la escritura.
- `sessions/<stamp>.md`: qué se pidió, qué se buscó y qué se cambió. Abre esto cuando
  una edición resulte incorrecta más tarde.
- `events/*.jsonl`: una línea JSON por paso, llamada, pliegue y tarea, con tokens,
  aciertos de caché y latencia.
- `memory.md`, `reviews/` (páginas de revisión guardadas) y `log.txt` (traza de eventos).
- `consolidated_session.md`: la entrega de consolidación de sesión más reciente (A20).
- `skills/`: skills personalizadas, como `NAME/SKILL.md` o `NAME.md` (A19, B13).
- `micro_facts.json`: comandos confirmados como funcionales en este repositorio, y lecciones de
  reparación bajo `anti_patterns`. Una lección se escribe solo después de verificar una corrección.
  Una lección confirmada dos veces cuyos archivos comparten un directorio se promueve a una
  regla con alcance de ruta administrada automáticamente (`rules/auto_*.md`) y sale de `anti_patterns`;
  esos archivos se limpian solos cuando se eliminan los archivos que nombran. Las lecciones
  llevan contadores `runs` / `recurred` / `avoided`; una que nunca evita su
  tipo de fallo a lo largo de cinco ejecuciones observadas pasa a `retired_lessons`.
  Es seguro leer, editar o eliminar.

Eliminar la carpeta siempre es seguro. Solo pagas una reindexación. `pedacito reset`
la limpia por ti, y `pedacito prune` recorta los archivos antiguos.

Tus ajustes viven en `~/.config/pedacito/config.toml` (`%APPDATA%\pedacito\` en
Windows; un `XDG_CONFIG_HOME` definido reemplaza a `~/.config`). `pedacito config` muestra
cada valor resuelto y qué archivo o perfil lo definió. Los ajustes se resuelven en
capas: valores por defecto integrados, luego ese archivo, luego el `.pedacito.toml` del proyecto,
luego `--config FILE`, luego `--profile` y por último las opciones de línea de comandos. Junto al
archivo de configuración, la GUI recuerda los proyectos recientes en
`gui_recent.json`. `server_modes.json` almacena lo que admite cada servidor y modelo
(modos JSON y tasa del tokenizador, actualizados por `--doctor`). `model_cards.json` contiene
los resultados de `pedacito calibrate`. Todos estos se pueden eliminar sin riesgo.
