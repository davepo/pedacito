🌐 [English](README.md) | **Español**

# Pedacito

[![PyPI](https://img.shields.io/badge/PyPI-v0.3.3-blue)](https://pypi.org/project/pedacito/)
![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)
[![MIT License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
![Local-First](https://img.shields.io/badge/Local--First-yes-brightgreen)
![EN / ES](https://img.shields.io/badge/Docs-EN%20%2F%20ES-orange)

**Pedacito** (`pdc`) es un agente de programación con IA, local primero, pensado para los modelos que puedes
ejecutar en la GPU que ya tienes, de 4B a 27B parámetros. Corta tu código en piezas pequeñas y
con límites precisos, de modo que un modelo pequeño lee y edita solo las líneas que importan, en lugar de
necesitar un modelo de frontera con una ventana de 200k tokens.

- **Servidores locales e híbridos:** LM Studio, Ollama, Lemonade Server, llama.cpp, vLLM, Strata/Avalon
  y cualquier endpoint compatible con OpenAI.
- **Modelos alojados cuando quieras:** Claude, ChatGPT, DeepSeek, Qwen, Gemini y OpenRouter para planificar,
  verificar o escalar, con un tope de gasto.
- **Seguro por defecto:** escaneo de secretos, sandbox del sistema operativo y checkpoints a los que puedes volver.

![Espacio de trabajo de la GUI de Pedacito: chat con un diff preparado, paneles de actividad y plan en vivo, el compositor con Auto-route, Post-edit verify y Creative, y la barra de telemetría](docs/assets/gui-workspace.png)

![TUI de Pedacito en pantalla dividida: transcripción junto a la barra lateral Activity & Plan, con el medidor de contexto y el selector de idioma EN|ES en el encabezado](docs/assets/tui-split.png)

## Inicio rápido en 30 segundos

```bash
pip install pedacito                 # Windows: py -m pip install pedacito
pdc gui ~/code/your-project          # browser control panel
pdc tui ~/code/your-project          # split-screen terminal
pdc ask "where is login handled?"    # one-shot CLI, from the project folder
```

Primero inicia un servidor de modelos (LM Studio, Ollama, Lemonade o llama.cpp). Luego, en la GUI abre
**Settings → Model discovery → Scan local servers**, o ejecuta `pdc models scan`.

> [!TIP]
> ¿Prefieres español? Usa el selector `EN | ES` en la barra superior de la GUI, presiona `F3` en la TUI o ejecuta
> `pdc --lang es gui`. Consulta [Idioma y localización](docs/es/CONFIG.md#idioma-y-localización-language-llm_reply_language).

## Novedades de la v0.3.3

| Función | Qué obtienes | Más información |
|---|---|---|
| **Puente Agent Client Protocol** | `pedacito acp` habla JSON-RPC 2.0 por stdio para editores y hosts de agentes: `session/new`, `session/prompt` con bloques de texto, `@file` e imagen, eventos `session/update` en streaming y `session/request_permission`. Envuelve el mismo motor caliente que `rpc`, que gana `--events` y `--describe`. | [Puente ACP](docs/es/USAGE.md#acp-puente-de-agent-client-protocol) |
| **TUI enriquecida sin dependencias** | Diffs en línea con resaltado de sintaxis, tarjetas plegables de llamadas a herramientas (`Ctrl+O`, `/expand`, `/collapse`), un selector difuso `@` para archivos, símbolos e imágenes, y zonas de ratón SGR clicables para el cambio de idioma, las pestañas de la barra lateral y la barra de desplazamiento. | [`tui`](docs/es/USAGE.md#tui-sesión-de-terminal-en-pantalla-dividida) |
| **Endpoints locales, LAN y Tailscale** | Una sola regla de localidad cubre loopback, `.local`, RFC 1918, Tailscale/CGNAT (`100.64.0.0/10`), link-local y ULA de IPv6. Las tarjetas locales sin temperatura calibrada se marcan como obsoletas para recalibrar. | [Endpoints locales](docs/es/CONFIG.md#endpoints-locales-lan-y-tailscale) |
| **Mejoras para modelos pequeños y MoE disperso** | `tool_mode = "plain"` (pasos de búsqueda en palabras simples, comparados A/B por `calibrate`), un enrutador `proactive_escalation` opcional, `best_of` probado en sombra y un verificador previo al cierre en modo intercalado. La salida grande de herramientas se vuelca a `.pedacito/scratch/`, que el modelo puede leer pero nunca escribir. | [Configuración](docs/es/CONFIG.md#referencia-de-configuración) |
| **Git guiado, reforzado** | `pedacito commit` y `pedacito pr` ejecutan un escaneo previo de secretos antes de cada commit y PR, nunca hacen push ni force-push, y los comparten `/git` en la TUI y el cajón Git de la GUI. `undo` se niega a rebobinar más allá de un HEAD de git movido sin `--force`. | [Git y pull requests](docs/es/USAGE.md#git-commit-y-pull-requests) |
| **Ejecuciones fiables y pulido de la CLI** | Checkpoints previos a la ejecución para árboles divergentes, `restore --checkpoint`, respaldo offline de tree-sitter con `parsers prefetch`, `pedacito diag`, `models --profile/--model`, `new --no-complex`, semántica honesta de lotes parciales y un límite duro `--max-minutes`. | [Analizadores sin conexión y diag](docs/es/USAGE.md#analizadores-sin-conexión-parsers-prefetch-y-diag) |
| **GUI reforzada** | Aprobación de revisión por bloque parcial con procedencia por bloque, tarjetas de hilo persistentes, traspasos deterministas con `/compact`, un diseño adaptable a pantallas estrechas y desconexiones de cliente silenciosas. | [Página de revisión](docs/es/USAGE.md#página-de-revisión) |
| **Nota de Windows PowerShell 5.1** | En PowerShell 5.1, `2>&1` envuelve las líneas de stderr como registros de error; usa `2>$null` para una salida JSON limpia. | [stderr en Windows](docs/es/USAGE.md#windows-powershell-51-capturar-stderr-21) |

## Novedades de la v0.3.2

| Función | Qué obtienes | Más información |
|---|---|---|
| **Ejecución intercalada** | `execution_mode` elige `two_phase`, `interleaved` o `auto`. Las ejecuciones intercaladas mantienen una sola conversación agéntica con herramientas nativas (OpenAI `tool_calls`, Anthropic `tool_use`, respaldo JSON y XML `<tool_call>` de Qwen3.5/3.8 / Strata), y las ejecuciones en seco se preparan en un espacio de trabajo en sombra. Defínelo por ejecución con `--execution-mode`, o en vivo con `/mode` en la TUI. | [Configuración](docs/es/CONFIG.md) |
| **Gemini y OpenAI Responses nativos** | `gemini_native` (activado por defecto) usa `generateContent` de Google; `openai_responses` usa `/v1/responses` para los modelos de razonamiento o3, o4 y GPT-5. | [Configuración](docs/es/CONFIG.md) |
| **Catálogo de tres niveles y llavero** | Los metadatos del proveedor, una caché de 24 h de `models.dev` y una tabla integrada proporcionan ventanas de contexto y precios. `pedacito models refresh` fuerza una actualización. `api_key_keyring` lee tu llavero del sistema cuando está instalado `pedacito[keyring]`. | [Configuración](docs/es/CONFIG.md) |
| **Onboarding y coexistencia Strata / Avalon** | Un preset de un solo paso en el puerto 8080, un campo de clave API en el escaneo, respaldo con `STRATA_API_KEY`, insignia **API key required** y auto-evict síncrono Lemonade ↔ Strata en GUI, TUI y CLI. | [Configuración del servidor de modelos](docs/es/INSTALLATION.md#configuración-del-servidor-de-modelos) |
| **Insignia RAM/VRAM en vivo y panel de emparejamiento** | La insignia de hardware de la GUI se actualiza cada 3 segundos y se envía en vivo. La VRAM local se detecta automáticamente en reinicios y escaneos locales, manteniendo los perfiles LAN independientes del cliente. El panel Dual-GPU / Scout + Architect Pairing empieza plegado. | [Espacio de trabajo de la GUI](docs/es/USAGE.md) |
| **Revisión por bloque y reglas de shell por proyecto** | Acepta bloques individuales en el panel Changes. **Permitir en este proyecto** guarda una regla de shell estrecha (`shell_allowlist_by_project`) solo para el proyecto actual. | [Página de revisión](docs/es/USAGE.md#página-de-revisión) |
| **Calibración GGUF, temperaturas y oscilación creativa** | La calibración infiere familias base de modelos ajustados desde encabezados GGUF/runtime (o caché de HF Hub en LAN), añade selector Rápida/Completa en la GUI, muestra las temperaturas calibradas en Ajustes/`config --check` y añade `creative_temperature_delta`. | [Configuración](docs/es/CONFIG.md) |

## Novedades de la v0.3.1

| Función | Qué obtienes | Más información |
|---|---|---|
| **GUI, TUI y documentación bilingües EN / ES** | Cambia la GUI con el selector EN/ES de la barra superior, la TUI con `F3` o `/lang`, y la CLI con `--lang es` o `PEDACITO_LANG`. `llm_reply_language` controla el idioma de las explicaciones del modelo. Cada guía tiene una edición en español. | [Idioma y localización](docs/es/CONFIG.md#idioma-y-localización-language-llm_reply_language) |
| **Perfilado avanzado del sistema y auto-ajuste** | `pedacito profile` muestra una instantánea de RAM/VRAM independiente del fabricante, con una fuente para cada número (sysfs de Linux, `nvidia-smi` de NVIDIA, `vm_stat` de macOS, `winapi` de Windows), una sonda de prefill/generación en caché y el ajuste del perfil activo. `auto_tune = true` (opcional) adapta el contexto, los presupuestos de recuperación y el tiempo límite a la memoria libre; si falta telemetría, se abstiene, y las configuraciones existentes no cambian. | [Perfilado del sistema y auto-ajuste](docs/es/CONFIG.md#perfilado-avanzado-del-sistema-y-auto-ajuste) |
| **Auto-actualización consciente del contexto** | Una verificación de versiones discreta y desactivable: una sola petición GET anónima a PyPI al iniciar (tiempo límite de 2 s, en caché por 24 h), un banner descartable en la GUI con actualización en streaming y Restart, `pdc update --run` y el `/update` de la TUI. El comando coincide con tu método de instalación (pipx, venv, pip --user o git para copias editables). `check_for_updates = false` mantiene Pedacito totalmente sin conexión. | [Actualización](docs/es/INSTALLATION.md#actualización) |
| **Generación local de imágenes** | `pedacito image` y `/image` en la GUI/TUI generan imágenes en un Lemonade Server local, con calibración para modelos de imagen. | [Generación local de imágenes](docs/es/USAGE.md#generación-local-de-imágenes) |
| **Entrada multimodal de imágenes** | Envía PNG/JPEG/WebP/GIF a un modelo con visión: `--image RUTA`, pegar/arrastrar/`📎 Attach` en la GUI, `/attach` en la TUI (4 imágenes, 10 MB cada una). | [Entrada multimodal](docs/es/USAGE.md#entrada-multimodal-en-el-chat) |
| **Caché de prompts nativa de Anthropic** | `anthropic_native = true` (por defecto) usa `/v1/messages` con marcadores `cache_control` explícitos, TTL `5m`/`1h`, razonamiento, herramientas e imágenes nativos. | [Caché de prompts alojada](docs/es/CONFIG.md#caché-de-prompts-alojada) |
| **Compositor multilínea en la TUI** | `Shift+Enter`, `Alt+Enter`, `Ctrl+J` o una `\` final añaden saltos de línea; el pegado entre corchetes conserva los bloques pegados. | [`tui`](docs/es/USAGE.md#tui-sesión-de-terminal-en-pantalla-dividida) |


## Novedades de la v0.3.0

| Función | Qué obtienes | Más información |
|---|---|---|
| **Administrador de VRAM entre servidores** | Trata todos tus servidores de modelos locales como un solo conjunto. Muestra qué modelos están residentes, los carga y descarga mediante la API de cada runtime, y expulsa a los demás servidores antes de una ejecución. | [Administrador del ciclo de vida de servidores](docs/es/ARCHITECTURE.md#administrador-del-ciclo-de-vida-de-servidores) |
| **Strata local-moe** | Los servidores de mezcla de expertos por niveles (Strata, Avalon) tienen su propio perfil de ejecución: presupuestos de respuesta mayores, un tiempo límite de 10 minutos y un solo stream a la vez. | [Nivel local-moe](docs/es/ARCHITECTURE.md#perfiles-de-ejecución-y-el-nivel-local-moe) |
| **Streaming SSE** | Los tokens llegan en vivo mediante server-sent events, con un panel de razonamiento en vivo, un latido por llamada y un Stop casi instantáneo. | [Telemetría en vivo](docs/es/USAGE.md#telemetría-en-vivo) |
| **Security Guard** | Busca secretos antes de cualquier petición a un servicio alojado (`sk-`, `pypi-`, `ghp_`, claves de AWS y Google, claves privadas PEM), mantiene los archivos sensibles fuera del alcance del modelo, pregunta antes de acciones riesgosas y ejecuta los comandos en un sandbox del sistema operativo. | [Guía de seguridad](docs/es/SECURITY.md) |
| **Temas personalizados** | Esquemas de color de diez tokens compartidos por la GUI y la TUI. Nueve preajustes, tus propios esquemas guardados y un anclaje por proyecto. | [Esquemas de color](docs/es/CONFIG.md#esquemas-de-color) |
| **Anclaje con `@`** | Escribe `@file.py` o `@Class.method` en un prompt para fijar ese código exacto en cada fase de la ejecución. | [Fijar contexto](docs/es/USAGE.md#fijar-contexto-con-) |
| **Árbol de checkpoints** | Cada ejecución genera un checkpoint. Inspecciona cada cambio como un diff unificado y luego vuelve atrás con `pdc undo` o `pdc checkout`. | [Checkpoints y reversión](docs/es/SECURITY.md#pilar-4-registro-de-auditoría-inmutable-y-viaje-en-el-tiempo) |
| **Enrutamiento híbrido** | Un modelo arquitecto potente planifica una tarea y un editor local rápido la implementa. En una sola GPU, el servidor del arquitecto se expulsa entre las dos etapas. | [Enrutamiento híbrido](docs/es/ARCHITECTURE.md#enrutamiento-híbrido-un-arquitecto-planifica-un-editor-implementa) |


## Documentación

| Guía | Qué cubre |
|---|---|
| [Instalación](docs/es/INSTALLATION.md) | pip, pipx, extras, servidores de modelos, servidores de inferencia en GPU, actualización |
| [Uso](docs/es/USAGE.md) | GUI, CLI, TUI, ejecuciones autónomas, anclaje con `@`, cambio de idioma |
| [Configuración](docs/es/CONFIG.md) | Archivos de configuración, reglas, temas, perfiles, idioma, cada ajuste |
| [Arquitectura](docs/es/ARCHITECTURE.md) | Motor, administrador del ciclo de vida de servidores, `local-moe`, enrutamiento híbrido, límites |
| [Skills y herramientas](docs/es/SKILLS_AND_TOOLS.md) | 19 skills incluidas, skills personalizadas, servidores MCP, hooks, sobrescritura de prompts |
| [Seguridad](docs/es/SECURITY.md) | Escaneo de credenciales, sandbox, checkpoints, reportes |
| [Contribuir](docs/es/CONTRIBUTING.md) | Entorno de desarrollo, pruebas, mapa de módulos, reglas de docs y locales |
| [QUICKSTART.es.md](QUICKSTART.es.md) | Primeras ejecuciones paso a paso con la salida esperada |
| [MODEL_GUIDE.es.md](MODEL_GUIDE.es.md) | Cómo elegir, combinar y ajustar modelos |
| [TECHNIQUES.es.md](TECHNIQUES.es.md) | Cómo Pedacito logra que los modelos pequeños funcionen bien |

**Propiedad de la documentación.** `docs/` (INSTALLATION, ARCHITECTURE, USAGE, CONFIG, SKILLS_AND_TOOLS, SECURITY y CONTRIBUTING) es la referencia canónica de ajustes, comandos y garantías. Las tres guías de la raíz, [QUICKSTART.es.md](QUICKSTART.es.md), [MODEL_GUIDE.es.md](MODEL_GUIDE.es.md) y [TECHNIQUES.es.md](TECHNIQUES.es.md), siguen siendo las guías canónicas orientadas a tareas: primeras ejecuciones, combinación y ajuste de modelos, y técnicas para modelos pequeños. Este README es el índice, y [CHANGELOG.md](CHANGELOG.md) es el historial. La nueva prosa de referencia va en `docs/`, los nuevos tutoriales van en las guías de la raíz, y cada conjunto enlaza al otro en lugar de copiar texto.

Ejecuta `pdc --help` para ver todos los comandos, o `pdc how "tu pregunta"` para la ayuda integrada.

## Contribuir

Los issues y pull requests son bienvenidos; por favor abre un issue antes de un PR grande. Consulta
[docs/es/CONTRIBUTING.md](docs/es/CONTRIBUTING.md). El sitio web del proyecto es
[pedacito.dev](https://www.pedacito.dev), y puedes contactar al mantenedor en
contact@pedacito.dev. Reporta los problemas de seguridad de forma privada, como se describe en
[docs/es/SECURITY.md](docs/es/SECURITY.md#reportar-una-vulnerabilidad).

## Licencia

MIT, consulta [LICENSE](LICENSE).
