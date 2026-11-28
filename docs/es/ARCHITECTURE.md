🌐 [English](../ARCHITECTURE.md) | **Español**

# Arquitectura

Este documento describe cómo está construido Pedacito: el motor local primero, el administrador del ciclo de vida
de servidores que trata tus servidores de modelos como un solo conjunto de memoria, los perfiles de ejecución (incluido el
nivel `local-moe` para servidores MoE por niveles) y el enrutamiento multimodelo.

Para el uso diario, consulta [USAGE.md](USAGE.md). Para los límites de seguridad, consulta
[SECURITY.md](SECURITY.md). Para los ajustes, consulta [CONFIG.md](CONFIG.md). Para skills, servidores MCP y hooks, consulta
[SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md); para contribuir, consulta [CONTRIBUTING.md](CONTRIBUTING.md).

**Propiedad de la documentación.** `docs/` (INSTALLATION, ARCHITECTURE, USAGE, CONFIG, SKILLS_AND_TOOLS, SECURITY y CONTRIBUTING) es la referencia canónica de ajustes, comandos y garantías. Las tres guías de la raíz, [QUICKSTART.es.md](../../QUICKSTART.es.md), [MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md) y [TECHNIQUES.es.md](../../TECHNIQUES.es.md), siguen siendo las guías canónicas orientadas a tareas: primeras ejecuciones, combinación y ajuste de modelos, y técnicas para modelos pequeños. Este README es el índice, y [CHANGELOG.md](../../CHANGELOG.md) es el historial. La nueva prosa de referencia va en `docs/`, los nuevos tutoriales van en las guías de la raíz, y cada conjunto enlaza al otro en lugar de copiar texto.

## Visión general del motor

La CLI, la TUI, la GUI en el navegador y la interfaz JSONL-RPC manejan todas el mismo motor. Una
tarea pasa por estas etapas:

1. **Indexar.** El proyecto se analiza en archivos, símbolos y rangos de línea exactos. Python usa
   `ast`. Los demás lenguajes usan tree-sitter. Los resúmenes se guardan en caché por hash de contenido, de modo que
   volver a indexar cuesta solo lo que cambió. Consulta [Qué se indexa](#qué-se-indexa).
2. **Mapear.** Un resumen del repositorio (stack, puntos de entrada, comandos de prueba y compilación) y un mapa de
   el código se sitúan sobre la conversación. Los proyectos pequeños reciben un listado plano. Los proyectos grandes reciben
   un mapa ordenado de los archivos más relevantes para la tarea.
3. **Recolectar.** El modelo trabajador da pasos pequeños y tipados (`read`, `grep`, `find`, `callers`,
   `context` y otros) hasta que llama a `done`. Donde el servidor lo admite, cada paso está
   restringido por un esquema JSON, de modo que no se puede muestrear un paso mal formado. Consulta
   [Qué puede hacer el modelo mientras recolecta](#qué-puede-hacer-el-modelo-mientras-recolecta).
4. **Responder y editar.** El modelo escribe bloques de buscar y reemplazar, o reemplazos a
   nivel de símbolo. Los bloques se aplican en seco y se verifican antes de escribir nada. Consulta
   [Edición](#edición-buscarreemplazar-no-diffs).
5. **Compuerta.** Tras las escrituras, Pedacito ejecuta primero las verificaciones baratas: sintaxis, análisis
   estático de nombres no definidos, contratos de frontend y luego las pruebas afectadas. Las rondas de reparación solo devuelven
   los hallazgos que son nuevos para esta tarea.
6. **Registrar.** Se escriben un checkpoint, un registro de sesión y eventos de telemetría. Consulta
   [SECURITY.md, pilar 4](SECURITY.md#pilar-4-registro-de-auditoría-inmutable-y-viaje-en-el-tiempo).

El orden es deliberado. Un servidor guarda en caché el prefijo del prompt, así que el motor mantiene los primeros
turnos idénticos byte a byte y pliega las búsquedas antiguas solo cuando la ventana está realmente llena. Cada
compuerta determinista se ejecuta antes de gastar una llamada al modelo.

## ¿Por qué "Pedacito"?

Una base de código grande es demasiado para entregársela de una vez a un modelo pequeño, así que Pedacito se la
entrega en pedazos:

- **Fragmentos, no archivos.** Python se analiza con `ast`. JavaScript, TypeScript,
  Go, Rust, Java, C, C++, C#, Ruby, PHP, Kotlin, Swift, Scala y Lua se analizan con tree-sitter. Cada función
  o clase es una unidad con un rango de línea exacto.
- **Esqueletos, no volcados.** Una clase de 500 líneas vuelve como sus firmas y
  punteros de línea, de modo que el modelo lee el único método que piensa cambiar.
- **Buscar antes de leer.** Los mapas ordenados, la búsqueda léxica e híbrida, un grafo de importaciones
  y llamadas, y los vínculos con pruebas deciden qué piezas importan para esta tarea.
- **Presupuestos en todas partes.** Las lecturas, las observaciones y las respuestas están limitadas por presupuestos
  de tokens ajustados a la ventana de contexto real de tu modelo.

## Para quién es

- Desarrolladores con una GPU de consumo, desde una laptop de 6 GB hasta un escritorio de 24 GB, que
  quieren un agente que corra en su propia máquina.
- Personas que quieren modelos locales por defecto, y un modelo alojado solo para las partes
  difíciles, con un tope de lo que puede gastar.
- Personas que quieren ver qué está haciendo el modelo. Cada edición es un diff de simulación
  hasta que la aplicas, y cada escritura se guarda primero en una instantánea.

Pedacito es **local primero y listo para híbrido**. Corre por completo en tu propio servidor
de modelos por defecto, y puede incorporar un modelo alojado para las partes que lo
necesitan: como planificador, como verificador o como destino de escalamiento, con un tope de gasto.
Cada modo se describe en [¿Por qué Pedacito?](#por-qué-pedacito-cómo-se-diferencia-de-los-arneses-que-priorizan-la-nube).

Fue construido y probado por una sola persona en una sola máquina doméstica con GPU, así que los casos límite en otro
hardware pueden ser ásperos. Los reportes de errores con un caso reproducible son muy bienvenidos.

## ¿Por qué Pedacito? (Cómo se diferencia de los arneses que priorizan la nube)

**Por qué los arneses que priorizan la nube sufren en GPU locales.** Los agentes de programación construidos para
APIs en la nube de 200k tokens envían más de 15k tokens de esquemas de herramientas estáticos en cada petición,
y muchos reescriben los turnos anteriores en cada paso (re-resumiendo, reordenando o editando
el prompt del sistema). Una API en la nube lo absorbe. Un servidor local de una sola ranura no:
`llama.cpp`, Ollama y LM Studio reutilizan su caché KV solo mientras el prefijo del prompt
se mantiene idéntico byte a byte. Si mutas el prefijo, el servidor reevalúa
todo el prompt en cada turno, de modo que la GPU pasa el tiempo releyendo contexto en lugar de
generar.

**Cómo lo resuelve Pedacito:**

1. **Prefijo de caché KV idéntico byte a byte (`fold_middle`).** Los turnos intermedios se pliegan
   de forma determinista en recibos de tramo de `EvidenceStore` fijados por hash. Los primeros dos
   mensajes (`messages[0:2]`) nunca se tocan, así que la caché KV de tu servidor se mantiene
   caliente. El pie de la GUI muestra el resultado como **KV Cache Hit %**.
2. **Compuertas deterministas antes de cualquier llamada al LLM.** Las verificaciones estáticas (`verify`,
   `webcheck`, `domcheck`, `apicheck`), la reparación automática de importaciones faltantes y
   la selección de pruebas afectadas detectan muchos errores en milisegundos, sin gastar
   tokens del modelo.
3. **Un conjunto pequeño de dependencias.** El arnés necesita Python 3.11+ e instala
   `requests` y tree-sitter; no hay requisito de Node.js para el arnés
   en sí. Playwright (verificaciones de navegador) y los servidores LSP son opcionales, y se usan solo
   cuando los tienes instalados.
4. **Cero llamadas a casa.** "Telemetría" en Pedacito significa *tu* Context Meter local,
   Burn Rate y las cifras de velocidad `tok/s`, escritas en `.pedacito/events/` en tu
   proyecto. Pedacito no envía analíticas ni reportes de fallos. La única verificación de versiones
   es una petición GET anónima a los metadatos públicos de PyPI al iniciar (tiempo límite de 2 s,
   en caché 24 h, nunca bloqueante), y `check_for_updates = false`
   desactiva incluso eso: cero llamadas de red. Por lo demás, la red se usa solo
   para los servidores de modelos y las APIs alojadas que configures, y para las herramientas web que
   actives.

### Arquitectura local primero, lista para híbrido

Pedacito te ofrece cuatro modos. Usa el primero solo, o agrega los otros según lo permitan tu
hardware y presupuesto.

1. **100% local.** Ejecuta completamente sin conexión contra `llama.cpp`, Lemonade Server,
   LM Studio, Ollama o vLLM, sin dependencia de la nube.
2. **Enrutamiento automático por especialista (`--auto-route`).** Un modelo orquestador rápido (por
   ejemplo, uno de clase 8B) planifica el trabajo y despacha cada tarea al
   mejor especialista local calibrado: código, matemáticas, SVG/3D, SQL o i18n. Entre
   tareas, el servidor descarga un modelo y carga el siguiente, de modo que la VRAM se mantenga limpia.
   Consulta [Enrutamiento automático por especialista](#enrutamiento-automático-por-especialista).
3. **Scout + arquitecto en dos GPU (`scout_profile`).** Un scout rápido de 4B-9B en una
   segunda GPU o puerto reúne archivos e índices a alta velocidad, mientras un arquitecto de 24B-35B
   hace la planificación y la edición. La caché KV del arquitecto no se ve
   perturbada por las búsquedas.
4. **Escalamiento híbrido y enrutamiento de roles (`planner`, `verifier`, `escalate`).**
   Combina un trabajador local gratuito con un modelo de frontera alojado (DeepSeek, Qwen, OpenAI
   o Anthropic) para la planificación inicial, para la verificación o para el escalamiento
   automático cuando un modelo local pequeño se estanca. Mantén bajo el gasto en la nube con la
   **Consolidación de sesión** (`/consolidate`) y el tope de presupuesto **`--max-cost`**.
   Consulta [Enrutamiento de roles](#roles-un-modelo-distinto-para-cada-trabajo).

## Administrador del ciclo de vida de servidores

Un servidor MoE por niveles mantiene residente la mayor parte de la VRAM de la GPU y un gran conjunto de trabajo de DDR5. Si
cargas un modelo denso en otro servidor local junto a él, los dos servidores compiten por las
mismas páginas de memoria, y uno de ellos se satura. Por eso Pedacito trata todos tus servidores
de modelos locales como **un solo conjunto con controles explícitos del ciclo de vida**. El código está en `pedacito/servers.py`.

| Capacidad | Qué ocurre |
|---|---|
| **Descubrimiento** | `configured_local_servers` lista los endpoints locales de tu configuración. `list_server_states` sondea cada uno con un tiempo límite de 1.5 segundos e informa si está en línea y qué modelos están residentes. |
| **Vista de memoria** | `get_hardware_memory` lee la RAM del sistema de `/proc/meminfo` y la VRAM de los contadores sysfs de DRM para cada tarjeta que los expone (amdgpu de AMD, xe de Intel y otros controladores). Su forma no cambia. `profiler.snapshot()` agrega encima lectores de NVIDIA (`nvidia-smi`), macOS (`sysctl` + `vm_stat`) y Windows (`GlobalMemoryStatusEx`), con una fuente por número. |
| **Cargar y descargar** | Se usa la API nativa de cada runtime: `/v1/load` y `/v1/unload` de Lemonade, `/models/load` y `/models/unload` de llama.cpp, y los endpoints genéricos que usan los servidores MoE por niveles. Ollama se descarga con `keep_alive: 0`. Para un demonio de un solo modelo, `load_cmd` y `unload_cmd` pueden ejecutar tu propio comando en su lugar. |
| **Fijar / desfijar** | Solo Lemonade: `pin_server_model` llama a `POST /internal/pin` con `{"model_name": "...", "pinned": true|false}` para que la expulsión LRU omita un modelo cargado. `/api/v1/internal/pin` es la grafía alternativa; los demás runtimes tratan la fijación como una operación nula. |
| **Auto-expulsión** | Antes de que comience una ejecución en un servidor local, se descargan los modelos residentes en los *otros* servidores locales. Luego se sondea brevemente la memoria, para que los pesos del destino puedan reclamarla. Se controla con `auto_evict_servers` (activado por defecto). La GUI tiene un interruptor. |
| **Mejor esfuerzo** | Un servidor dormido es un estado normal, nunca una excepción. Las APIs alojadas nunca se tocan. |

El administrador se usa en los límites de las tareas, por el enrutamiento por especialista y por el enrutamiento híbrido.
Nunca descarga un modelo a mitad de un paso de recolección, de modo que la caché de prompt del servidor
que está trabajando se mantiene caliente.

### Qué significa la insignia de memoria

La insignia de memoria de la barra de herramientas muestra la RAM del sistema, más la VRAM donde el controlador la expone. Los
números son informativos y son **solo totales**: Pedacito no atribuye VRAM a un
proceso, y las etiquetas de la insignia lo indican. La expulsión funciona sobre el estado del servidor: los modelos residentes en
los otros servidores locales se descargan antes de la ejecución, y luego se sondea brevemente la memoria para que
el destino pueda cargar. Una tarjeta cuya VRAM Pedacito no puede leer igual recibe esa transferencia.

### Telemetría avanzada y auto-ajuste

`pedacito/profiler.py` agrega una instantánea independiente del fabricante sin cambiar
`servers.get_hardware_memory()`:

- **Linux AMD/Intel**: el lector existente de `/proc/meminfo` + `/sys/class/drm`, etiquetado
  `drm-sysfs`. Su glob cubre todos los controladores que exponen `mem_info_vram_total`/`_used`, lo cual
  incluye amdgpu e Intel xe.
- **Linux NVIDIA**: `nvidia-smi --query-gpu=index,memory.total,memory.used
  --format=csv,noheader,nounits`, ejecutado como una lista de argumentos y solo cuando el binario se resuelve,
  etiquetado `nvidia-smi`.
- **macOS**: `sysctl -n hw.memsize` para el total y `vm_stat` para las páginas libres, etiquetado `vm_stat`.
  La memoria unificada de Apple Silicon cuenta como RAM; no hay una reclamación de VRAM separada.
- **Windows**: `kernel32.GlobalMemoryStatusEx` mediante `ctypes` de la biblioteca estándar (la misma API en
  x86-64 y ARM64), etiquetado `winapi`. Las tarjetas NVIDIA siguen usando `nvidia-smi`; AMD/Intel en
  Windows no tienen un lector de VRAM independiente del fabricante.

Cada lector está protegido por una verificación de plataforma o `shutil.which`, limitado a dos segundos,
y se omite en silencio cuando está ausente. Un número que falta lleva una fuente vacía y es
**desconocido**, nunca cero de memoria libre.

La sonda de velocidad opcional ejecuta un completado diminuto más una medición de prefill a partir de los tiempos
de SSE. Es explícita (`pedacito profile --run`, `/profile run` de la TUI) o se programa una vez
por servidor+modelo en un hilo demonio después de una carga explícita exitosa del servidor. La sonda
usa el punto de entrada público de un solo disparo del cliente y el registro de tiempos
(`LLMClient.complete_once()`, `LLMClient.timing_record()`), de modo que no depende de
la contabilidad privada del stream. Los resultados se guardan en caché por una semana en `profiler.json` junto a
la configuración del usuario; los fallos son silenciosos.

`auto_tune` (por defecto **false**) se aplica en el mismo punto que los valores por defecto de `local-moe`
en `config.load()`. Cuando está activado usa solo números y sus fuentes: VRAM libre (o
RAM unificada), los pesos y la caché KV estimados y la velocidad de prefill medida. Reduce
un contexto demasiado grande hasta un piso de seguridad de 4,096 tokens, propone crecer cuando hay
margen de sobra (nunca lo fuerza), escala los presupuestos de recuperación con el contexto, mantiene el
tiempo límite de 600 s y la concurrencia 1 para los servidores MoE por niveles, y se abstiene cuando no hay telemetría
disponible. Los valores explícitos del usuario siempre ganan. La tabla completa de reglas y la matriz de plataformas
están en [Configuración](CONFIG.md#perfilado-avanzado-del-sistema-y-auto-ajuste).

## Perfiles de ejecución y el nivel `local-moe`

Cada modelo se clasifica en uno de tres **perfiles de ejecución**. La clase decide los presupuestos
de salida, los tiempos límite y la concurrencia que Pedacito usa para él.

| Perfil | Se aplica a | Comportamiento |
|---|---|---|
| `local-dense` | Un modelo local convencional (menos de unos 35B de parámetros) | Una cola de GPU, presupuestos de salida modestos, un tiempo límite corto. |
| `local-moe` | Un servidor de mezcla de expertos por niveles (Strata, Avalon o un modelo que coincida) | Razonamiento de estilo frontera en hardware local, con los presupuestos y tiempos límite de abajo. |
| `hosted` | Una API remota | El muestreo, los presupuestos y los controles de razonamiento propios del proveedor. |

### Cómo se clasifica un modelo como `local-moe`

La detección necesita cualquiera de estas señales:

- **La etiqueta del runtime o del proveedor, o el nombre del perfil,** contiene `strata`, `avalon`,
  `local-moe`, `local_moe`, `localmoe` o `tiered-moe`.
- **El id de modelo de un endpoint local** coincide con un patrón de MoE por niveles. Ejemplos son
  `qwen3.8-flash`, `deepseek-v4.1-flash`, `qwen3-35b-a3b`, o cualquier id que contenga `moe`.

Un endpoint remoto es `local-moe` solo cuando la etiqueta de su proveedor lo dice. Un modelo alojado cuyo
nombre parece un MoE sigue siendo `hosted`.

### Qué cambia con `local-moe`

Aplicar el nivel solo **sube** los presupuestos. Nunca baja un valor que tú o un preajuste hayan definido.

- **Presupuestos de respuesta.** Los pasos y las respuestas reciben al menos 8,192 tokens. Las llamadas de plan y compilación reciben al
  menos 16,384. Los bloques de razonamiento cuentan contra el presupuesto de respuesta, así que un presupuesto que le sirve a un modelo
  denso de 9B corta a un MoE de razonamiento a mitad del pensamiento.
- **Tiempo límite.** Al menos 600 segundos, para la carga de pesos y un prefill lento de 30,000 tokens.
- **Concurrencia.** Exactamente **un stream**. Estos servidores retienen la mayor parte de la VRAM y la DDR5, y un segundo
  prefill concurrente puede expulsar los pesos residentes.
- **Esfuerzo de razonamiento.** `reasoning_effort` es `default` (no envía nada), o `low`, `medium` o
  `high`, que se reenvían como `reasoning_effort`. Nunca cambia el muestreo. El modelo local
  conserva su propia temperatura.

### Configura un perfil `local-moe`

```toml
[profiles.strata]
base_url         = "http://127.0.0.1:8000/v1"   # example port; use your server's address
provider         = "strata"
model            = "your-moe-model-id"
tier             = "local-moe"
reasoning_effort = "high"
```

La clave `tier` acepta `tiny`, `small`, `medium`, `large`, `local-moe` y `hosted`. Definir
`tier = "local-moe"` completa los presupuestos de arriba. La detección cubre el caso en que la dejes
sin definir.

## Roles: un modelo distinto para cada trabajo

```toml
[universal]
roles = { planner = "big", verifier = "big", summarizer = "tiny" }

[profiles.local]                  # the default worker: lookups and edits
model = "qwen3-9b"

[profiles.big]
base_url = "https://api.deepseek.com/v1"
api_key  = "sk-..."
model    = "deepseek-chat"

[profiles.tiny]
model = "qwen3-1.7b"
```

Los roles son `planner` (convierte una meta en la lista de verificación de `auto` y en los pasos del modo de
plan), `verifier`, `summarizer` (los resúmenes de una línea del índice y la
compactación) y `repair`, más dos peldaños de recuperación: `escalate` (un intento más
en ese modelo) y `fallback` (toma el control cuando el trabajador falla con un
error). Cada peldaño se usa como máximo una vez por tarea y solo cuando está configurado, de modo que
un mal día de un modelo local no pueda quemar silenciosamente un presupuesto alojado. Un rol sin definir
usa el modelo activo; un rol cuyo servidor es inalcanzable recurre a él
con una advertencia. El uso se cuenta por sitio de llamada en cualquier caso. `--role
planner=big` define uno para una sola ejecución.

`escalate` se dispara con una razón registrada, no solo por la falta de una edición
(`auto_escalate`, activado por defecto): una tarea que termina sin ninguna edición aplicada, dos
arrendamientos sin progreso en una tarea, pruebas que siguen fallando con errores de
ejecución/semánticos (un TypeError, una aserción fallida, un bucle que nunca termina) tras las
rondas de reparación, o `escalation_zero_edit_runs` ejecuciones consecutivas sin ediciones en
el perfil activo (por defecto 3, contadas por perfil en
`.pedacito/profile_stats.json`; 0 desactiva la estadística). Una razón gana el
reintento incluso cuando SÍ se aplicaron ediciones —un parche que se aplica pero no funciona es
aun así una tarea fallida— y el primer intento se conserva solo cuando estaba en verde
y el escalado volvió en rojo. Cada razón también emite un evento
`escalation_suggested`, de modo que la señal llega a una persona o a un orquestador externo incluso cuando no hay un rol `escalate` configurado. Cuando no existe un rol
`escalate` distinto, una razón disparada compra UN reintento con el mismo modelo con el estilo de edición
invertido (anchor <-> quote, sin definir -> anchor) en lugar de nada: un
evento `strategy_shift` lo marca, las ejecuciones verdes nunca se reintentan y
`auto_escalate = false` lo desactiva. Un éxito borra la cuenta de ejecuciones sin ediciones
del perfil y cualquier sugerencia en cola.

`pedacito profiles --recommend` sugiere un perfil por rol (worker, escalate,
verifier, planner, summarizer) con una razón de una línea, a partir de la evidencia medida de
ajuste al arnés y de evaluación; `--apply-roles` escribe los roles una vez que
confirmas.

La recolección tiene una división que no es un rol: `scout_profile = "tiny"` ejecuta los pasos de
búsqueda del ciclo de recolección (`read`, `grep`, `find`, `outline`, `callers`, `context`, `usages`, `run_test`) en
el servidor de ese perfil mientras el perfil *arquitecto* activo sigue planificando,
sintetizando, respondiendo y editando. Un scout desconocido o inalcanzable recurre
al arquitecto con una advertencia, su uso igual aterriza en el mismo registro de
sesión, y la GUI muestra un scout activo como una insignia `Scout: <profile>`
junto a la etiqueta del modelo. En una sola GPU mantiene libre el contexto del modelo grande durante
la investigación; con dos es la división natural: tarjeta barata para el
ciclo conversador, tarjeta cara cargada una sola vez para la respuesta. En Settings esto
es un menú desplegable sobre los perfiles configurados (o "None (use active
profile)"). Los tamaños a combinar y la salvedad de una sola GPU están en
[../../MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md#dual-gpu--dual-model-enrutamiento-scout--arquitecto-scout_profile).

### Enrutamiento automático por especialista

Los roles dividen una tarea por *trabajo*; el enrutamiento por especialista divide una ejecución por *tarea*.
Con `auto_route_specialties = true` (la casilla **✦ Auto-route** de la barra superior,
o `--auto-route` por ejecución; `--no-auto-route` anula la configuración), cada tarea
va al perfil cuyas etiquetas `specialties` le encajan mejor: un modelo de código para
"refactor the parser", un modelo de visión para "fix the button layout in
styles.css", un razonador para "optimize the sorting algorithm". Las tareas se
puntúan por palabras clave y por las extensiones de los archivos que nombran; un
prefijo `[specialty: TAG]` en el título de una tarea fuerza la etiqueta, y al planificador se le
indica que agrupe las tareas de la misma especialidad cuando las dependencias lo permitan.

Las etiquetas provienen de dos lugares, y una lista explícita siempre gana:

- `specialties = ["coding", "testing_qa"]` en un perfil (en la GUI: las etiquetas
  bajo **⚙ Settings → Profiles → Connection & Capabilities**, más un campo
  de slug personalizado). Etiquetas canónicas: `general`, `coding`, `frontend_ui`,
  `3d_graphics_svg`, `math_algorithms`, `testing_qa`, `sql_data`,
  `translation_i18n`; cualquier slug en minúsculas (`[a-z0-9_-]+`) también funciona.
- `pedacito calibrate` las detecta: heurísticas del nombre del modelo (familias coder,
  modelos `-vl`/visión, razonadores `r1`/QwQ), las etiquetas
  recomendadas de la biblioteca de arquitecturas, más seis microsondas: un SVG bien formado con atributos
  exactos, un cálculo entero, una traducción al español que conserva los placeholders, un join de SQL
  calificado ejecutándolo contra una base de datos de prueba, una pregunta de testing/QA respondida como JSON estricto (`avg([])` lanza
  `ZeroDivisionError`) y un plan de tres pasos que debe ordenar parser → CLI →
  pruebas; cada una es una llamada diminuta con el razonamiento desactivado. Las etiquetas aterrizan en la tarjeta
  (`measured.specialties`), se muestran en `pedacito profiles` y en el renderizado de la tarjeta,
  y completan las `specialties` sin definir de un perfil exactamente como lo hace `prompt_pack`.

Los cambios ocurren **solo en los límites de las tareas**, nunca a mitad de la recolección, de modo que el
prefijo KV de la conversación de recolección siga siendo cacheable en el servidor que la posee
(../../TECHNIQUES.es.md §2 y §50). En un solo servidor local la transferencia es limpia: el
modelo saliente se descarga primero (Ollama `keep_alive: 0`, LM Studio
de mejor esfuerzo) para que el especialista quepa en la VRAM, y luego se carga y se calienta; los perfiles
en servidores distintos (dos GPU, o un especialista alojado) permanecen calientes juntos
sin descarga. Cada cambio emite eventos `specialist_routed` y `model_swap`,
los tokens y el costo del especialista se suman al registro principal de la sesión,
y un especialista inalcanzable recurre al trabajador con una advertencia. Una
tarea que no coincide con nada, o cuya etiqueta ganadora el perfil activo ya
tiene, se queda donde está; cuando un especialista anterior aún ocupa la tarjeta, se
devuelve primero al trabajador.

Una tarea que termina sin una edición utilizable también se recuerda en
`.pedacito/attempts.json` (archivos consultados, por qué); a una tarea similar posterior se le dice
qué se intentó ya. Un éxito la borra. Las mismas ejecuciones se cuentan por
perfil en `.pedacito/profile_stats.json`, que alimenta la estadística de
escalamiento de arriba.

## Enrutamiento híbrido: un arquitecto planifica, un editor implementa

El enrutamiento híbrido divide una tarea entre dos modelos. Un **arquitecto** potente la planifica. Un
**editor** rápido, el perfil activo, la implementa. El arquitecto suele ser un modelo más grande,
posiblemente alojado. El editor suele ser un modelo local que ejecutas todo el día.

La tarea se ejecuta en dos etapas:

1. **Etapa 1, arquitecto.** Una llamada enfocada devuelve un `HYBRID_BLUEPRINT`: la causa raíz,
   los archivos objetivo, las firmas y contratos exactos que se deben conservar y los casos límite.
2. **Etapa 2, editor.** El editor reúne contexto y escribe las ediciones. El plano se
   inyecta antes del contexto reunido.

Entre las etapas, el servidor que alojaba al arquitecto se expulsa si los dos modelos viven en
servidores locales distintos (`auto_evict_servers`). En una sola GPU, un arquitecto de 27B y un editor de 9B
pueden turnarse en la tarjeta. Un arquitecto alojado coexiste con un editor local y no necesita
expulsión.

La telemetría se combina. Los tokens del arquitecto y del editor se suman en un total de la ejecución, con un desglose
por etapa. Si no se puede alcanzar al arquitecto, la tarea degrada a una ejecución simple del editor e
imprime una advertencia.

### Actívalo

En la GUI, usa el interruptor **Hybrid Architect** en la barra de Options y elige un proveedor y un modelo.
En la TUI, usa `/hybrid off` o `/hybrid <provider> <model>`. En la configuración:

```toml
[universal]
hybrid_routing     = true
architect_provider = "big"                  # a [profiles.*] name, a provider label, or a base URL
architect_model    = "your-architect-model-id"
```

El enrutamiento híbrido permanece desactivado mientras `architect_model` esté vacío. Un modelo arquitecto igual al
modelo activo es una operación nula, porque no hay nada que dividir.

El enrutamiento híbrido es independiente del [enrutamiento de roles](#roles-un-modelo-distinto-para-cada-trabajo). Los roles eligen
un modelo para un trabajo, como la planificación o la verificación. El enrutamiento híbrido es una división en dos etapas de
una sola tarea.

## Arena de sandbox con dos modelos

La arena compara dos modelos con el mismo prompt sin tocar tu proyecto. Cada
contendiente trabaja en su propia copia aislada del espacio de trabajo. La copia excluye `.git`, `.venv`,
`node_modules`, checkpoints, `dist` y `build`, y el proyecto real nunca se modifica.

El resultado es una tarjeta de puntuación lado a lado para cada contendiente: si tuvo éxito, archivos
modificados, un diff unificado, tiempo transcurrido, tokens por segundo, tokens totales y un resumen. El último
resultado se almacena en `.pedacito/arena_last.json`.

Aplicar a un ganador genera un checkpoint antes y después, de modo que la adopción puede deshacerse con
`pdc undo`. Usa el botón **Arena (A/B)** de la GUI, o `/arena <a> <b> <prompt>` y
`/arena apply a|b` de la TUI. La arena no tiene un comando de CLI independiente.

## Qué puede hacer el modelo mientras recolecta

El modelo trabaja con un conjunto pequeño de acciones:

| Acción | Propósito |
|---|---|
| `read` | Líneas numeradas de un archivo, o una definición por nombre de símbolo. Una lectura a mitad de archivo anota la función/clase que la contiene y las importaciones del módulo, de modo que el modelo siempre sepa dónde está. |
| `grep` | Búsqueda por regex o literal (proyecto, archivo, directorio o glob), con los resultados etiquetados con su símbolo contenedor. Recurre a coincidencias literales y sin distinguir mayúsculas, y sugiere símbolos similares cuando no hay resultados. |
| `find` | Busca definiciones indexadas por nombre, firma o resumen, navegación por concepto, donde `grep` necesita la grafía exacta del código fuente. Primero coincidencias de subcadena, luego BM25 (de modo que un concepto de varias palabras como `retry backoff` funciona). Una línea por resultado con rangos de línea exactos; si no hay resultados, sugiere nombres similares. |
| `callers` | Dónde se define un símbolo, qué llama, quién lo llama, y los archivos y pruebas que un cambio en él puede afectar (importadores/llamadores transitivos, dos saltos). |
| `context` | Una respuesta en lugar de cuatro: el código fuente numerado de una definición, el encabezado de su clase, las firmas que llama, sus llamadores con los sitios de llamada, sus pruebas y el probable radio de impacto. También funciona sobre un archivo. |
| `outline` | Cada unidad de un archivo (o de los archivos en un directorio/glob) con firmas y rangos de línea. |
| `sessions` | Busca o lee registros de ejecuciones anteriores. |
| `diff` | Qué cambió desde las propias instantáneas previas a la escritura de Pedacito (`allow_diff`, activado por defecto). |
| `git` | git de solo lectura: `status`, `diff`, `log`, `show`, `blame` (`allow_git`, activado en un repositorio). |
| `remember` | Guarda un hecho duradero (un comando de compilación, una convención, una trampa) en `.pedacito/memory.md` para ejecuciones futuras. |
| `skill` | Carga un procedimiento con nombre de la lista de skills (`add-a-test`, `fix-a-failing-test`, ...). |
| `todo` | Lista de verificación corta `- [ ]` guardada en `.pedacito/todo.md` y mostrada en el prompt de respuesta. |
| `question` | Te hace *a ti* una pregunta corta (solo en sesiones interactivas, como máximo dos veces por tarea). |
| `run_test` | Ejecuta una verificación de prueba acotada (pytest, Node/Vitest, Go o Cargo, elegidos según el objetivo) y lee el traceback condensado. Activado por defecto; consulta [Ejecuciones de pruebas acotadas](#ejecuciones-de-pruebas-acotadas-durante-la-investigación) |
| `api` | Lee firmas públicas exactas de una biblioteca que la tarea nombra y los manifiestos del proyecto declaran. Primero `.pyi`/`.py`/metadatos de distribución y `node_modules/**/*.d.ts` contenidos; un fallo es desconocido, nunca prueba de ausencia. Consulta [Oráculo de API sin conexión](#oráculo-de-api-sin-conexión-api_oracle_exec) |
| `done` | Deja de recolectar y responde. |

Repetir una búsqueda de solo lectura (incluso muchos pasos después) sobre la que nada ha cambiado
recibe un puntero al resultado anterior en lugar de una segunda copia, y cualquier
observación que no sea de lectura (un `grep` amplio, un comando conversador, una respuesta de MCP) se limita
a una fracción de la ventana (`max_observation_tokens`) con una nota sobre cómo
acotarla.

La lista de acciones se **filtra según lo que está habilitado y el tamaño del modelo**: un
perfil tiny ve once acciones y un prompt de ~450 tokens, no dos docenas de herramientas y
consejos de eval del navegador que no puede usar.

Acciones opcionales, cada una detrás de un indicador de configuración porque amplían lo que un modelo puede
hacer:

| Acción | Se habilita con | Propósito |
|---|---|---|
| `write` / `append` / `move` / `delete` | `allow_write = true` | Operaciones de archivo completo; `append` construye un archivo más grande que una sola respuesta. |
| `rename` | `allow_write = true` | Renombra mecánicamente una definición y sus referencias en los archivos que la importan (cadenas y comentarios intactos, cada archivo se vuelve a analizar, con instantánea). Rechaza un nombre definido en varios lugares y lista las referencias que dejó intactas. |
| `undo` | `allow_write = true` (+ `backup`, activado por defecto) | Autorrecuperación a mitad de tarea: revierte lo que hicieron hasta ahora las acciones `write`/`append`/`move`/`delete` de esta tarea (o solo uno de sus archivos), las instantáneas previas a la tarea regresan, los archivos que creó la tarea desaparecen. |
| `run` | `allow_shell = true` | Ejecuta un comando (p. ej. `pytest -q`), sin shell, de modo que `&` y `nohup` se rechazan y los procesos de larga duración van a `serve`. Una GUI/servidor silencioso se detiene después de `shell_idle_seconds`. |
| `serve` | `allow_serve = true` | Inicia, lista, muestra los registros de o detiene un servidor de desarrollo en segundo plano. El registro y los logs viven bajo `.pedacito/procs/`; también `pedacito serve` / `pedacito procs`. La lista colapsa las entradas detenidas reemplazadas (historial de reinicios) tras un contador; `procs --all` las muestra. |
| `diagnose` | `allow_shell = true` | Ejecuta el linter/verificador de tipos detectado (`ruff`, `pyflakes`, `flake8`, `mypy`/`pyright` cuando están configurados, `npm lint`, `tsc`, `cargo check`, `go vet`, `rubocop`). También disponible como `pedacito diagnose`. |
| `lsp` | `allow_lsp = true` | Definición/referencias/hover/diagnósticos exactos mediante pyright/pylsp/typescript-language-server/rust-analyzer/gopls, o comandos en `lsp_servers`. También `pedacito lsp`. La puerta de diagnósticos opcional (`lsp_diagnostics_gate`, que también requiere `allow_lsp`) se ejecuta en `loop.py` alrededor de los lotes de ediciones de símbolos: una línea base antes de la edición y una comprobación con versión coincidente después comparten un único presupuesto de 2,9 s, y solo los hallazgos nuevos en líneas cambiadas pasan a ser hallazgos de reparación. `lsp.py` mantiene las rutas contenidas mediante `security.resolve_safe_path` y convierte las URI de unidad y UNC de Windows. Los estados `unavailable`, `timeout`, `stale` o `denied` nunca se muestran como correctos. |
| `screenshot` | `allow_screenshots = true` | Captura la pantalla como texto: tamaño, brillo, veredicto de pantalla en blanco, OCR, descripción opcional con `vision_model`. Necesita el extra `vision`. |
| `browser` | `allow_browser = true` | Navegador headless: inicia un comando de servidor, abre la página, hace clic/escribe/mantiene teclas e informa el texto de la página, los tamaños de canvas y los errores de consola. Necesita el extra `browser`. |
| `mcp` | `allow_mcp = true` + `mcp_servers = [...]` | Llama a herramientas, prompts (`prompt:server/name`) o recursos (`resource:server/uri`) de servidores MCP. El catálogo muestra los argumentos, las llamadas se validan, se mantiene caliente una sesión activa por servidor, y se reintenta un servidor que estaba caído al iniciar. |
| `mcp_call` | `enable_mcp = true` + `[mcp_servers.<name>]` | Llama a una herramienta de un servidor MCP *con nombre*. Solo se muestran las hasta cuatro herramientas cuyos nombres y descripciones coinciden con la tarea, la acción en sí aparece solo cuando una coincide, y los servidores no confiables consultan al intermediario de permisos en cada llamada. Consulta [Servidores MCP con nombre](CONFIG.md#servidores-mcp-con-nombre). |
| `task` | `allow_subagents = true` | Entrega una pregunta de investigación acotada y de solo lectura a un agente auxiliar con una ventana de contexto nueva; solo regresa un resumen breve. La profundidad está limitada (1 local, 2 alojado). |
| `fetch` | `allow_fetch = true` + `fetch_allowed_hosts` | URL → texto legible, con lista de hosts permitidos, tamaño limitado, en caché, etiquetado como no confiable. Las direcciones privadas/loopback se rechazan —incluidos los nombres públicos que *resuelven* a ellas— a menos que las nombres explícitamente, p. ej. `fetch_allowed_hosts = ["127.0.0.1", "localhost"]` para leer tu propio servidor de desarrollo; la conexión se fija a la IP verificada. |
| búsqueda semántica | `semantic_search = true` | Un `grep` sin resultados recurre al endpoint `/v1/embeddings` del servidor. |

`run` y las dos acciones MCP ejecutan comandos elegidos por el modelo en tu proyecto.
Trátalas como una
decisión de confianza, pero consulta [Seguridad](SECURITY.md): `run` está en sandbox por
defecto donde el sistema operativo lo permite, y el resto del radio de impacto también está limitado.

### Resolución del entorno

`env_resolve.resolve_environment` es un detector sin conexión. Lee marcadores
del proyecto (`.venv`, metadatos de uv/poetry/pipenv/conda, entornos de
tox/nox, archivos de bloqueo de pnpm/yarn/bun, manifiestos de Cargo/Go) y
rutas de intérpretes existentes; nunca ejecuta un script de activación,
`uv`/`poetry`/`conda`/`tox`/`nox`, `npx`, una instalación ni una petición de
red. La precedencia es fija: un intérprete existente del proyecto, un
`VIRTUAL_ENV`/`CONDA_PREFIX` activo explícito, los metadatos del gestor de
paquetes y por último el `PATH` ambiental. El resultado es un
`ResolvedEnvironment(kind, python, bin_dirs, package_manager, workspace_root,
evidence)` congelado; los entornos competidores se nombran en la línea de
evidencia `ambiguous:` en lugar de descartarse en silencio. Las rutas
descubiertas en el proyecto pasan por `security.resolve_safe_path`, así que un
enlace simbólico o junction de `.venv` que escape del proyecto se rechaza.

`env_resolve.command_env` parte de `safety.child_env` y solo antepone los
directorios bin resueltos, de modo que `run_test`, `diagnose`, los servidores
LSP, los comandos laterales y los hijos de `run` ven las herramientas del
proyecto mientras la depuración con lista permitida sigue eliminando secretos
ambientales. Un `test_command` explícito sigue siendo autoritativo y nunca se
reescribe. Los scripts y manifiestos descubiertos por el resolutor son datos,
nunca comandos: nada se ejecuta solo porque un manifiesto lo nombre.

### Ejecuciones de pruebas acotadas durante la investigación

Un modelo que solo lee código tiene que adivinar qué hará una corrección. `run_test` le permite
comprobar la suposición primero: ejecuta una verificación de prueba acotada durante la investigación, de modo que
el traceback en vivo esté frente al modelo antes de que escriba el parche.

```json
{"action": "run_test", "target": "tests/test_app.py::test_save", "filter": "save or retry"}
```

- **Acotado.** `target` es un archivo de pruebas, un id de nodo o un directorio. `filter` es una
  expresión `-k` opcional (un patrón de nombre para los demás ejecutores). El modelo
  nunca proporciona una línea de comandos de shell.
- **Sigue el ejecutor del proyecto.** Los objetivos de Python usan pytest. Un archivo
  `*.test.js`/`*.spec.js`/`*.test.ts`/`*.spec.ts` usa Vitest cuando el paquete
  o el comando detectado lo nombra, y de lo contrario el `node --test` integrado de Node.
  `*_test.go` (o un directorio que contenga uno) usa `go test ./dir`, y
  `tests/*.rs` usa `cargo test --test NAME`.
- **Se ofrece cuando es útil.** Aparece cuando el proyecto tiene pruebas (`tests/`,
  `test_*.py`, `*.test.js`) o la tarea nombra un fallo (test, fail, bug, error,
  traceback, reproduce, broken).
- **Limitado.** Cada ejecución está limitada a 15 segundos y al mismo presupuesto de tokens que una
  lectura. La salida es una línea de conteo más hasta cuatro tracebacks de fallo condensados que
  muestran solo los marcos del proyecto. Una ejecución verde se informa como verde, de modo que el modelo no
  la vuelva a ejecutar para verla pasar otra vez.
- **En sandbox.** Se ejecuta bajo el mismo sandbox que `run` (sin red, escribe solo
  dentro del proyecto), y la fase de respuesta no tiene herramientas en absoluto. La ejecución se registra como un hecho del repositorio, de modo que un comando de prueba
  que sigue funcionando se convierte en un comando recordado.
- **Honesto sobre lo que significa un código de salida.** La respuesta nombra el veredicto con
  palabras —`EXIT: 5 (NOT COLLECTED)`— y lo explica: 5 significa que no se
  recolectó nada, lo que no dice nada sobre si el código funciona. Un `target` que
  no declara ningún `def test_...` ni ninguna `class Test...` se rechaza antes de que pytest
  inicie, con la alternativa que funciona explicitada (ejecutarlo con `run` como script),
  porque un simple `no tests ran` una vez hizo que un modelo reescribiera las funciones de un script de humo
  independiente que pasaba como `test_*` para hacer desaparecer el número.

Está activado por defecto (`enable_run_test_in_gather = true`). Defínelo en `false` para que
cada ejecución pregunte primero en el diálogo **Permission requested** bajo **Test Execution**.
Una ejecución simple de CLI no tiene diálogo, así que allí la herramienta informa que está deshabilitada.

### Oráculo de API sin conexión (`api_oracle_exec`)

`api_oracle.py` responde una pregunta concreta durante la recolección: *¿cuál es
la firma pública exacta de este símbolo en una biblioteca que la tarea nombró?*
Se ofrece como la acción de solo lectura `api` únicamente cuando el texto de la
tarea nombra una biblioteca que los manifiestos del propio proyecto (metadatos
y requisitos de Python, `package.json`, `go.mod`, `Cargo.toml`/`Cargo.lock`)
declaran exactamente. Las tareas que no nombran esa dependencia nunca ven la
acción, así que su esquema y su prompt quedan idénticos byte a byte.

La ruta por defecto es estática y acotada:

- primero archivos `.pyi` de Python, luego `.py` y metadatos de distribución
  (`*.dist-info/METADATA`, `top_level.txt`) bajo los directorios
  `site-packages` del intérprete resuelto;
- `package.json` y declaraciones `*.d.ts` de un `node_modules` contenido
  (incluida la alternativa `@types/<nombre>`);
- las firmas se extraen con `ast` o analizando líneas de declaración, nunca
  importando el paquete ni evaluando sus valores por defecto.

`env_resolve.dependency_roots` entrega las raíces aprobadas: el proyecto, su
workspace, los `site-packages` del intérprete resuelto y un `node_modules`
contenido. Cada archivo candidato pasa por `security.resolve_safe_path`, y el
recorrido omite symlinks/junctions, de modo que un enlace de declaración que
escapa de una raíz aprobada se ignora en vez de leerse.

El enriquecimiento opcional está doblemente condicionado por `api_oracle_exec`
(un ajuste de `EXEC_FIELDS`, por defecto `false`) y el consentimiento existente
`allow_shell` más el motor de políticas: `go doc -all PAQUETE` (con
`GOPROXY=off`), `cargo metadata --offline --no-deps --format-version 1` o el
`inspect` del intérprete resuelto. Esos hijos son solo argv, se ejecutan con `env_resolve.command_env` y el manejo
de sandbox/plazo de `proc.run`, y su salida está acotada, redactada y etiquetada
`go-doc`/`cargo-metadata`/`python-inspect`. Con el indicador apagado, esas rutas
devuelven `unknown`, no una "API faltante" inventada.

El índice vive en `.pedacito/cache/api/`. Su clave es un SHA-256 sobre el nombre
del paquete, la identidad del entorno resuelto (tipo, intérprete, directorios
bin, workspace), la versión y los hashes de contenido de las declaraciones y
archivos de bloqueo inspeccionados. `fileio.write_bytes` hace el reemplazo
atómico, así que un lector concurrente ve un registro completo viejo o nuevo, y
una versión o declaración cambiada simplemente falla la clave y poda la
anterior. Los registros guardan solo metadatos de paquete/símbolos (redactados,
con topes de 512 KB y 900 símbolos), nunca código fuente, prompts ni salida de
herramientas.

### Lecciones de reparación y trampas del repositorio

Cada vez que una ronda de reparación corrige una verificación fallida, Pedacito registra cuál fue el fallo
y que la corrección funcionó. La lección va a `.pedacito/micro_facts.json` bajo
`anti_patterns`, y las ejecuciones posteriores en ese repositorio la ven como una pista:

```
[REPO GOTCHAS & LESSONS] (repair lessons distilled from earlier runs; hints, not truth):
- dom_id_mismatch: '#add-task-btn' was queried by JS but no rendered element carried that id
```

- **Solo correcciones comprobadas.** Una lección se escribe solo cuando una compuerta (sintaxis, estática,
  contrato de frontend) o una prueba se puso en rojo durante la tarea y vuelve a estar en verde tras
  las propias rondas de reparación de la tarea. Una suposición sin verificar nunca se registra.
- **Texto simple, sin prosa del modelo.** Cada lección proviene de una plantilla fija que
  completa los nombres del fallo, de modo que puede verificarse con las mismas herramientas.
- **Acotadas y autolimpiables.** Se conservan como máximo ocho lecciones. Las repeticiones se fusionan
  y se cuentan, no se agregan. Una lección sobre archivos que ya no existen se descarta.
  Cada lección también lleva contadores `runs` / `recurred` / `avoided` de las
  ejecuciones de verificación observadas: una que enfrentó cinco ejecuciones sin evitar nunca su tipo de fallo se
  retira a `retired_lessons` (las reglas autopromovidas están exentas). Son
  hechos del arnés, nunca reflexiones escritas por el modelo.
- **Con alcance, no a gritos, cuando se repiten.** Una lección confirmada dos veces cuyos
  archivos viven todos bajo un directorio (`src/api/...`, `frontend/...`) se promueve
  a una regla con alcance de ruta administrada automáticamente en `.pedacito/rules/auto_<dir>.md`
  (`paths: <dir>/*`, `auto_generated: true`, hasta cinco líneas sin duplicados). Entonces
  aparece solo cuando el modelo lee un archivo de ese directorio, no en
  cada prompt de cada tarea. Cuando se eliminan todos los archivos que nombra una lección promovida,
  su línea se descarta y una regla automática vacía se elimina. Los archivos sin
  la marca `auto_generated: true` son tuyos; Pedacito nunca los edita ni los elimina.
- **Pistas, nunca verdad.** El ciclo vuelve a verificar todo en vivo, de modo que una lección equivocada
  cuesta una mirada, no un cambio equivocado.

El archivo también contiene los comandos que Pedacito confirmó en este repositorio (consulta
[micro-hechos](../../TECHNIQUES.es.md)).
Es JSON simple. Léelo, edítalo o elimínalo para olvidar todo lo que ha
aprendido. `micro_facts = false` desactiva el registro y la inyección, y
`max_micro_fact_tokens` limita cuánto de él llega al prompt.

### Ejecuciones en rojo: pruebas inestables, infraestructura y enrutamiento de la reparación

`pedacito/test_classify.py` decide qué fallos de una ejecución de pruebas en rojo posterior a una edición pueden llegar a la reparación del modelo. Solo puede hacerlo `real_failure`. Las identidades de fallo y las diferencias con la línea base provienen del mismo `loop.failure_identities` que usa el paquete de reparación.

- Un timeout, un error de recolección, una dependencia ausente u otra parada de infraestructura se clasifica antes de cualquier reejecución. Nunca se vuelve a ejecutar ni se repara.
- Un nodo de pytest nuevo que falla se vuelve a ejecutar como máximo una vez por verificación, con el runner original y el mismo argv y sandbox; como máximo se reejecutan tres nodos. Una reejecución que pasa y muestra evidencia de paso es un `flake`. La misma aserción que vuelve a fallar es un `real_failure`. Cualquier otro caso es `unknown`.
- Los runners sin identificador de nodo estable (npm, `node --test`, cargo, go) nunca reciben sintaxis de pytest inventada, así que sus fallos nuevos son `unknown` y se detienen como no verificados.
- Los identificadores en cuarentena de `.pedacito/flakes.json` se informan y no se reejecutan. Un fallo en cuarentena nunca convierte una ejecución en verde.
- Cada decisión es un evento `test_category` con un motivo redactado y acotado. Los resultados no verificados se registran como un hallazgo de error que indica que la ejecución no está verificada. Nunca se registran como aprobados.

Esto no afirma que una reejecución demuestre determinismo. Una reejecución que pasa es evidencia de inestabilidad, no prueba de que la prueba esté bien. El sandbox de Windows no cambia esta ruta.

## Qué se indexa

Los archivos de código se analizan en funciones, clases y métodos con rangos de línea
exactos (los métodos de Go se califican con su receptor, los bloques `impl` de Rust, los
`mod` anidados y los namespaces de C# se fragmentan, y un script sin funciones se
divide en los límites de las sentencias en lugar de convertirse en una unidad gigante). Una unidad mayor que `max_unit_lines` (220), `max_unit_chars` (12000) o
`max_unit_tokens` (3000) se divide en partes `name#2`, `name#3`, ... para que una sola unidad enorme
función enorme no sature a un modelo pequeño. Otros archivos útiles (`.md .txt .rst .csv
.tsv .json .toml .yaml .yml .ini .cfg .conf .env .sql .sh`) se indexan
de forma estructural: el mapa muestra las columnas de un CSV, las claves de un archivo JSON, los
encabezados de un archivo markdown, y así sucesivamente, mientras que `read` y `grep` llegan a su contenido.

Otras extensiones son invisibles hasta que las agregas: `--ext .kt` (o
`extra_extensions` en la configuración) las dirige por el fragmentador estructural.

**Excluir archivos**, en orden creciente de precedencia:

1. `.pedacitoignore` en la raíz del proyecto (sintaxis de gitignore); `.gitignore` también se
   lee.
2. Opciones: `--exclude 'deprecated/'`, `--include 'deprecated/still_used.py'`,
   `--no-refs` (solo código), `--no-gitignore`.
3. Nombrar una ruta directamente siempre gana.

**Niveles del mapa.** Los proyectos pequeños reciben un listado plano de cada función; los más grandes
reciben una tarjeta por archivo; los muy grandes reciben un **mapa ordenado**: las tarjetas de los
archivos con más probabilidad de ser relevantes para la tarea (nombrados en ella, que coinciden con sus palabras, cercanos
a esos en el grafo de importaciones, probados por ellos o que los tocan, sin confirmar o
cambiados recientemente), cada una con una línea `why listed:`, sobre un esqueleto jerárquico de
directorios de todo lo demás. Un mapa que cabe se lista alfabéticamente
y se mantiene idéntico byte a byte entre tareas para que sobreviva la caché de prompt del servidor;
el ordenamiento solo decide *qué* mostrar cuando hay que dejar algo fuera.
`read`, `outline` y `find` llegan al texto exacto desde cualquier nivel. Un resumen del repositorio
(stack, puntos de entrada, comandos de prueba/lint/compilación, roles de directorios) se sitúa sobre el
mapa, construido a partir de tus manifiestos sin llamada al modelo.

**Mapa del repositorio y el prefijo KV.** El resumen y el mapa se colocan en **user0**, el
primer turno del usuario, nunca en el prompt del sistema. `messages[0:2]` es por lo tanto el
prefijo en caché que comparte cada paso de recolección dentro de una tarea, y el prompt del sistema se mantiene
idéntico byte a byte entre tareas. El ordenamiento es determinista para un índice y una tarea dados,
de modo que las llamadas repetidas devuelven texto de mapa idéntico byte a byte. `flat_map_token_budget`
(por defecto 4,000, autoajustado por el perfilador) es la clave absoluta para el nivel
plano; los proyectos más grandes recurren a tarjetas de archivo y luego a un mapa ordenado sobre un
esqueleto de directorios. Los anclajes explícitos (incluida una skill inyectada manualmente) se
resuelven antes de la recuperación automática y se sitúan antes del mapa, de modo que nunca se
puntúan ni se expulsan.

Las tarjetas de archivo llevan el radio de impacto, `imported by: 7 file(s)`, o un mensaje etiquetado
"no importers or callers found (evidence, not proof)", y vinculan los módulos
con sus pruebas (`tested by:` / `tests:`) para Python, JS/TS (`x.test.ts`,
`__tests__/`), Go (`x_test.go`), Rust (`tests/`), Java/C# (`XTest`) y Ruby
(`x_spec.rb`) cuando la coincidencia es inequívoca. `pedacito map PROJ -t "task"`
imprime exactamente lo que vería una tarea.

**Los secretos nunca se indexan.** `.env`, `.env.*` (pero no `.env.example`),
`*.pem`, `*.key`, `id_rsa*`, `.npmrc`, `.pypirc`, `credentials.json` y
similares se omiten por defecto; vuelve a incluir uno deliberadamente con `!.env` en
`.pedacitoignore`. Los directorios excluidos (`node_modules`, `build`, ...) se
podan antes de recorrerse, y los archivos `.gitignore` anidados se respetan.

## Edición: buscar/reemplazar, no diffs

```
--- FILE: src/app.py
<<<<<<< SEARCH
    def is_leech(self, lapses):
        return lapses >= 8
=======
    def is_leech(self, lapses):
        return lapses >= self.leech_threshold
>>>>>>> REPLACE
```

Un bloque coincide exactamente con el archivo (ignorando los espacios finales) o falla
limpiamente, y un fallo muestra la ventana coincidente más cercana en el archivo en ese
momento, con números de línea y una puntuación de similitud, de modo que el reintento se vuelve a anclar en
texto real en lugar de adivinar. Un encabezado FILE también puede anclar un bloque desde el principio:
`--- FILE: src/app.py LINES 40-60` reemplaza ese rango exacto, y
`SYMBOL is_leech` reemplaza esa definición completa, útil cuando un fragmento
aparece más de una vez o el texto sigue desplazándose bajo el modelo.

Una definición también puede reemplazarse *sin* repetir su texto antiguo en absoluto, que
es lo que el modelo suele equivocar. Nombra el símbolo en lugar de buscarlo,
y escribe solo el cuerpo nuevo:

```
--- FILE: src/app.py
<<<<<<< SYMBOL src/app.py :: Report.summarise
    @staticmethod
    def summarise(rows):
        return sum(rows)
>>>>>>> END_SYMBOL
```

El tramo proviene del propio AST del archivo (`ast` para Python, tree-sitter para
JavaScript/TypeScript/Go/Rust), decoradores incluidos, de modo que un bloque no puede aterrizar en
medio de algo. La indentación se normaliza a la columna real de la definición,
y los decoradores iniciales se conservan cuando el cuerpo nuevo no los
repite. `<<<<<<< AFTER_SYMBOL file :: name` empalma el cuerpo *después* de esa
definición en lugar de reemplazarla (agrega un método sin volver a emitir la
clase). Las mismas dos operaciones existen como cargas estructuradas para el muestreo
restringido: `{"edit_action": "replace_symbol", "file": ..., "symbol": ...,
"replacement": ...}` y `{"edit_action": "insert_after_symbol", ...}`. Los bloques simples de
SEARCH/REPLACE no se ven afectados y ambas formas pueden aparecer en una misma respuesta.
El resultado enfrenta la misma compuerta de sintaxis: un cuerpo que no se analiza se rechaza y
no se escribe, y un nombre de símbolo desconocido o ambiguo regresa con una
instrucción de hacer `outline` del archivo o volver a emitir el bloque con `LINES a-b`.

Antes de escribir, Python se verifica con `ast.parse` y los lenguajes de tree-sitter
con sus analizadores: una edición que rompe un archivo limpio se rechaza, mientras que
un archivo que ya estaba roto se edita con una advertencia. El error de un bloque fallido
regresa al modelo para hasta `repair_rounds` reintentos (por defecto 2).

Después de una escritura, `post_edit_check` (activado por defecto) ejecuta la verificación barata, el
linter detectado o, si no, un análisis de sintaxis, sobre los archivos modificados y devuelve solo los
hallazgos que *no* estaban antes de la tarea, dándole al modelo hasta
`repair_rounds` rondas de corrección para limpiar su propio desorden sin perseguir
problemas preexistentes.

Define `fuzzy_match = true` para permitir que los bloques casi coincidentes se anclen al texto más
similar (`fuzzy_min_ratio`, por defecto 0.85). Esas ediciones se marcan como aproximadas.
Un bloque que el modelo *empezó* pero nunca terminó (un `=======` o
`>>>>>>> REPLACE` faltante, un bloque sin encabezado `--- FILE:`) se informa como
fallo en lugar de descartarse en silencio.

### Renombrar un símbolo en todo el proyecto

Un renombrado no necesita un modelo, así que Pedacito lo hace mecánicamente: cada identificador
exacto se reescribe, y los comentarios se dejan intactos. Los literales de cadena
también se dejan intactos a menos que pases `--strings` (todos ellos) o `--include-labels`
(solo cadenas de etiquetas de prueba).

```bash
pedacito rename apply_discount discounted_price                       # preview the diffs
pedacito rename apply_discount discounted_price --file pkg/core.py    # that file and its importers
pedacito rename apply_discount discounted_price --strings             # also whole-word string literals
pedacito rename toggleStep flipStep --include-labels                  # also test/it/describe/suite labels only
pedacito rename apply_discount discounted_price --apply               # write it
```

Un nombre calificado acota el renombrado a una clase: `pedacito rename Store.save save_record`
renombra `Store.save` y los sitios de llamada cuyo receptor es demostrablemente un `Store` (Python),
y lista los parecidos que dejó intactos. La vista previa es lo predeterminado, así que no se necesita ninguna opción para obtenerla. `--apply` escribe el
cambio y guarda primero las instantáneas de los originales, de modo que `pedacito restore` lo deshace.

- **Primero los identificadores.** Los nombres de Python pasan por `tokenize`. JavaScript,
  TypeScript, Go y Rust pasan por sus nodos identificadores de tree-sitter. Una
  cadena o comentario que simplemente contiene el nombre antiguo nunca se toca
  por defecto; `--strings` además renombra las apariciones de palabra completa dentro de
  literales de cadena (etiquetas de prueba, documentación que nombra la API), mientras que los comentarios permanecen
  intactos. `--include-labels` es la opción intermedia acotada: solo se mueven los literales de cadena
  que son etiquetas de prueba (el primer argumento de `test`/`it`/`describe`/
  `suite`, incluidos `.only`/`.skip`); toda otra cadena se queda.
- **Todo o nada.** Cada archivo modificado debe pasar la verificación de sintaxis antes de que se
  escriba cualquier archivo. Si uno se rompiera, no se escribe nada.
- **Basado en nombres, a propósito.** Un parámetro de una función no relacionada que comparte el
  nombre también se renombra. La vista previa lo muestra, para que puedas juzgar antes de escribir.
- **También disponible para el modelo.** Con `allow_write = true`, el ciclo de recolección tiene una
  acción `rename_symbol` que llama al mismo motor, y pasa por las mismas
  reglas de permisos e instantáneas que las demás escrituras.

En la GUI, **Project ▾ → Rename symbol…** toma el nombre antiguo, el nombre nuevo y un
archivo de alcance opcional. **Preview** muestra los diffs y **Rename** los escribe.
El diálogo renombra solo identificadores; `--strings` es una opción de la CLI.

### Verificar una respuesta antes de escribirla

El fallo habitual de un modelo pequeño no es una sintaxis incorrecta sino una edición plausible en el
lugar equivocado. Antes de escribir nada, `verify_answer = "deterministic"` (el
valor por defecto) aplica en seco los bloques y ejecuta verificaciones gratuitas, sin llamada al modelo:

| Verificación | Qué detecta |
|---|---|
| unread | una edición a un archivo, o a una región de uno, que la fase de búsqueda nunca `read` |
| placeholder | `TODO` / `NotImplementedError` / `pass` simple en lugar de código funcional (a menos que la tarea pida un stub) |
| stale references | una definición eliminada o renombrada que aún se referencia en otro lugar, juzgada contra lo que dejan *todas* las ediciones de la respuesta, de modo que un renombrado correcto en varios archivos pasa |
| moved imports | un archivo movido que aún se importa por su ruta antigua |
| citations | afirmaciones `file:line` sobre archivos nunca leídos |
| coverage | un requisito de la tarea (extrae hasta cinco) que ninguna edición ni oración toca; una advertencia, y en el ciclo de edición incremental siembra una pasada de seguimiento acotada |

Los errores provocan una nueva pregunta corregida que le muestra al modelo sus propios bloques y
los hallazgos; las advertencias se imprimen y se registran. `verify_answer = "full"` agrega una llamada
de verificador adversarial (en el rol `verifier`, si definiste uno), y
`best_of = 3` muestrea candidatos extra a una temperatura más alta y conserva el
de mejor puntuación; ambos cuestan llamadas al modelo, y `pedacito calibrate` los
activa para los modelos que miden débiles.

Con `verify_tests = true` y `allow_shell = true`, las pruebas *afectadas* por una
edición (las pruebas con el nombre del archivo, las pruebas de sus importadores y llamadores) se ejecutan
después de escribirla; los fallos nuevos, comparados con una ejecución antes de la edición, regresan
al modelo para hasta `repair_rounds` rondas de corrección. Un fallo que el registro de comandos
ya había anotado antes de esta ejecución se etiqueta *pre-existing failure
(recorded before this run)* y no dispara reparación, a menos que esta ejecución
haya capturado su propia línea base, que reemplaza al historial. `test_command`
anula la detección (`pytest`, `npm test`, `cargo test`, `go test ./...`,
`make test`, ...). Esas dos opciones solo desactivan las verificaciones por **subproceso**.
En un árbol mixto (pruebas de Python junto a `*.test.js`), el ejecutor sigue el
lenguaje del cambio: una edición solo de Python ejecuta `pytest` incluso cuando un script de pruebas de
`package.json` ganaría la detección, y una edición de JavaScript
ejecuta el ejecutor de JS. Un `test_command` explícito nunca se anula. Las compuertas de sintaxis, estática, frontend y de contrato entre archivos leen archivos en
este proceso, de modo que examinan cada edición aplicada incluso en el perfil por defecto
`allow_shell = false`: una ejecución que no puede ejecutar pruebas aun así no puede
entregar un grafo de módulos que no enlazaría.

### Verificaciones de contrato del frontend

Pasar las pruebas no demuestra que una página web cargue. Después de una edición aplicada, y
antes de que se ejecute el comando de pruebas, Pedacito verifica cinco cosas sin llamada al modelo:

- **Enlaces de recursos.** Cada `<script src>` y `<link href>` local en un archivo HTML
  debe apuntar a un archivo que exista. Las URL externas, los URI `data:` y los marcadores de plantilla
  (`{{ }}`, `<%= %>`, `${...}`) se omiten, porque su ruta real
  no puede conocerse de forma estática.
- **Ids del DOM.** Cada `getElementById('x')` y `querySelector('#x')` literal debe
  coincidir con un `id="x"` en algún archivo HTML, o con un id que el propio JavaScript cree. Una
  búsqueda construida con un id interpolado no se verifica, ya que su valor no se
  conoce hasta el tiempo de ejecución.
- **Sintaxis.** El JavaScript modificado debe analizarse. Se usa `node --check` cuando Node está
  instalado, y de lo contrario la gramática de tree-sitter. Analizar no es enlazar: un
  archivo puede analizarse y aun así llegar muerto, así que se verifican dos formas más
  contra la gramática.
- **Enlaces de módulo.** Un nombre enlazado dos veces a nivel de módulo —el segundo
  `import { WAYPOINTS }` que una edición incremental agrega a un archivo que ya
  lo importa— es un `SyntaxError` por el que el navegador rechaza todo el grafo.
  Se escanea el nivel superior de cada módulo ES en busca de esto, con excepción de los pares `var`/`function`
  porque son redeclaraciones legales.
- **Llamadas inalcanzables.** Un módulo ve solo los nombres que declara o importa. Una
  llamada a un nombre que ningún módulo enlaza es un `ReferenceError` en el momento en que la línea
  se ejecuta, y el hallazgo indica qué corrección aplica: el hermano que lo exporta
  (agregar la importación), el archivo que lo declara sin exportarlo (agregar el
  `export`), o nada en absoluto (definirlo o quitar la llamada). Los globales del navegador,
  los paquetes de CDN y los globales de scripts clásicos nunca se informan. El
  parámetro propio de un callback de flecha cuenta como enlazado, de modo que `rows.forEach(m => m.name)`
  no es una llamada a una `m` no definida.

Un hallazgo se informa con la misma forma que las demás compuertas, por ejemplo:

```
FrontendContractError in app.js:42: queries DOM element '#add-task-btn',
but index.html only defines ids ['board', 'new-task-btn', 'todo-col']
```

Solo los hallazgos que son nuevos para la tarea actual bloquean la edición, de modo que los problemas
existentes en otras partes del proyecto no detengan trabajo no relacionado. Si editas un archivo
HTML, el JavaScript que busca sus ids se vuelve a verificar. La compuerta está activada por
defecto y se desactiva con `frontend_contracts = false`. `pedacito webcheck PROJ`
ejecuta el mismo vocabulario más las verificaciones heurísticas (sintaxis ESM en scripts clásicos,
globales duplicados, importaciones no declaradas, llamadas a miembros de globales de la página) a pedido,
solo como advertencias, y `auto` imprime sus hallazgos tras las tareas que tocan archivos
web. Tanto `pedacito new --apply` como `pedacito new --complex` ejecutan la mitad de
examen sobre el árbol terminado —el ciclo autónomo más que nada, ya que cada
una de sus tareas escribe un archivo en un contexto nuevo y nunca ve la página completa—:
un andamiaje cuya página no puede enlazar recibe rondas de reparación, y si las violaciones
sobreviven a ellas la compilación imprime `BUILD INCOMPLETE` y sale con 1 en lugar de
informar un proyecto que no puede iniciar.

### Verificación de humo del DOM headless (opcional)

La compuerta de frontend demuestra los contratos que un navegador impone al *cargar*; no puede
demostrar que la página realmente se *monta*. Con `verify_dom = true` Y `allow_shell =
true` (o `ask --verify-dom`), Pedacito también carga la página de entrada en Chromium headless
tras las ediciones web aplicadas y compara la superficie montada con una
instantánea tomada antes de la edición. Un módulo de entrada reemplazado que deja de
renderizar en silencio —elementos o controles con nombre desaparecidos, texto visible colapsado— es un
hallazgo de la compuerta `dom` que alimenta una ronda de reparación, junto con errores de página no capturados y
recursos/módulos que ya no cargan. La misma verificación detecta los fallos silenciosos de Web Audio:
un script de inicialización registra contextos, llamadas a resume, el grafo de conexiones, inicios de fuentes
y la automatización de ganancia; una sonda de humo activa un paso si la cuadrícula
está vacía, hace clic en Play y recorre los controles deslizantes. Un contexto nunca reanudado, un grafo
que nunca llega a `ctx.destination`, un botón Play que inicia cero fuentes,
ganancias fijas en cero o una automatización inválida de AudioParam también son errores de la compuerta `dom`
—Web Audio falla sin errores de consola, así que nada más los ve.
Las páginas sin audio permanecen inactivas (cero hallazgos de audio). Los errores de página preexistentes
no se le atribuyen a la tarea. Necesita la instalación opcional de Playwright (`pip
install playwright && playwright install chromium`); sin ella la verificación
informa una pista de instalación y la compuerta se hace a un lado. `dom_entry` anula la
página de entrada detectada y `dom_timeout` limita una carga. `pedacito domcheck PROJ`
la ejecuta a pedido (esa invocación es el consentimiento para ejecutar código del proyecto);
`--snapshot`/`--baseline` comparan dos ejecuciones. La sesión se ejecuta en un proceso hijo
con un límite de reloj, de modo que una página cuyo script nunca cede se detiene y se
informa como *not verified* en lugar de bloquear la ejecución. A diferencia de los hallazgos de nivel de aviso basados en regex de `webcheck`,
los errores de esta herramienta pueden impulsar la reparación. Consulta ../../TECHNIQUES.es.md #45.

## Cómo se administra el contexto

Una ventana pequeña se gasta de tres maneras: el texto fijo del prompt, el mapa y la
conversación. Pedacito gasta cada una a propósito.

- **Paquetes de prompts.** Las instrucciones vienen en densidad `full`, `small` y `tiny`
  (`prompt_pack`, o si no la tarjeta del modelo, o si no el nivel) y mencionan solo
  las acciones que están habilitadas. Con navegador, shell y escritura desactivados, el prompt de
  búsqueda por defecto es de unos 1,700 tokens en lugar de más de 3,000; `tiny` es de unos 450.
  Sobrescribe cualquier sección con `.pedacito/prompts/{gather,edit_format,answer}.md`.
- **Un solo punto de pliegue.** Un servidor guarda en caché el *prefijo* del prompt; reescribir un turno
  antiguo descarta la caché desde ese turno en adelante. Por eso la conversación de búsqueda se
  deja intacta hasta que la ventana está realmente llena, y entonces toda la franja más antigua se
  pliega en **un** resumen `STATE` (qué se buscó, una línea cada uno) y
  los últimos pasos permanecen textuales. Una sola ruptura de caché, no una por paso. Un modelo puede
  escribir el resumen en lugar del extractivo por defecto (`compaction = "model"`).
- **La respuesta continúa la conversación de búsqueda** (`answer_mode`): la instrucción
  final se anexa, de modo que el servidor reutilice todo el prefijo caliente en lugar de
  hacer prefill del mapa y del código fuente por segunda vez. Si la conversación se
  plegó, o la respuesta continuada está vacía o aún tiene forma de llamada a herramienta,
  recurre a un prompt reconstruido. Las APIs alojadas usan `auto` por defecto; los modelos locales
  `rebuild` hasta que una tarjeta de modelo diga otra cosa.
- **Degradar, no rechazar.** Un prompt de respuesta que excede la ventana descarta las notas
  del proyecto, luego reduce el mapa y luego descarta los elementos de código fuente *completos* más antiguos
  (dejando un puntero de relectura para cada uno) antes de dar error.
- **Recuperación de atascos.** Un paso repetido, una serie de pasos fallidos o lecturas
  repetidas reciben un prompt de recuperación dirigido ("you sent the identical `read` four
  times; do something different") antes de que la recolección se rinda.
- **Contabilidad visible.** Cada tarea imprime llamadas, tokens de entrada/salida, la
  proporción de aciertos de caché y (con `price_*_per_mtok` definido) el costo; `/cost` lo desglosa
  por sitio de llamada; `max_total_tokens` / `max_cost` (o `--max-tokens-budget` /
  `--max-cost`) detienen la ejecución antes de la siguiente llamada.

## Estado del proyecto en disco

Todo lo que Pedacito escribe para un proyecto vive bajo `<project>/.pedacito/`:

```
index.json            the map (versioned, written atomically)
summaries.json        summary cache, keyed by content hash
embeddings.json       semantic-search cache (if enabled)
backups/<stamp>/      pre-write snapshots (never merged: -2, -3 on a clash)
sessions/<stamp>.md   what was asked, looked up, and changed (secrets masked)
checkpoints/          time-travel tree (index.json + snapshots/) and ask/chat history
events/<stamp>.jsonl  one JSON line per step, call, fold, verify, task
reviews/<stamp>.html  saved review pages
screenshots/          captures (if enabled)
procs/                background process records and logs
design.json/.md       spec-driven design (interfaces, data, tests)
memory.md             remembered facts (hints for the model)
consolidated_session.md  the newest session-consolidation handoff
micro_facts.json      commands confirmed to work here, and repair lessons
rules/*.md            path-scoped rules       skills/           your skills
                                             (NAME/SKILL.md or NAME.md)
prompts/*.md          prompt overrides        fetch/              fetch cache
prompt_reports/       prompt provenance (hashes/counts; no text)
todo.md               current task checklist
log.txt               timestamped event trace
```

`sessions/` es el lugar donde mirar cuando una edición resulta incorrecta días después;
`log.txt` muestra dónde ocurrió un fallo. Sin un proyecto abierto, los modos de la GUI
sin proyecto (How y New) no tienen carpeta `.pedacito/`, así que sus chats
y registros de sesión van a una carpeta de estado por usuario:

```
~/.local/state/pedacito/            Linux/macOS ($XDG_STATE_HOME replaces
                                    ~/.local/state)
  threads/global/<id>.json          chats opened with no project
  sessions/global/<stamp>.md        task logs written with no project
%LOCALAPPDATA%\pedacito\            Windows
```

Abrir un proyecto cambia a los hilos propios de ese proyecto bajo
`<project>/.pedacito/`; el chat sin proyecto permanece en la carpeta de estado y
regresa cuando el alcance vuelve a ser sin proyecto. No se copia nada entre
los dos.

En la primera ejecución Pedacito crea un
`.pedacitoignore` comentado y agrega sus entradas a `.gitignore` (desactívalo
con `create_ignore_file = false` / `manage_gitignore = false`). Eliminar
`.pedacito/` siempre es seguro; solo cuesta una reindexación. `pedacito reset PROJ`
lo hace en un solo paso: detiene los servidores en segundo plano registrados y luego borra el
estado de ejecución (índice, resúmenes, planes, sesiones, checkpoints, respaldos, eventos,
revisiones, capturas de pantalla, caché de fetch, procs, `log.txt`) dejando intactos tu código,
`.pedacito.toml`, `.pedacitoignore` y `.gitignore`. `--keep-memory`
conserva los hechos recordados, `--keep-config` conserva las `rules/`,
`skills/`, `prompts/` y los archivos de instrucciones redactados por ti, y `--reindex` reconstruye el mapa
(`--no-summaries` hace esa reconstrucción estática, sin llamadas al modelo). `--dry-run`
muestra lo que se iría; los git worktrees activos bajo `.pedacito/worktrees/` hacen que se
rehúse a menos que pases `--force`. `pedacito prune` conserva las N sesiones,
respaldos y archivos de eventos más recientes y recorta `log.txt`.

## Módulos de v0.3.3 de un vistazo

- **`pedacito/gitops.py`**: git guiado. Cada llamada a git usa solo argv. Ejecuta el escaneo previo de
  secretos sobre el diff preparado (commit) o el diff de la rama (pr), y nunca hace push ni force-push.
  `pedacito commit`, `pedacito pr`, `/git` en la TUI y el cajón Git de la GUI llaman a las mismas funciones.
- **`pedacito/acp.py`**: el puente Agent Client Protocol (`pedacito acp`). Es una capa de encuadre
  JSON-RPC 2.0 sobre el motor caliente `rpc.py`, así que la lógica de tareas no se duplica. Las
  notificaciones `session/update` salen del mismo flujo de eventos que emite `rpc`.
- **`pedacito/rpc.py`**: `params.question` en `explain`, `--events on|off` y `--describe`. El `id` del
  cliente se conserva junto a `req_id` en los eventos transmitidos.
- **`pedacito/config.py: _endpoint_is_local`**: el único predicado de localidad (loopback, `.local`,
  RFC 1918, CGNAT RFC 6598 / Tailscale, link-local, ULA de IPv6, etiquetas de proveedor local).
  `config`, `llm`, `modelcard`, `servers`, `model_catalog` y `cli` delegan en él.
- **Volcado a `pedacito/tools.py` (scratch)**: las observaciones grandes de las acciones que producen
  salida se escriben en `.pedacito/scratch/*.log`. El modelo las lee y busca con `read` y `grep`, y no
  puede escribirlas.
- **Formato de herramienta `tool_mode = "plain"`**: un formato de palabras simples junto a los formatos
  grammar y native. `calibrate` lo compara A/B.
- **Enrutador (`proactive_escalation`)**: enrutador de dos vías opcional. Una compuerta estructural sin
  tokens escala las tareas difíciles; un Scout de pocos ejemplos en la frontera mantiene en el worker las
  tareas simples de un solo archivo. Desactivado por defecto.
- **`pedacito/tui.py`**: la vista enriquecida sin dependencias: diffs en línea, tarjetas plegables de
  llamadas a herramientas (`Ctrl+O`, `/expand`, `/collapse`), el selector `@` de seis filas y las zonas de
  ratón SGR. Sigue siendo una capa de terminal solo con la biblioteca estándar.
- **`pedacito/i18n.py` + `pedacito/profiles.py`**: los encabezados de `pedacito profiles` y el título de
  recomendaciones se buscan con `i18n.t`, así que `pedacito --lang es profiles` imprime etiquetas en
  español. El inglés sigue siendo el predeterminado.

## Registro de proveedores y evidencia de capacidades

`pedacito/providers.py` es el único registro detrás de las peculiaridades de
compatibilidad alojadas, los presets de alta rápida de la GUI y la matriz de
capacidades. Enumera 24 proveedores alojados más LiteLLM y tres marcadores solo
pasarela, y 13 runtimes locales (LM Studio, Ollama, llama.cpp, vLLM,
Lemonade/lemond, Strata, Avalon, MLX-LM, SGLang, TabbyAPI, KoboldCpp, LocalAI).
La coincidencia de host es igualdad o un sufijo delimitado por puntos / etiqueta
`*`, nunca una subcadena, así que un host parecido no hereda las afirmaciones de
caché o capacidades de un proveedor.

`ProviderCapabilities` mantiene cada función en tres estados: `None` es
desconocido y nunca se presenta como soportado. `get_capabilities` funciona sin
red por defecto y combina la evidencia del catálogo, los metadatos de models.dev
en caché y los rechazos observados en vivo por `llm._JSON_CAPS` /
`_ACCEL_CAPS`. `probe_capabilities` solo se ejecuta tras consentimiento
explícito, hace una petición mínima y acotada, rechaza redirecciones para que la
autenticación no cruce un límite de host, y guarda una entrada de caché
redactada de 24 h con clave URL normalizada + modelo exacto + hash de identidad
de autenticación (nunca el secreto). Azure OpenAI, AWS Bedrock y Google Vertex
son entradas solo pasarela cuya fuente de capacidades dice `unverified: requires
configured OpenAI-compatible gateway`; no son adaptadores nativos.

Los transportes conservan el razonamiento opaco entre turnos de herramientas del
asistente: `reasoning_content` compatible con OpenAI, bloques de pensamiento
firmados de Anthropic y partes de pensamiento de Gemini / `thoughtSignature` se
llevan como campos separados del protocolo, no como contenido visible, y se
excluyen de los registros y de las cachés de capacidades.

## Aislamiento de sesiones, daemon y subagentes paralelos (v0.4.0)

v0.4.0 permite que varias sesiones se ejecuten en un mismo proceso, o detrás de `pedacito daemon`,
sin compartir estado de ejecución. El contrato siguiente reúne las reglas que los criterios de
aceptación de `tests/test_v040_acceptance_gates.py` demuestran de extremo a extremo.

### El contrato de aislamiento de sesiones

| ID | Regla | Dónde vive |
|---|---|---|
| D5 | Los servidores de modelos locales se comparten mediante arrendamientos; nunca se desalojan bajo una sesión activa | `leases.py` |
| D6 / D15 | El estado del proyecto está protegido por un bloqueo, y el mantenimiento espera a las sesiones y daemons activos | `workspace.py` |
| D7 | Cada tarea escribe un diario de escritura anticipada (WAL) duradero que sobrevive a un fallo | `journal.py` |
| D8 | Una sesión que usa un worktree tiene su propia rama y directorio; un árbol sucio se rechaza, se transporta o se guarda en stash, nunca se mezcla en silencio | `worktree.py` |
| D9 | El enrutamiento de revisiones y la cola de revisión pertenecen a una sola sesión | `review.py`, `session_ctx.py` |
| D10 | Los artefactos sensibles nunca llegan a checkpoints, ratchet, copias de respaldo, espejos ni revisiones | `security.py`, `edits.py`, `workspace_overlay.py` |
| D11-D12 | El protocolo de sesión lleva su versión y sus funciones por sesión | `rpc.py`, `daemon.py` |
| D13 | Cancelar una sesión ejecuta una cascada determinista solo sobre sus propios recursos | `resources.py`, `session_ctx.py` |
| D14 | La ejecución desatendida se realiza dentro de un sandbox del sistema operativo o se rechaza | `sandbox.py`, `sandbox_win.py` |

Dos notas sobre la numeración. Los números D son las etiquetas de paquete de trabajo que usan el
código y las pruebas; algunos (D1-D4) los usan otras áreas de prueba para trabajo no relacionado y no
forman parte de este contrato. La suite de criterios es la definición ejecutable del contrato.

### SessionContext y propagación de contexto

Un `SessionContext` (`session_ctx.py`) posee todo lo que modifica una sesión: su clon de `cfg`, `idx`,
`client`, `event_bus`, `permission_broker`, `model_lease`, `budget_ledger`, `worktree`, `policy`,
`review_queue`, `resource_registry`, `run_telemetry` y los espacios de `tool_caches`. El contexto activo
es un valor de `contextvars`. `use_session(ctx)` lo enlaza para un bloque,
`spawn_thread_with_context(...)` inicia un hilo con una copia del contexto del llamador, y
`ContextThreadPoolExecutor` copia el contexto en cada `submit`. Los hilos iniciados desde una sesión
ven solo esa sesión, y las tareas concurrentes nunca compiten por el presupuesto, la telemetría o la caché
de otra.

### Eventos, permisos y el daemon

`SessionEventBus` (`events.py`) es un flujo monótono por sesión con `replay(after_seq)` acotado. Cada
envoltorio lleva su `session_id` y su `seq`. Los eventos también se escriben en el registro JSONL de la
raíz, en `<raíz>/.pedacito/events/`, etiquetados de la misma forma. `SessionPermissionBroker`
(`security.py`) indexa las solicitudes pendientes por `(session_id, request_id)`: una decisión de otra
sesión se rechaza con `session mismatch`. Una solicitud no interactiva o con `approval_policy = "never"`
se deniega de inmediato y emite un evento `permission_decision`.

`DaemonServer` (`daemon.py`) es un servicio HTTP/SSE solo de loopback que posee `WorkspaceSession`s
aisladas. Cada ruta está limitada a un id de sesión: `/v1/sessions/<id>/events`, `/prompt`, `/cancel`,
`/permission` y `/review`. Cada solicitud necesita el token portador de la ejecución. El hook `quiesce`
del daemon hace que `check_maintenance_safe` rechace `reset` y `prune` mientras una sesión del daemon esté
activa.

### ModelLeaseManager

Los modelos locales son un recurso compartido y costoso. `ModelLeaseManager` (`leases.py`) concede un
arrendamiento por `(servidor, modelo)`. Un arrendamiento marca su modelo como en uso: `can_evict()` es
False mientras haya algún arrendamiento activo, y `servers.evict_other_local_servers()` omite los modelos
arrendados, así que el modelo de una sesión nunca se descarga bajo ella. Una sesión que necesita otro
modelo local espera en cola. Un arrendamiento `exclusive=True` (por ejemplo `calibrate`, con
`LeasePriority.EXCLUSIVE`) vacía primero los arrendamientos activos y luego se ejecuta, sin desplazar a una
sesión que ya está en marcha. Los endpoints alojados no pasan por el grupo.

### ProjectLockManager y mantenimiento

`ProjectLockManager` (`workspace.py`) mantiene un solo escritor por raíz de proyecto. Una segunda
reclamación exclusiva desde otra sesión lanza `ProjectBusy`, y las reclamaciones compartidas coexisten.
`atomic_update_json` realiza lectura-modificación-escritura bajo un bloqueo de archivo, así que las
actualizaciones concurrentes no se pierden. `check_maintenance_safe(root, action=...)` devuelve
`(False, motivo)` mientras un hook de quiesce registrado informe sesiones activas o el bloqueo de
ejecución esté tomado, y `force=True` es la única anulación.

### TaskJournal (WAL)

`TaskJournal` (`journal.py`) escribe archivos WAL versionados con nombre UUID
(`.pedacito/journal/<wal_id>.jsonl`, separados del título redactado de la tarea),
mantiene números de secuencia monotónicos al reabrir un `wal_id` explícito y hace
fsync por defecto en cada línea. Los fallos de escritura o fsync marcan la
durabilidad como perdida en lugar de silenciarlos; los diarios truncados,
corruptos o no duraderos se informan como no recuperables en vez de ignorarse.
`list_recovery` inspecciona de forma conservadora tanto diarios UUID v2 como
diarios heredados por título, y `recover` (`pedacito recover
list|resume|rollback|cleanup`, vista previa por defecto y `--apply` para mutar)
verifica bloqueos entre procesos, seguridad de mantenimiento, propiedad y
manifiestos SHA-256 de checkpoints antes de restaurar o limpiar, negándose a
marcar la tarea como terminal si falla la restauración o la durabilidad del
registro terminal. Un `run.lock` obsoleto que dejó un proceso muerto lo toma
`acquire_lock`.

### SessionResourceRegistry y cancelación

Todo lo que inicia una sesión registra una limpieza en su `SessionResourceRegistry` (`resources.py`):
servidores de desarrollo, clientes LSP y MCP, espejos temporales, worktrees efímeros y servidores de
revisión. `SessionContext.cancel()` deniega las solicitudes pendientes, libera el arrendamiento del modelo,
ejecuta `teardown_all` en orden inverso y despierta cualquier revisión que la sesión esté esperando. Solo se
tocan los recursos de esa sesión. Un hijo de `proc.run` se detiene mediante el evento de cancelación de la
sesión, y su árbol de procesos se recoge antes de que termine la cancelación.

### Subagentes paralelos acotados

`subagent_max_parallel` (por defecto 3) limita cuántos subagentes se ejecutan a la vez. Se ejecutan en un
`ContextThreadPoolExecutor`, así que cada uno tiene su propio contexto. A un hijo se le da un
`PolicyEngine.child_policy(role=...)` que solo puede intersectar con su padre. Los roles de solo lectura
(`explore`, `verifier`) se fuerzan al modo `plan`.

### Manifiestos de paquete

Una extensión, una habilidad, un conjunto de comandos o un paquete de modelos se declara en un
`pedacito-package.toml` en su raíz: su `kind`, las `capabilities` que solicita y la versión mínima de
Pedacito que necesita. `pkg_manifest.py` valida el archivo, y una persona lo revisa antes de que se ejecute
nada del paquete. Ver [SKILLS_AND_TOOLS.md](SKILLS_AND_TOOLS.md#manifiestos-de-paquete-pedacito-packagetoml).

### Autopsia y disponibilidad de herramientas

`autopsy.autopsy(root, run_id="last")` es una función pura sobre los registros
de eventos tipados locales. Nunca llama a un modelo. Cada categoría proviene de
un evento que el arnés ya escribe con un campo legible por máquina: un `step`
con `obs_code` (en `step` o en el `tool_result` intercalado, fijado por
`autopsy.classify_observation` ante una marca fija del arnés, nunca ante prosa
del modelo), `payload_cut`, `step_rejected`,
`gate_error`, `verify` (tipos de hallazgo `unread` y `syntax`), `task_budget`,
`incremental_partial`, `llm_error` (`context_overflow` o `schema_rejection`) y
`test_result`. Gana la categoría más grave, los empates van al `seq` más
antiguo, y la evidencia solo contiene ids, números de secuencia, el reloj
relativo, el tipo y una lista blanca de campos escalares cortos, todo redactado.
Un registro ausente o antiguo, o ningún evento coincidente, da `unknown`; y
`incomplete` marca una ejecución sin evento de cierre o con un hueco en la
secuencia. `flake` y `api_hallucination` no tienen emisor en esta versión y se
reportan como no verificados.

`tools.inspect_tools(cfg, index, task, force=...)` lee la misma tabla
`_offer_rows` y la misma pasada `_prune_optional_actions` que `schema_for`, así
que el inspector no puede desviarse del esquema que recibe el modelo. Cada fila
lleva un motivo de `active`, `config_disabled`, `policy_denied`,
`missing_dependency`, `not_relevant`, `pruned` o `forced`, además del permiso
que necesita. `--allow-tool` (`cfg._force_tools`, fijado por invocación de
`ask`/`auto` y borrado al terminar la tarea) solo conserva herramientas que la
poda dinámica habría quitado. Las puertas de configuración, capacidad,
consentimiento y política nunca lo leen, y el inspector no puede ejecutar nada.
Listar herramientas MCP para la disponibilidad de `mcp_call` arranca los
servidores nombrados, como ya hace la construcción real del esquema.

### Validación de planes e impacto

`plan.check_plan(root, plan_path, index)` se ejecuta antes de que Auto y Complex
ejecuten su primera tarea (y se expone como `pedacito plan --check [PLAN]
[PROJECT]`). Contiene la ruta del plan y cada mención de archivo con
`security.resolve_safe_path`, rechaza recorridos de ruta, escapadas por
symlink/junction, destinos sensibles y protegidos, resuelve las referencias
explícitas `ruta: símbolo` contra el índice estático, y comprueba las
anotaciones de dependencia por ciclos, números inexistentes y orden de tareas
posteriores. Una tarea que dice crear un archivo es una creación válida cuando
su directorio padre queda dentro del proyecto; un archivo ausente con un verbo
de edición y sin marca de creación es un error definitivo. En Windows una
colisión solo por mayúsculas/minúsculas con un archivo existente es un error
porque el sistema de archivos no puede tener ambos; un sistema sensible a
mayúsculas trata la casi-colisión como advertencia.

`PlanImpact` es una vista previa conservadora construida con el mismo índice:
archivos y símbolos tocados, llamadores indexados (importadores, aristas de
llamada y llamadores de símbolos), pruebas enlazadas por las convenciones de
nombre de cada lenguaje y `unknowns` deterministas para importaciones
dinámicas, llamadores aproximados por nombre simple, archivos sin indexar,
creaciones planificadas y pruebas ausentes. Un índice ausente o una dependencia
sin resolver hacen la vista previa más pequeña y explícitamente desconocida,
nunca verde. Las advertencias no bloquean; solo los errores definitivos detienen
una ejecución, y un plan inválido se asigna al código de salida 2.

### Procedencia del presupuesto del prompt

`prompt_inspect` captura los límites reales del ensamblado en lugar de
reconstruir una aproximación después. `Agent.gather` registra los bloques del
sistema (prompt de recolección, pista del mapa, instrucciones del proyecto,
memoria, catálogo de skills, todo, MCP) y el primer turno del usuario
(orientación, pines explícitos, mapa del proyecto, few-shot, tarea);
`Agent.answer` registra el brief, las notas, los pines, el mapa, el fuente
empaquetado y la tarea después de cualquier degradación. Los dos primeros
mensajes se hashean una vez como prefijo estable, y `llm._http_post` hashea la
carga serializada *después* del redactado de salida (o de la traducción al
transporte nativo). `_account` asciende la sección de transporte a
`exact:provider-usage` solo cuando el proveedor informó un conteo de tokens de
entrada; todo lo demás se etiqueta `estimate:chars-div-3.6`.

Una `PromptSection` guarda nombre, ordinal, conteo de tokens, método de conteo,
hash de contenido, disposición y un motivo acotado con IDs de reglas, skills y
memoria coincidentes. Nunca guarda el texto de la sección, así que un informe no
puede filtrar un prompt, archivo fuente, salida de herramienta ni credencial.
Los informes viven en `.pedacito/prompt_reports/`, resuelven cada ruta con
`security.resolve_safe_path`, llevan la etiqueta de la sesión que los escribió y
se podan a los 40 más recientes. `pedacito prompt --explain [--last]
[--compare RUN_ID]` y la pestaña Prompt de la GUI (`GET /api/prompt/report`) leen
los mismos archivos; una diferencia compara solo IDs, hashes y conteos. Explain
nunca construye un cliente de modelo ni hace una llamada de red, y la captura no
añade ninguna instrucción, mensaje ni byte de prefijo a ninguna petición.

## Límites conocidos

Estos son los límites reales. Las funciones faltantes del arnés no están en esta lista:
la TUI, `pedacito bench`, MCP, LSP, los subagentes y los proyectos de varios archivos están todos
incluidos.

- **El tamaño del modelo fija el techo, no el arnés.** Un modelo de 4B a 8B es un
  excelente scout y un editor confiable de un solo archivo. El trabajo de arquitectura de varios archivos
  es donde los modelos locales de 14B a 30B+, o un `planner` híbrido, se ganan su lugar. Las tareas
  estructurales amplias como "modernize this codebase" exigen a todo modelo pequeño, así que
  divídelas en pasos acotados (`pedacito auto` está hecho justo para eso).
- El grafo de llamadas registra destinos calificados por receptor (`Store.get`, `this.x()`,
  locales tipados) donde el receptor es demostrable, pero una llamada sin tipo aún coincide
  su nombre simple, de modo que dos métodos `run()` no relacionados pueden confundirse cuando el receptor
  es desconocido. La refactorización debe usar el renombrado calificado por receptor (consulta
  [Renombrar un símbolo](#renombrar-un-símbolo-en-todo-el-proyecto)). Las aristas de importación se resuelven por lenguaje
  (rutas relativas, paquetes, `crate::`), pero una importación que no puede vincular con un archivo
  indexado se trata como externa, y la pista "no importers" es evidencia,
  no prueba (carga dinámica, puntos de entrada).
- `py-tree-sitter` 0.26 regresionó el manejo del tiempo de vida de los nodos y puede provocar un segfault en
  CPython 3.12/3.14 (upstream py-tree-sitter#472; reproducible sin
  Pedacito). La dependencia está fijada a `<0.26` (0.25.2 tiene wheels para cp314);
  el CI cubre 3.11–3.14. El fragmentador también pausa el GC cíclico alrededor de los nodos
  vivos como segunda línea de defensa.
- `run` y los comandos de prueba están en sandbox donde funcionan bubblewrap/sandbox-exec (no
  en Windows, ni donde los espacios de nombres de usuario están desactivados: recibes un aviso, no una
  degradación silenciosa); `browser`, `fetch`, MCP, los procesos `serve` en segundo plano y
  los servidores LSP están protegidos (entorno depurado, rutas protegidas, listas permitidas,
  avisos de confianza), no en sandbox.
- `rename` es mecánico y conservador. Un nombre simple se renombra solo cuando tiene
  exactamente una definición en cualquier parte, y luego sigue las importaciones, no los tipos. La herramienta
  del agente además admite una consulta calificada por receptor (`Store.save`): los sitios de llamada de Python
  se renombran cuando el receptor está enlazado de forma determinista a esa clase
  (`x = Store()`, `s: Store`), y todo lo que no puede demostrar —`y.save` en
  otra clase, receptores sin tipo, archivos que nunca importan la definición— se
  lista como "not renamed" en lugar de adivinarse; el seguimiento de receptores es
  solo de Python. Cada escritura de varios archivos es transaccional: un error de instantánea o de
  disco a mitad de camino revierte los archivos ya escritos en ese lote, de modo que ningún
  árbol a medio renombrar sobrevive. Varios bloques de edición entre archivos se aplican por
  archivo a menos que el lote atómico por etapas esté activo (`atomic_edits` explícito, o
  auto-activado para un endpoint local tiny/small/medium): entonces cada candidato se valida
  y su imagen previa se guarda en un WAL duradero antes de la primera escritura, y un corte
  a mitad de la confirmación se revierte en la siguiente ejecución — no es visibilidad atómica
  instantánea a nivel de sistema de archivos. La instantánea y
  `pedacito restore` son la red de seguridad en cualquier caso.
- Lenguajes: Python por AST; JS/TS, Ruby, Go, Rust, Java, C, C++, C#, PHP,
  Kotlin, Swift, Scala y Lua mediante tree-sitter (una rareza de la gramática puede omitir una
  arista de llamada ocasional en Swift); todo lo demás es texto de referencia con búsqueda.
- La búsqueda semántica necesita un endpoint de embeddings. Se ejecuta con candidatos primero: una
  consulta sin resultados incrusta solo la lista corta léxica acotada (las 50 mejores unidades) y los
  dos ordenamientos se combinan con fusión de rango recíproco, nunca una pasada sobre
  todo el proyecto. Los vectores se guardan en caché por hash de contenido y se reutilizan hasta que la
  unidad cambia o se cambia el modelo de embeddings.
- Un modelo solo de texto no puede ver gráficos; `screenshot` y `browser` le dan evidencia
  en texto, no juicio visual.
- Las tareas acotadas y localizadas son donde los modelos pequeños funcionan mejor. Empieza ahí y luego
  amplía el alcance a medida que lo permitan el tamaño de tu modelo y tu hardware.
