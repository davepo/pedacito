🌐 [English](MODEL_GUIDE.md) | **Español**

# Guía de modelos de Pedacito

Esta guía te ayuda a elegir un modelo para tu hardware, configurarlo en Pedacito
y comprobar que funciona. Se aplica a cualquier servidor compatible con OpenAI (LM
Studio, Ollama, Lemonade Server, llama.cpp, vLLM y otros) y a las APIs alojadas.
Los nombres de modelos y los detalles de cuantización cambian con frecuencia. Trata los ejemplos como
puntos de partida y mide en tu propia máquina. La referencia canónica de los
perfiles de modelo y de cada ajuste es [docs/es/CONFIG.md](docs/es/CONFIG.md), y la configuración
de servidores está en [docs/es/INSTALLATION.md](docs/es/INSTALLATION.md).

Pedacito no es un benchmark de chat. Le pide a un modelo que haga tres trabajos:

1. **Elegir búsquedas** mediante un esquema de herramientas JSON (`read`, `grep`, `find`,
   `callers`, `context`, `done`).
2. **Escribir o editar código** como bloques exactos de búsqueda/reemplazo, sobre código fuente que
   realmente haya leído.
3. **Verificar o explicar** lo que encontró, en una respuesta breve en prosa.

Un modelo que gana una tabla de clasificación de chat no siempre es el que gana aquí.
Las llamadas a herramientas confiables, seguir el formato de edición y la disciplina con los tokens importan más
que la cantidad bruta de parámetros.

---

## La versión corta

- **Elige el modelo más grande que quepa con 1.5 a 2.5 GB de VRAM libres.** Un
  modelo que se desborda de la VRAM es mucho más lento que uno más pequeño que cabe.
- **Empieza con el razonamiento desactivado** en tu modelo trabajador. Actívalo solo para un
  perfil de planificador o verificador, y solo después de medirlo.
- **Usa Q4_K_M como cuantización por defecto.** Usa Q5 o Q6 si el modelo aún cabe
  con holgura. Evita Q2 e IQ2 para editar código.
- **Calibra cada modelo una vez** con `pedacito calibrate`. Compara modelos en
  tu propio hardware con los seis benchmarks integrados, o con `pedacito eval`.
- **No elijas una configuración solo por el nombre del modelo.**
- **Deja que el servidor sostenga la gramática JSON.** Pedacito le pide al motor un
  esquema JSON estricto en cada paso de recolección y, por modelo, recurre a
  json_object o a prompts simples cuando un servidor lo rechaza
  (`enforce_json_schema`, TECHNIQUES.es.md §27). Los modelos débiles dejan de fallar
  en la sintaxis.
- **Combina un scout rápido con el arquitecto.** `scout_profile = "<small
  profile>"` ejecuta los pasos de búsqueda de la recolección en un modelo de 4B a 9B mientras el
  modelo grande sigue planificando, respondiendo y editando (TECHNIQUES.es.md §28). En una sola
  GPU mantiene libre el contexto del arquitecto; con dos es la división natural.

---

## Niveles de hardware

Busca la fila que coincide con la VRAM disponible para tu servidor de modelos. Esa es la
memoria de GPU que el servidor puede usar, no el total de la tarjeta si otros programas la usan.
En Apple Silicon u otra máquina con memoria unificada, cuenta solo la porción que
le das al modelo.

| Hardware | Clase de modelo | Ejemplos (Q4_K_M salvo indicación) | Contexto | Nivel de Pedacito | Razonamiento del trabajador |
|---|---|---|---|---|---|
| **Menos de 8 GB de VRAM** | 4B a 7B | Qwen3-4B, Gemma 3 4B, Qwen2.5-Coder 7B (justo) | 4k a 8k | `tiny` o `small` | off |
| **8 a 16 GB de VRAM** | 8B a 14B | Qwen3-8B, Gemma 3 12B, Qwen3-14B (cómodo con 16 GB, justo con 12 GB) | 8k a 16k | `small` o `medium` | off. El perfil de planificador o verificador puede usar razonamiento |
| **24 GB+ de VRAM** (RX 7900 XTX, RTX 3090, RTX 4090) | 24B a 35B, denso o MoE | Mistral Small 24B, Gemma 3 27B, Qwen3-32B, Qwen3-30B-A3B (MoE) | 16k a 32k | `medium` o `large` | off por defecto. Mide antes de activarlo |
| **Dos GPU o dos servidores** | Scout 4B a 9B, arquitecto 24B a 35B | Consulta [enrutamiento scout + arquitecto](#dual-gpu--dual-model-enrutamiento-scout--arquitecto-scout_profile) | Scout: 8k a 16k; arquitecto: 16k a 32k | combinación con `scout_profile` | Por perfil |
| **Híbrido local + nube** | Cualquiera de los anteriores, más un modelo alojado | Un trabajador local, con un planificador o verificador alojado | Por perfil | Por perfil | Por perfil |

Los tamaños de abajo son solo para los pesos. Suma la caché KV y unos 1.5 a 2.5 GB de
sobrecarga.

### Menos de 8 GB de VRAM (4B a 7B)

- **Buenas opciones:** modelos de 3B a 4B en Q4_K_M o Q5_K_M, como Qwen3-4B o
  Gemma 3 4B. Un modelo de código de 7B en Q4 funciona con contexto de 4k a 8k sin otros programas
  usando la GPU.
- **Espera:** un mapa más pequeño y menos código fuente por búsqueda, así que las tareas deben ser acotadas.
  Nombra el archivo en la tarea.
- **Límites de 4B:** las llamadas estructuradas a herramientas y las ediciones de varios archivos son lo primero
  que se degrada en 4B: espera pasos mal formados y ediciones que arreglan un archivo mientras
  rompen otro. Mantén cada tarea en un archivo y una preocupación, y prefiere la
  división con scout (un 4B haciendo búsquedas, un modelo más grande juzgando) en lugar de pedirle al 4B que
  haga ambas cosas.
- **Evita:** modelos de 12B o más, y el razonamiento en un modelo de 4B para ediciones rutinarias.
- **Ajustes:** `tier = "tiny"` (4B o menos) o `"small"` (hasta 10B),
  `disable_thinking = true` y contexto de 4k a 8k. Deja que `auto_fit` elija los
  presupuestos. Define `context_tokens` solo si el servidor informa la ventana equivocada.

```toml
[universal]
vram_gb = 8

[profiles.worker]
model = "qwen3-4b"
tier = "tiny"
disable_thinking = true
```

### 8 a 16 GB de VRAM (8B a 14B)

- **Buenas opciones:** modelos de 8B en Q6 o Q8, un modelo de 12B en Q4 y modelos de 14B en Q4 a
  Q6. Un modelo de 14B en Q4 necesita unos 9 GB para sus pesos, así que es cómodo con
  16 GB y justo con 12 GB.
- **Espera:** que un contexto de 16k sea práctico con 12 GB o 16 GB. Redúcelo si el
  servidor informa menos memoria libre.
- **Evita:** 24B y más grandes en Q4 con este hardware. Los pesos dejan muy poco
  espacio para una caché KV útil.
- **Ajustes:** `tier = "small"` o `"medium"`, `disable_thinking = true` para el
  trabajador y contexto de 8k a 16k. Este es el primer nivel donde vale la pena probar un perfil
  de planificador con razonamiento. Consulta [Enrutamiento de roles híbrido local + nube](#enrutamiento-de-roles-híbrido-local--nube).

```toml
[universal]
vram_gb = 16
roles = { planner = "think", verifier = "think" }

[profiles.worker]
model = "qwen3-14b"
tier = "medium"
disable_thinking = true

[profiles.think]
model = "qwen3-14b"
tier = "medium"
disable_thinking = false
max_tokens_answer = 8000     # room for reasoning plus the edit
context_tokens = 16384
```

### 24 GB de VRAM (24B a 35B, denso o MoE)

- **Buenas opciones:** Mistral Small 24B en Q4 a Q6, Gemma 3 27B en Q4 o Q5,
  Qwen3-32B en Q4 y Qwen3-30B-A3B. Este último es un modelo de mezcla de expertos
  con unos 3B de parámetros activos por token.
- **Modelos MoE:** la VRAM contiene los pesos totales, no solo los activos. Un MoE de 30B
  en Q4 aún necesita aproximadamente 18 GB para sus pesos. Genera más rápido que un
  modelo denso del mismo tamaño, lo que lo hace atractivo con 24 GB. Dimensiónalos según
  el tamaño de archivo que informa el servidor.
- **Espera:** un contexto de 32k en Q4 en una tarjeta de 24 GB sin nada más en la GPU.
- **Evita:** modelos de 70B en Q3 con este hardware. Pueden cargar, pero la calidad de edición
  y la velocidad se resienten.
- **Ajustes:** `tier = "medium"` o `"large"`, contexto de 16k a 32k y razonamiento del
  trabajador desactivado hasta que lo hayas medido. El enrutamiento de roles para el planificador y
  el verificador es útil con este tamaño.

```toml
[universal]
vram_gb = 24
roles = { planner = "think", verifier = "think", summarizer = "worker" }

[profiles.worker]
model = "qwen3-32b"
tier = "large"
disable_thinking = true

[profiles.think]
model = "qwen3-32b"
tier = "large"
disable_thinking = false
max_tokens_answer = 12000
context_tokens = 32768
```

No enrutes `summarizer` a un modelo aparte en una sola GPU a menos que la calidad extra
sea medible. Las llamadas de resumen son frecuentes y recargar un modelo es lento.

### Enrutamiento de roles híbrido local + nube

Un trabajador local se encarga de la mayor parte del trabajo, y un modelo alojado toma los trabajos donde
un modelo más fuerte marca la mayor diferencia: planificar una meta grande, verificar un
cambio o escalar tras un intento fallido. Pedacito enruta cada trabajo por rol,
y cada rol es opcional.

```toml
[universal]
roles = { planner = "big", verifier = "big", summarizer = "tiny", escalate = "big" }

[profiles.local]                  # the default worker: lookups and edits
model = "qwen3-9b"

[profiles.big]
base_url = "https://api.deepseek.com/v1"
api_key_env = "DEEPSEEK_API_KEY"  # read from the environment, not stored in the file
model = "deepseek-chat"

[profiles.tiny]
model = "qwen3-1.7b"
```

| Rol | Qué hace | Ajuste sugerido |
|---|---|---|
| `worker` | Búsquedas y ediciones, el modelo activo | Local; razonamiento desactivado en modelos pequeños |
| `planner` | Convierte una meta en una lista de verificación, y los pasos del plan | Un modelo local más fuerte o alojado, razonamiento activado si cabe |
| `verifier` | Verificación adversarial de una respuesta antes de escribirla | Razonamiento activado desde 16 GB, o un modelo alojado |
| `repair` | Corrige ediciones fallidas y fallos de pruebas | Normalmente el trabajador. El razonamiento puede ayudar con errores de formato |
| `summarizer` | Resúmenes de una línea y compactación | El trabajador, o un modelo local diminuto si mediste la ganancia |
| `escalate` | Un intento más en un modelo más grande cuando una tarea falló: no aterrizó ninguna edición, un fallo semántico de pruebas sobrevivió a la reparación, dos arrendamientos sin progreso, o ejecuciones repetidas sin ediciones | Un modelo local más grande o alojado |
| `fallback` | Toma el control cuando el trabajador falla con un error | Un modelo alojado o más grande |

Cada peldaño de escalamiento se usa como máximo una vez por tarea, y solo cuando está configurado, de modo que
una mala ejecución local no pueda gastar silenciosamente un presupuesto alojado. Usa `--role planner=big` para
enrutar un trabajo en una sola ejecución. `pedacito profiles --recommend` sugiere un
perfil para cada rol, usando puntuaciones medidas, y `--apply-roles` escribe la
elección una vez que la confirmas.

El escalamiento se basa en razones (`auto_escalate`, activado por defecto): un parche que
se aplica pero no funciona también gana el reintento. Las razones son pruebas que siguen
fallando con errores de ejecución/semánticos (un TypeError, una aserción fallida, un
bucle infinito) tras las rondas de reparación, dos arrendamientos sin progreso en una tarea
y `escalation_zero_edit_runs` ejecuciones consecutivas sin ediciones en el perfil activo
(por defecto 3; se cuenta por perfil en `.pedacito/profile_stats.json`,
0 desactiva la estadística). Cada razón emite un evento `escalation_suggested`
incluso cuando no hay un rol `escalate` configurado, así que obsérvalo antes de
gastar en un peldaño alojado: nombra QUÉ forma de fallo sigue repitiéndose. Un
perfil que sigue sugiriendo escalamiento por ejecuciones sin ediciones normalmente necesita un
mejor paquete de prompts o un trabajador más grande, no más reintentos: el peldaño aún
se dispara como máximo una vez por tarea, y un éxito borra la cuenta del perfil. Cuando
no hay un rol `escalate` distinto configurado, una razón disparada compra UN reintento
con el mismo modelo con el estilo de edición invertido (anchor <-> quote, sin definir -> anchor) en lugar de
nada; un evento `strategy_shift` lo marca. Las ejecuciones verdes nunca se reintentan, y
`auto_escalate = false` desactiva el cambio junto con el resto.

Los modelos alojados reciben el texto de la tarea y los extractos de código que la tarea necesita. Pedacito
advierte antes de enviar una tarea que parezca contener un secreto. Limita el gasto con
`max_total_tokens` o `max_cost`, y define `price_input_per_mtok` y el precio de salida
correspondiente para que se cuente el costo.

---

## Dual-GPU / Dual-Model: enrutamiento Scout + Arquitecto (`scout_profile`)

La recolección es el ciclo en el que el modelo busca cosas antes de editar. Sus pasos son
cortos y muy acotados: haz grep de esto, lee ese rango, esquematiza este archivo.
Un modelo pequeño los hace bien y con rapidez. La planificación, la respuesta y las ediciones necesitan
el modelo más grande. `scout_profile` le da cada trabajo al modelo correcto:

- **Scout (4B a 9B):** los pasos de búsqueda de la recolección. Son `read`, `grep`,
  `find`, `outline`, `callers`, `context`, `usages`, `run_test` y las demás acciones
  de investigación. Colócalo en una segunda GPU o en un endpoint ligero.
- **Arquitecto (27B a 35B):** el perfil activo. Se encarga de la planificación, la
  respuesta, cada edición, reparación y verificación.

```toml
[universal]
scout_profile = "scout"             # the lookup model; must name a profile below

[profiles.architect]                # the active profile: the big model
base_url = "http://localhost:1234/v1"
model    = "qwen3-32b"
tier     = "large"
vram_gb  = 24
disable_thinking = true
context_tokens = 32768

[profiles.scout]                    # the second GPU, or a lightweight endpoint
base_url = "http://192.168.1.50:11434/v1"
model    = "qwen3-4b"
tier     = "tiny"
vram_gb  = 8
disable_thinking = true
```

Define el `vram_gb` del perfil con la memoria de la máquina que sirve ese modelo.
Cada perfil se dimensiona por separado, así que los 8 GB del scout no limitan al arquitecto.

Qué esperar:

- **El arquitecto y el scout se usan en la misma tarea.** La respuesta ve el mismo
  material recolectado sin importar qué modelo hizo las búsquedas, así que el enrutamiento no agrega
  una segunda opinión a la respuesta. El uso de ambos modelos aparece en los mismos totales.
- **Un scout ausente o inalcanzable no es un error.** Pedacito advierte una vez y ejecuta
  las búsquedas en el arquitecto. Un scout definido con el mismo perfil que el arquitecto se
  ignora.
- **La GUI muestra qué modelo es el scout.** Mientras está activo, la barra superior muestra
  una insignia `Scout: <profile>`. En **⚙ Settings → All settings**, `scout_profile` es un
  menú desplegable de tus perfiles, con **None (use active profile)** para desactivarlo.
  **⚙ Settings → Profiles → Dual-GPU / Scout + Architect Pairing** escribe el par
  en una sola edición que conserva los comentarios (y puede crear un perfil de scout inicial para el
  segundo servidor), y la respuesta dual-GPU de **Auto-Tune** ofrece
  **Create starter Scout & pair…**.
- **Los scouts alojados envían tu código.** Un scout alojado recibe los resultados de las búsquedas,
  que son extractos de tu código. Se aplican las mismas advertencias que a los roles
  alojados.

Verifica el scout antes de depender de él. Su trabajo es elegir la búsqueda correcta, así que ejecuta
`pedacito calibrate --profile scout` (la elección de herramientas es la sonda que más importa)
y compara `pedacito eval` antes y después de agregarlo.

**En una sola GPU**, el beneficio es menor. Ambos modelos deben caber juntos en la VRAM, con
espacio para la caché KV del arquitecto. Si no es así, cada cambio recarga un modelo,
lo que es más lento que ejecutar las búsquedas en el arquitecto. El enrutamiento rinde más
cuando el scout corre en su propia tarjeta, o cuando el scout permanece cargado junto al
arquitecto. El costo es un segundo servidor, un ping de calentamiento antes de la primera búsqueda y
un ajuste más que mantener sincronizado con tus perfiles.

---

## Enrutamiento automático por especialista (`auto_route_specialties`)

Mientras que el scout divide una tarea por *fase* (búsquedas vs. criterio), el enrutamiento por
especialista divide una ejecución por *tarea*: cada tarea va al perfil cuyas etiquetas de
especialidad le encajan mejor. Un modelo de código toma las tareas de implementación, un modelo de visión
toma las tareas de SVG/diseño, un modelo de razonamiento toma las tareas de algoritmos. Actívalo
con `auto_route_specialties = true`, o `--auto-route` en `ask`, `auto` y
`new` (`--no-auto-route` lo fuerza a apagado para una ejecución). La GUI tiene una casilla **✦ Auto-route**
en la barra superior.

El enrutamiento necesita al menos dos perfiles. Cada perfil lleva etiquetas de especialidad, de
tres fuentes, en este orden de precedencia:

1. **Explícita:** `specialties = ["coding", "testing_qa"]` en el cuerpo del perfil
   siempre gana. Las etiquetas son slugs en minúsculas; además del conjunto canónico
   (`general`, `coding`, `frontend_ui`, `3d_graphics_svg`, `math_algorithms`,
   `testing_qa`, `sql_data`, `translation_i18n`) puedes usar etiquetas personalizadas
   (`[a-z0-9_-]+`).
2. **Medida:** las etiquetas de especialidad de la tarjeta del modelo, de la calibración (abajo).
3. **Heurísticas de nombre:** sin tarjeta, el nombre del modelo hace de sustituto: `coder`,
   `codestral`, `starcoder`, `devstral` implican coding + testing; `math`, `qwq`,
   `r1`, `reasoning` implican matemáticas/algoritmos; `-vl`, `vision`, `pixtral`, `llava`,
   `gemma-3` implican frontend + gráficos. Todo lo demás es `general`.

`pedacito profiles` muestra las etiquetas efectivas en su columna **Specialties**, y
en la GUI, **Settings → Profiles** te permite activar las etiquetas canónicas, agregar una
personalizada y ver lo que la tarjeta detectó para cada perfil.

Las etiquetas medidas dependen de la sonda del formato de edición: cada especialidad de edición
(`coding`, `frontend_ui`, `3d_graphics_svg`, `math_algorithms`, `testing_qa`,
`sql_data`) necesita `edit_format >= 0.65` y `tool_choice >= 0.5` en la tarjeta,
sin importar qué tan bien responda el modelo a una sonda de un solo tiro: un modelo que no puede mantener el
formato de edición no es un especialista en edición. `translation_i18n` es trabajo en prosa y
no tiene esa condición. Los nombres que dicen OCR, router, orchestrator o embedder nunca ganan
etiquetas de edición.

Cómo se asocia una tarea: un prefijo `[specialty: TAG]` en el título de la tarea fuerza el
etiqueta (se le indica al planificador que la use). En caso contrario, el título se puntúa según palabras clave
("refactor", "svg", "translate", …) y según las extensiones de archivo que nombra
(`.css` → frontend, `.sql` → datos, una forma `test_*.py` → testing). El perfil
con una etiqueta ganadora toma la tarea; en caso de empate se prefiere un modelo ya cargado, luego un
perfil en un servidor distinto y luego la puntuación de tarjeta más alta. Una tarea sin
coincidencia positiva, y cualquier tarea cuya etiqueta ganadora el perfil activo ya
tenga, se queda donde está. `general` nunca gana una coincidencia.

Qué esperar:

- **Los cambios ocurren solo en los límites de las tareas**, nunca dentro de una
  conversación de recolección. La transcripción de recolección conserva su caché de prefijo en el
  servidor que la posee (TECHNIQUES.es.md §2 y §8); el enrutamiento no puede hacer que una tarea pague
  un re-prefill a mitad de conversación.
- **Un servidor local: una transferencia limpia de VRAM.** Cuando el especialista comparte el
  `base_url` local del trabajador, Pedacito descarga primero el modelo saliente (Ollama
  recibe `keep_alive: 0`; LM Studio una descarga de mejor esfuerzo) y luego calienta el nuevo,
  para que los dos modelos nunca peleen por la misma VRAM. El registro de Activity y el
  flujo de eventos registran un `model_swap` con lo que se descargó, lo que se cargó y
  cuánto tardó. Cuando la siguiente tarea vuelve al trabajador, este se
  recalienta de la misma manera.
- **Servidores distintos: ambos se mantienen calientes.** Los perfiles en `base_url` separados (una
  segunda GPU, o un endpoint alojado) no necesitan transferencia; cada uno conserva su propia caché.
- **Un especialista inalcanzable no es un error.** El enrutador advierte una vez y ejecuta
  la tarea en el trabajador, igual que con el scout y los demás roles.
- **Un solo registro de uso.** Los tokens y el costo del especialista, incluido el calentamiento,
  se suman a los mismos totales que los del trabajador. La barra superior y el panel de Activity de la GUI
  muestran qué especialista está activo.
- **Los especialistas alojados envían tu código.** Se aplican las mismas advertencias que a los
  roles alojados.

El planificador ve la lista de especialistas cuando el enrutamiento está activo, y se le pide que anteponga
`[specialty: TAG]` a las tareas y que agrupe las tareas de la misma especialidad cuando
las dependencias lo permitan: cada cambio en un solo servidor cuesta una descarga/carga, así que
menos cambios es una ejecución más rápida.

En una sola GPU, mide antes de depender de ello: cada cambio paga una recarga de modelo, así que
el enrutamiento rinde cuando las tareas se agrupan por especialidad (un lote de tareas de UI y luego un
lote de tareas de pruebas), no cuando se alternan. Con un segundo servidor los cambios
son gratuitos y la división es la natural. Calibra cada especialista
(`pedacito calibrate --profile <name>`) para que sus etiquetas provengan de mediciones, y
compara `pedacito eval` antes y después de activar el enrutamiento.

---

## JSON impuesto por el motor según el servidor (`enforce_json_schema`)

Cada paso de recolección es un objeto JSON con un conjunto fijo de campos. Con
`enforce_json_schema = true` (el valor por defecto), Pedacito envía el esquema del paso al
servidor como `response_format`, de modo que el servidor restringe el muestreo a JSON válido para
ese esquema. Un paso que no coincide con el esquema no puede producirse, así que la mayoría de
los reintentos que antes reparaban respuestas mal formadas ya no son necesarios.

Pedacito desciende por tres niveles, una petición a la vez:

| Nivel | Petición enviada | Qué garantiza el servidor |
|---|---|---|
| `schema` | `json_schema`, `strict: true` | JSON válido para el esquema del paso, impuesto durante el muestreo |
| `object` | `json_object` | JSON válido, pero de cualquier forma. El prompt lista los campos |
| `plain` | nada extra | Nada por parte del servidor. El prompt y los ejemplos few-shot mantienen el formato |

Si un servidor responde HTTP 400 y el mensaje nombra `response_format`,
`json_schema` o `json_object`, ese nivel se desactiva para este `base_url` y
modelo, y la misma petición se envía de inmediato un nivel más abajo. La primera llamada
paga por el rechazo, y las siguientes omiten el nivel rechazado. El resultado se
recuerda durante la sesión. `pedacito config --doctor` sondea los modos de antemano y
guarda la respuesta en `server_modes.json`, de modo que un servidor que rechaza un nivel se maneja
antes de la primera tarea.

Esto es lo que se espera por servidor. Las funciones de los servidores cambian entre versiones, así que
tómalo como punto de partida y confirma con `pedacito config --doctor`.

| Servidor | Salida estructurada | Qué esperar |
|---|---|---|
| **llama.cpp** (`llama-server`) | `json_schema` se convierte en una gramática | El nivel `schema` en las versiones actuales. Una versión anterior que rechaza `json_schema` cae a `object` |
| **Lemonade Server** (`lemond`) | Su backend llama.cpp acepta `response_format` | El nivel `schema` cuando el backend cargado lo acepta. `--doctor` muestra qué nivel usa |
| **LM Studio** | Salida estructurada con `json_schema` | El nivel `schema` en versiones recientes. Las versiones anteriores caen a `object` |
| **Ollama** (`/v1`) | `response_format` con `json_schema` | El nivel `schema` en versiones recientes. Las versiones anteriores caen a `object` |
| **vLLM** | `response_format` con `json_schema` (decodificación guiada) | El nivel `schema` cuando el backend de salida estructurada está disponible en tu compilación |
| **APIs alojadas** (OpenAI, DeepSeek y otras) | Pedacito define por defecto `json_mode = "object"` para hosts remotos, lo que le sirve a DeepSeek y a la mayoría | El nivel `object` por defecto. Define `json_mode = "schema"` en un perfil para una API que acepte `json_schema`, como OpenAI, para usar el nivel `schema` |

Dos cosas que debes saber:

- **Las llamadas nativas a herramientas lo omiten.** Con `tool_mode = "native"`, el paso se envía
  por el canal de llamadas a herramientas del modelo, y `response_format` no se usa. El
  ajuste sigue aplicando a todas las demás llamadas JSON, como la planificación.
- **Desactívalo solo para un servidor que se porta mal.** `enforce_json_schema = false` envía
  solo prompts simples. Úsalo cuando un servidor produce salida rota con muestreo
  restringido. De lo contrario, mantenlo activado, ya que es la forma más barata de evitar pasos
  mal formados.

---

## Aceleración de inferencia local (`enable_prompt_cache`)

La recolección es un ciclo de llamadas cortas que reenvían una conversación creciente y casi sin cambios.
Los runtimes autoalojados pueden abaratarlo con dos claves de petición,
y Pedacito las agrega por ti:

- `cache_prompt: true` le pide al servidor que conserve el prefijo KV del inicio idéntico
  del prompt entre llamadas, de modo que el paso 12 no vuelva a hacer prefill de los pasos 1 al 11.
  Activado por defecto (`enable_prompt_cache = true`).
- `prompt_lookup_num_tokens: N` activa la decodificación por búsqueda de n-gramas en el prompt: el
  borrador copia continuaciones del propio prompt. Encaja muy bien con
  esta carga porque la fase de edición cita líneas de código fuente que ya están
  en el contexto. Desactivado por defecto (`enable_prompt_lookup = false`);
  `prompt_lookup_num_tokens` define la extensión del borrador (por defecto 10).

Qué esperar por runtime (confirma con `pedacito config --doctor`):

| Servidor | `cache_prompt` | Búsqueda en el prompt |
|---|---|---|
| **llama.cpp** (`llama-server`) | Aceptado; se combina con `--cache-reuse` en el servidor para mejores resultados | Aceptado; activa opciones estilo `--lookup-ngram-draft` en el servidor, o déjalo y se aprende el rechazo |
| **Lemonade Server** (`lemond`) | Su backend llama.cpp acepta la clave | Misma superficie que llama.cpp |
| **LM Studio** | Aceptado en versiones recientes | El soporte varía según la versión |
| **Ollama** (`/v1`) | Aceptado | No es compatible en versiones antiguas; lo cubre el mecanismo de respaldo de abajo |
| **vLLM** | Aceptado (`--enable-prefix-caching` está activado por defecto en versiones recientes) | No es el mismo mecanismo; espera el respaldo |
| **APIs alojadas** | Nunca se envía | Nunca se envía |

Importan dos propiedades de seguridad. Las claves se adjuntan solo a endpoints que son
realmente tuyos: loopback, una dirección LAN (RFC1918, CGNAT, unique-local de IPv6), un
nombre `.local` o una entrada de proveedor que nombra un runtime local. Una API alojada
nunca las ve. Y un servidor que no entiende una clave no es una
configuración rota: cuando una petición respondida con HTTP 400 o 422 aún tenía alguna de las claves
adjuntas, Pedacito las quita, reenvía de inmediato esa petición
y recuerda el rechazo para este servidor y modelo, de modo que las llamadas posteriores empiecen limpias.
El costo de una clave no admitida es una petición extra por sesión, nunca una
tarea fallida. Una clave que escribas tú mismo en la tabla `[sampler]` de un perfil siempre
gana; la aceleración nunca sobrescribe el ajuste propio del usuario.

Regla práctica: mantén `enable_prompt_cache = true` en cualquier servidor local. Activa
`enable_prompt_lookup = true` cuando las ediciones dominan la ejecución (bloques SEARCH,
refactorizaciones) y la GPU está limitada por cómputo; mide con `pedacito calibrate` o
una sesión de `chat` lado a lado si tienes dudas.

---

## Fijación de modelos en Lemonade Server

El modo multimodelo de Lemonade expulsa el modelo menos usado recientemente cuando necesita
un espacio. Un modelo fijado se omite, lo que mantiene residente un modelo de uso frecuente
mientras otros modelos van y vienen.

- En la GUI, abre **Servers**, selecciona el modelo residente en una fila de Lemonade y
  haz clic en **Pin** o **Unpin**.
- En la TUI, usa `/servers pin SERVER [MODEL]` o
  `/servers unpin SERVER [MODEL]`. El id del modelo es opcional cuando el servidor
  tiene un solo modelo residente.
- Pedacito llama a `POST /internal/pin` de Lemonade con
  `{"model_name": "...", "pinned": true|false}`; una compilación que responde solo a la
  grafía anterior recibe `/api/v1/internal/pin` como respaldo. El modelo debe estar
  ya cargado.

La fijación es exclusiva de Lemonade. llama.cpp, Ollama y los servidores locales genéricos no tienen
equivalente, así que Pedacito trata la acción como una operación nula allí.

---

## Modelos de imagen (`modality = "image"`)

Algunos servidores también sirven **generadores de imágenes**. Lemonade Server expone modelos Stable Diffusion y Flux mediante
sus recetas `sd-cpp`, y cualquier servidor compatible con OpenAI que tenga `/images/generations` funciona igual.
Pedacito los detecta por los metadatos `labels`/`recipe` del servidor o por el id del modelo, marca el perfil con
`modality = "image"` y le pone la insignia **🎨 Image**. Los perfiles de imagen no son modelos de chat: sirven a
`pedacito image`, `/image`, el diálogo de la GUI y la acción opcional `generate_image`
del agente (consulta [`image_profile` y `allow_image_gen`](docs/es/CONFIG.md#generación-local-de-imágenes-image_profile-allow_image_gen)).

Calibrar un perfil de imagen (`pedacito calibrate --profile NAME`, `models scan` o **Probe image** en la GUI)
no ejecuta las sondas de texto. Hace **una** petición rápida de imagen (512×512, 4 pasos) en una carpeta temporal y
registra `image_ok`, el formato, las dimensiones y la latencia en la tarjeta del modelo, que es lo que muestra la
insignia **🎨 Image · ✓ Verified**.

---

## Razonamiento por perfil

Algunos modelos razonan antes de responder, en un bloque de razonamiento separado. Eso ayuda
en problemas difíciles. En un modelo pequeño también puede consumir el presupuesto de respuesta, de modo que la
edición nunca llega.

### Actívalo o desactívalo para un perfil

En la GUI, abre **Settings → Profiles**, selecciona el perfil y define
`disable_thinking` bajo **Reasoning & Execution Overrides**. La etiqueta de la barra
superior muestra **no-think** o **thinking** para el perfil activo.

En el archivo de configuración:

```toml
[profiles.worker]
model = "qwen3-14b"
disable_thinking = true      # no reasoning for answers, edits, repair, or verifier calls

[profiles.think]
model = "qwen3-14b"
disable_thinking = false
max_tokens_answer = 8000     # raise the budget when thinking is on
```

Pedacito ya desactiva el razonamiento para su ciclo de búsqueda, la planificación y las llamadas
de andamiaje, porque son llamadas JSON donde el razonamiento solo agrega costo. El ajuste
controla las llamadas de respuesta, reparación, verificador y resumen, donde el razonamiento
compite con el presupuesto de salida.

Si un modelo tiene un interruptor en el prompt en lugar de uno en la API, define `thinking_suffix`,
como `/no_think` para Qwen3. Déjalo vacío cuando el servidor ya respeta
`disable_thinking`.

### Esfuerzo de razonamiento

`reasoning_effort` es un ajuste de perfil (también `/effort` en la TUI y el
selector **Effort** en la GUI) con cuatro valores: `default`, `low`, `medium`
y `high`. `default` es la grafía incluida del `off` del plan: no envía ningún
campo `reasoning_effort`. Los demás valores se reenvían tal cual a los
servidores local-MoE y a los modelos alojados con capacidad de razonamiento, nunca a modelos que no
exponen el control. El ajuste nunca cambia el muestreo, y
`disable_thinking = true` gana sobre cualquier nivel de esfuerzo.

### Cuándo activar el razonamiento

- Depuración difícil, rastreo entre archivos, preguntas de arquitectura o una tarea con
  varios requisitos que interactúan.
- Un planificador que debe dividir una meta amplia en tareas que no se solapen.
- Un verificador que debe detectar un error lógico sutil.

Aumenta `max_tokens_answer` en unos 2,000 a 4,000 cuando el razonamiento está activado. Sin
el espacio extra, la mayoría de las ejecuciones terminan en truncamientos y reintentos.

### Cuándo desactivarlo

- Búsquedas rutinarias y ediciones pequeñas y localizadas.
- Ventanas de 4k a 8k, donde el razonamiento compite con el mapa y el código fuente.
- Modelos que razonan extensamente pero escriben ediciones débiles. El presupuesto de respuesta se agota
  antes del bloque de búsqueda/reemplazo.
- Cualquier configuración donde los registros muestren `finish_reason = "length"` o errores repetidos de "would not
  produce valid JSON".

`pedacito calibrate` ejecuta una prueba A/B de razonamiento. Si un modelo solo funciona con
el razonamiento desactivado, la tarjeta del modelo registra `disable_thinking = true`, de modo que las
ejecuciones posteriores comiencen en ese modo.

### Modelos que siempre razonan

Algunos modelos no tienen interruptor de apagado: destilados estilo R1, LFM2.5-Thinking,
LFM2.5-24B-A2B, GPT-OSS, y Kimi K2.7 y K3. Pedacito elimina sus bloques de razonamiento
de la salida, pero no puede eliminar el costo en tokens. Para un trabajador pequeño,
prefiere una variante instruct o sin razonamiento de la misma familia cuando exista.

### Notas por familia

Son ejemplos. Revisa la tarjeta del modelo y la plantilla de chat para el interruptor exacto.

| Familia | Qué esperar |
|---|---|
| **Qwen3 / 3.5** | Razonamiento híbrido, que se cambia con `enable_thinking` o `/no_think`. Excelente uso de herramientas. Un buen trabajador con el razonamiento desactivado y un buen planificador con él activado. |
| **Gemma 3 / 4** | Gemma 3 no tiene razonamiento. Gemma 4 tiene el razonamiento desactivado por defecto y es opcional. Buen seguimiento de instrucciones. |
| **Nanbeige 4.x** | El razonamiento está activado por defecto, y la salida incluye bloques `<think>`. Fuerte para su tamaño, pero con alto consumo de tokens. Desactívalo para el trabajador. |
| **LiquidAI LFM2.5 / LFM3** | Muy pequeños y rápidos. LFM2.5-Thinking y LFM2.5-24B-A2B siempre razonan. LFM3 admite `enable_thinking`. Prefiere una variante Instruct para la edición rutinaria. |
| **DeepSeek R1 y V3.x** | Los modelos estilo R1 siempre piensan y pueden ser lentos. V3.1 y los híbridos posteriores exponen un interruptor `thinking`. Presupuéstalo, o usa un modelo de código sin razonamiento como trabajador. |
| **GLM 4.x / 5** | `enable_thinking` híbrido y excelente uso de herramientas. Las variantes grandes necesitan bastante VRAM. |
| **GPT-OSS** | Siempre razona. `reasoning_effort` reduce el costo pero no lo apaga. Buen uso de herramientas. El tamaño de 20B es la opción local realista. |
| **Mistral / Magistral** | Los modelos estilo Magistral usan su propio formato de razonamiento y a menudo siempre razonan. Mistral Small 4 tiene un interruptor de esfuerzo. |
| **Granite** | 3.2 y 4.2 tienen interruptores de razonamiento. Pequeños y orientados a instrucciones. Revisa la versión. |
| **Phi-4 reasoning** | El razonamiento se escribe en etiquetas estilo `<think>`, impuestas por el prompt. No hay un interruptor de apagado estándar. Puede ser lento con razonamientos largos. |

---

## Calibración: `pedacito calibrate`

La calibración mide un modelo en las cosas de las que depende Pedacito y luego guarda el
resultado como una **tarjeta de modelo**. La tarjeta ajusta los valores por defecto de Pedacito para ese modelo. No
reemplaza tus propias mediciones.

```bash
pedacito calibrate --profile worker          # full run, about a few minutes
pedacito calibrate --quick --profile worker  # skip the long-context and capability probes
pedacito calibrate --repeats 5               # run each probe five times
pedacito calibrate --recalibrate             # measure again even with a fresh card
pedacito calibrate --all -y                  # every local profile in turn, no confirmation
```

La calibración es **solo para modelos locales**. Un perfil de proveedor alojado se rechaza
(la CLI sale con 2, el botón de la GUI está deshabilitado) porque los modelos alojados usan
automáticamente los valores por defecto de la tabla del proveedor; `models scan` tampoco calibra nunca las
importaciones alojadas. `--all` recorre los perfiles locales en secuencia, descargando cada
modelo antes de que cargue el siguiente, y omite los perfiles con una tarjeta vigente a menos que se indique
`--recalibrate`.

La calibración ejecuta un conjunto programado de sondas:

- **Elección de herramientas:** ¿el modelo elige la acción de búsqueda correcta?
- **Adherencia a JSON:** ¿responde con la estructura requerida?
- **Formato de edición:** ¿escribe bloques de búsqueda/reemplazo válidos?
- **Inyección y honestidad:** ¿ignora una instrucción plantada en el código fuente
  y admite lo que no ha leído?
- **Recuperación en contexto largo:** ¿puede encontrar una aguja en un prompt largo?
- **Continuación:** ¿puede responder a partir de una conversación caliente?
- **A/B de razonamiento:** ¿el razonamiento oculto consume el presupuesto de respuesta? Cada pregunta se
  responde con el razonamiento activado y desactivado.
- **Llamadas nativas a herramientas:** los mismos escenarios por el canal de llamadas a herramientas del modelo,
  comparados con la gramática JSON.
- **Microsondas de especialidad:** seis verificaciones diminutas, ejecutadas una vez con el razonamiento desactivado:
  emitir un SVG bien formado con atributos exactos (frontend/gráficos), calcular una
  expresión entera exacta (matemáticas/algoritmos), traducir una oración al
  español sin tocar sus placeholders `{name}`/`{count}` (i18n), escribir un
  join de SQL que se califica ejecutándolo contra una base de datos SQLite de prueba
  (sql/datos), responder qué lanza `avg([])` como el objeto JSON estricto
  `{"empty_raises": "ZeroDivisionError"}` (testing/QA) y ordenar un plan de tres pasos
  parser → CLI → pruebas (planificación; puntúa al orquestador, no otorga etiqueta).
  Una puntuación de 0.67 o mejor otorga las etiquetas correspondientes; puntuaciones altas de formato de edición y
  elección de herramientas (ambas ≥ 0.75) otorgan coding + testing, y un formato de edición
  débil (< 0.5) quita coding. Junto con las heurísticas de nombre y las etiquetas recomendadas de la
  biblioteca de arquitecturas, se convierten en las `specialties` de la tarjeta,
  que usa el enrutamiento automático por especialista (arriba).

Cada sonda se ejecuta tres veces por defecto, o una vez con `--quick` (las microsondas de
especialidad siempre se ejecutan una vez, para mantenerse en pocos segundos). La tarjeta registra la
media y la dispersión, de modo que las decisiones provengan de un promedio, no de una respuesta afortunada.

La tarjeta define estos valores:

- `prompt_pack`, que controla cuánto texto de instrucciones recibe el modelo
- `answer_mode`, si la respuesta continúa la conversación de búsqueda
- `verify_answer` y `best_of`, que agregan verificaciones y muestras extra para modelos más débiles
- `tool_mode`, `grammar` o `native`
- `fuzzy_match` y `edit_style`
- `disable_thinking`, cuando el A/B de razonamiento muestra que el modelo lo necesita desactivado
- `specialties`, las etiquetas con las que el enrutamiento automático por especialista asocia tareas
- para modelos locales, los **valores por defecto de arquitectura**: los
  `temperature`, `temperature_answer`, sampler (`top_p`, `top_k`,
  `repeat_penalty`) y `context_tokens` recomendados por la biblioteca integrada para la familia del modelo (Qwen coder,
  Qwen3, DeepSeek coder/R1, Nemotron, Gemma, Mistral, Granite, Phi-4,
  Llama-3). La ventana está limitada por la longitud de contexto que informa el servidor y
  tu VRAM, y una aguja de contexto largo no encontrada la reduce (16384 y luego
  8192) antes de que la tarjeta la registre.

Solo los ajustes que dejaste en sus valores por defecto siguen a la tarjeta. Cualquier cosa que definas
explícitamente sigue ganando. `pedacito config` muestra la tarjeta y si está en uso.

Las tarjetas caducan a los 30 días, y una tarjeta de una versión anterior se muestra con una nota
y no se aplica. Vuelve a calibrar después de cambiar el modelo, su cuantización o
su plantilla de chat. En la GUI, cada perfil en **Settings → Profiles** tiene un botón
**Calibrate** que ejecuta una medición rápida y transmite la salida al
panel de terminal, además de una insignia: ✓ Calibrated (con la fecha y la puntuación),
⚠ Calibration stale, ○ Not calibrated yet o ☁ Hosted API (botón deshabilitado).
**Project ▾ → Calibrate All Local Models…** ejecuta todo el conjunto local como un solo
trabajo en segundo plano. `pedacito models scan` también calibra cada modelo local nuevo con
una ejecución rápida por defecto, de modo que obtienes una tarjeta al agregar modelos.

---

## Compara modelos con los benchmarks de diagnóstico

Pedacito incluye seis proyectos de diagnóstico pequeños. Para cada uno, un modelo construye el
proyecto con el ciclo normal de Pedacito, y el resultado se califica de dos maneras:

1. **Las pruebas del propio modelo** (`pytest -q`, o el script de verificación del proyecto).
2. **Una rúbrica oculta de evaluación**, escrita por Pedacito, que usa solo la API
   pública que describe la tarea. Verifica las cosas en las que los modelos pequeños suelen equivocarse.

Los seis proyectos contienen 71 verificaciones de evaluación en total.

| Benchmark | Qué pide | Qué detecta |
|---|---|---|
| **Stateful CLI Expression & RPN Calculator** (`rpn_expression_calc`, 12 verificaciones) | Una calculadora infija con variables y un modo RPN, como módulo y como CLI | Precedencia de operadores, `**` asociativo por la derecha, menos unario, variables entre argumentos, códigos de salida |
| **Markdown Table & Box-Drawing Formatter** (`markdown_table_fmt`, 12 verificaciones) | Analizar tablas Markdown y CSV, y escribir salida Markdown o de dibujo de cajas alineada | Relleno desfasado por uno, pipes escapados, caracteres CJK anchos, marcadores de alineación, truncamiento |
| **Thread-Safe TTL + LRU Cache** (`ttl_lru_cache`, 13 verificaciones) | Una caché de dos archivos con TTL, expulsión LRU, un decorador memoize y estadísticas | Orden de expulsión, elementos vencidos expulsados antes que los vigentes, `functools.wraps`, importaciones entre archivos, seguridad entre hilos |
| **Streaming Log Parser & JSON Report** (`log_anomaly_analyzer`, 11 verificaciones) | Analizar registros mixtos, calcular percentiles por servicio, marcar picos de 5xx y emitir JSON estricto | Precisión de regex, percentiles por rango más cercano, ventanas móviles de 60 segundos, disciplina de JSON |
| **Single-File Interactive Kanban Board** (`mini_kanban_html`, 13 verificaciones) | Un tablero de HTML, CSS y JavaScript sin dependencias, con filtros, persistencia e importación/exportación | Ids del DOM que coinciden entre HTML y JS, conexión de eventos, localStorage, diseño CSS, entrega sin conexión |
| **Pelican Riding a Bicycle (Parametric SVG & Geometry)** (`pelican_bicycle_svg`, 10 verificaciones) | Un script de Python que escribe un dibujo SVG etiquetado de un pelícano en una bicicleta, una página de vista previa para el navegador y pruebas de geometría | Geometría de coordenadas 2D, los siete grupos semánticos requeridos, XML bien formado con `viewBox="0 0 800 600"`, dos ruedas sobre una misma línea de suelo, un pico grande, un cuerpo sobre el eje y un mínimo de detalle visual |

El benchmark del pelícano es el espacial. Sus verificaciones de ruedas y cuerpo leen las
coordenadas que escribió el modelo, así que un dibujo hecho de dos círculos vacíos falla. Sus
verificaciones visuales también necesitan al menos 15 primitivas gráficas y cuatro colores.

### Ejecuta un benchmark

Desde la GUI:

1. Haz clic en **▦ Benchmarks** en la barra superior de la GUI.
2. Elige el perfil a probar. Cada modelo que quieras comparar necesita su propio
   perfil, así que crea uno por modelo primero.
3. Opcionalmente elige la carpeta para la salida. Por defecto es una carpeta `benchmarks/`
   junto al proyecto que tienes abierto.
4. Haz clic en **Run Benchmark**. El diálogo muestra el progreso y luego el resultado.

Desde la CLI:

```bash
pedacito bench --list                      # the six ids and their checks
pedacito bench rpn_expression_calc         # one benchmark
pedacito bench all --json --strict         # all six, CI-friendly
pedacito bench --recent                    # earlier runs, no model needed
```

`--dir DIR` elige la raíz de salida. El progreso va a stderr, la tarjeta de puntuación a
stdout, y `--strict` sale con 1 a menos que pasen tanto las pruebas del modelo como la rúbrica
de evaluación.

Cada ejecución crea una carpeta nueva con un nombre como `pedacito_bench_<benchmark>_<profile>_<time>`.
No se elimina nada, así que puedes abrir el proyecto, ejecutar sus pruebas tú mismo y
leer el código.

### Lee el informe

Cada ejecución escribe `BENCHMARK_REPORT.md` en su carpeta, junto con `benchmark_run.json`.
El informe tiene estas secciones:

- **Run:** el benchmark, perfil, modelo, hora de inicio, duración y si
  la generación terminó limpiamente.
- **Score:** un PASS o FAIL general, con los resultados de las pruebas del modelo y los
  resultados de la evaluación. El resultado general es PASS solo cuando ambos pasan.
- **Usage:** tokens y llamadas al modelo de la ejecución.
- **Shortcomings detected:** los fallos específicos que encontró el arnés.
- **SVG preview:** cuando el proyecto contiene un SVG (el `pelican_bicycle.svg` del
  benchmark del pelícano), la tarjeta de puntuación de la GUI muestra el dibujo en línea. Se
  renderiza como una imagen en sandbox, de modo que nada en el archivo puede ejecutarse.
- **Harness holdout rubric:** cada verificación, su descripción y si pasó o falló.
- **Requirement ledger:** cada cláusula de la tarea y el estado que alcanzó.
- **Model test output** y **Holdout rubric output:** la salida completa de pytest.
- **Run it yourself** y **Generated files:** los comandos para volver a ejecutar el proyecto.

### Compara dos modelos de forma justa

1. Usa el mismo hardware y los mismos ajustes de Pedacito. Solo debe cambiar el modelo.
2. Ejecuta el mismo benchmark con el perfil de cada modelo. Para una comparación más sólida,
   ejecuta cada benchmark más de una vez, porque una sola ejecución es ruidosa.
3. Compara las verificaciones de evaluación aprobadas sobre el total, luego el PASS o
   FAIL general y después las pruebas del propio modelo.
4. Mira el tiempo y los tokens en **Usage**. Un modelo que aprueba al doble de costo
   puede seguir siendo la elección equivocada para tu hardware.
5. Lee la sección **Shortcomings**. Un modelo que aprueba la rúbrica pero deja un
   marcador de posición o un caso límite sin probar te dice algo que la puntuación no.

Los benchmarks necesitan un modelo que pueda construir un proyecto pequeño a partir de una especificación en una sola ejecución,
así que dicen más sobre todo el ciclo que sobre el conocimiento bruto de un modelo. Úsalos
para comparar modelos candidatos en el mismo trabajo, no como una tabla de clasificación.

---

## Cómo medir

### 1. Verifica la configuración

```bash
pedacito config --check --profile worker
pedacito config --doctor --profile worker
```

`--check` verifica el servidor y el modelo, e informa la ventana de contexto real.
`--doctor` sondea los modos de salida estructurada, la tasa del tokenizador y los detalles del servidor.
Si el servidor informa una ventana desactualizada, define `context_tokens` tú mismo.

### 2. Calibra el modelo

```bash
pedacito calibrate --profile worker
```

Consulta [Calibración](#calibración-pedacito-calibrate) arriba.

### 3. Ejecuta la evaluación y luego haz un A/B

`pedacito eval` ejecuta un conjunto fijo de tareas sobre repositorios de prueba pequeños (una
tienda en Python, una aplicación TypeScript, una CLI en Go y un servicio Python más grande). Algunas tareas
plantan una inyección de prompt en el código fuente, así que la evaluación también verifica si el modelo
sigue instrucciones que encuentra en el código que lee.

```bash
pedacito eval --profile worker --out worker-off.json
pedacito eval --profile worker-think --out worker-on.json \
  --baseline worker-off.json
```

Si la segunda ejecución retrocede, el comando sale con un código distinto de cero y lista lo que empeoró:

- tasa de aprobación, tasa de localización y tasa de primera aplicación limpia
- inyecciones de prompt que se siguieron
- tokens de prompt y de completado, llamadas y tiempo de reloj
- nuevos tipos de fallo

Ajusta la compuerta con `--max-token-growth` y `--max-time-growth` si tu hardware
tiene un perfil de costo conocido. Usa `--repeats 3 --seed 1` para ver la dispersión entre
ejecuciones.

La tasa de aprobación por sí sola no basta. Prefiere una configuración que también mantenga o mejore:

- **first-apply-clean** (menos llamadas de reparación)
- media de tokens de prompt y de completado
- proporción de aciertos de caché
- tiempo de reloj
- la tasa de truncamientos y fallos de formato

### 4. Pruébalo con trabajo real

La evaluación usa pruebas pequeñas. Antes de confiar en un modelo o perfil nuevo, pruébalo con
tu propio código:

```bash
pedacito ask PROJ -t "a real, narrow task"        # dry run
pedacito ask PROJ -t "..." --review               # inspect the diff
pedacito ask PROJ -t "..." --apply                # once you are confident
```

`--record` en una ejecución, y `--replay` después, prueban cambios en el propio
arnés de Pedacito sin un modelo. No miden una actualización de modelo. En CI, agrega
`--strict-replay` para que la compilación falle cuando el arnés haga un prompt distinto
al de la grabación.

### 5. Vigila la VRAM

Usa tu herramienta de GPU (`nvidia-smi`, `rocm-smi` o el panel del propio servidor) mientras corre una
tarea. Si el modelo se desborda a la RAM del sistema, la generación se vuelve mucho más lenta, y la
ejecución parece un problema de calidad del modelo cuando en realidad es un problema de dimensionamiento. Reduce
`context_tokens`, usa un modelo o cuantización más pequeños, o cuantiza la caché KV.

---

## Dimensionamiento y cuantización

El `auto_fit` de Pedacito ajusta sus presupuestos de tokens a la ventana real del modelo usando:

- `vram_gb`, la memoria de GPU disponible para el servidor de modelos
- el tamaño del archivo de pesos del modelo, informado por el servidor
- `context_tokens`, `reserve_tokens` y `k_cache_quant` / `v_cache_quant`

Mantiene libres unos 0.5 GB. Los valores que definas explícitamente en un perfil siempre ganan.

`context_tokens` declara la ventana contra la que planifica Pedacito; no
reserva VRAM. La caché KV de esa ventana la asigna el servidor de modelos,
y `auto_fit` solo estima si cabe junto con los pesos (a partir de
`vram_gb` y el tamaño del archivo de pesos): una ventana informada por el servidor o definida por el usuario es
la autoridad, y la estimación advierte cuando es demasiado grande en lugar de
reducir silenciosamente tus presupuestos. Definir un `context_tokens` grande en una GPU pequeña
por lo tanto no compra memoria: compra un desbordamiento a la RAM del sistema (lento) o un
servidor que se niega a cargar. Define la ventana que el servidor realmente ejecuta y deja que
`auto_fit` ajuste los presupuestos a ella.

Un presupuesto aproximado:

```
weights (quantized) + KV cache (grows with context) + 1.5 to 2.5 GB overhead
```

La caché KV es la segunda perilla después de los pesos. Duplicar el contexto duplica aproximadamente
el costo de KV. Si tu runtime cuantiza la caché KV, define
`k_cache_quant` y `v_cache_quant` (por ejemplo `"q8_0"`) para que `auto_fit` no
sobreestime. Si el servidor corre con `--parallel` o un lote grande, deja más
margen.

| Cuantización | Tamaño aproximado | Uso práctico |
|---|---|---|
| Q4_K_M | 0.55 a 0.6 bytes por parámetro | El valor por defecto. La mejor relación entre tamaño y calidad |
| Q5_K_M | Unos 0.7 bytes por parámetro | Cuando el modelo aún cabe con holgura |
| Q6_K | Unos 0.8 bytes por parámetro | Modelos pequeños, o VRAM de sobra. Mejor que Q4 para ediciones |
| Q8_0 | Unos 1.0 byte por parámetro | Solo cuando los pesos y el contexto claramente caben |
| Q3 / IQ3 | Unos 0.4 bytes por parámetro | Último recurso para modelos muy grandes. Evítalo para editar |
| Q2 / IQ2 | Unos 0.3 bytes por parámetro | Evítalo para el trabajo con Pedacito |

Tamaños aproximados de pesos en Q4: un modelo de 7B es de unos 4 a 5 GB, uno de 14B de unos 9 a
10 GB, uno de 24B de unos 14 a 16 GB y uno de 32B de unos 19 a 21 GB. Los tamaños exactos
varían según la arquitectura y el tokenizador. MLX, EXL2, AWQ y GPTQ usan nombres distintos
para la cuantización. Apunta al equivalente de unos 4 bits por peso.

> **Regla práctica:** si tienes dudas, elige un modelo un tamaño más pequeño y dale
> más contexto. Pedacito se beneficia más de un modelo que cabe con una ventana de 16k a 32k
> que de los pesos más grandes que apenas cargan con 4k.

---

## Modos de fallo comunes y soluciones

| Síntoma | Causa probable | Solución |
|---|---|---|
| "Kept running past the token limit", o sin JSON | Razonamiento activado con un presupuesto de respuesta pequeño | Define `disable_thinking = true` para el trabajador, o aumenta `max_tokens_answer` |
| Respuesta vacía con razonamiento oculto | Razonamiento desbocado | Pedacito reintenta con el razonamiento desactivado. Déjalo desactivado por defecto en modelos pequeños |
| "search block not found" | El modelo copió números de línea, o el archivo se desplazó | Acota la tarea. Pedacito vuelve a anclar. Revisa la puntuación de formato de edición del modelo en la calibración |
| Las respuestas son vagas, o nombran el archivo equivocado | El mapa está demasiado comprimido, o el modelo es débil en la elección de herramientas | Aumenta `flat_map_token_budget` o el contexto, o usa un modelo mejor con llamadas a herramientas |
| El modelo ignora las herramientas | La gramática y el modo nativo no coinciden | Ejecuta `pedacito calibrate` y compara `tool_mode = "grammar"` con `"native"` |
| Lento tras unos minutos | El modelo se desbordó de la VRAM, o el contexto es demasiado grande | Reduce el contexto o la cuantización. Vigila la memoria de GPU durante una ejecución |
| Los cambios de rol tardan un minuto | Modelos distintos en una GPU recargan sus pesos | Usa el mismo modelo para los roles, o acepta el costo de carga |
| Buen chat, malas ediciones | El modelo no está orientado a la edición | Prefiere un modelo de código o instruct. Revisa first-apply-clean en `eval` |

---

## Lista de verificación rápida

- [ ] Define `vram_gb` con la memoria realmente disponible para el servidor.
- [ ] Empieza con Q4_K_M y deja de 1.5 a 2.5 GB para la caché KV y la sobrecarga.
- [ ] Con menos de 8 GB, usa un trabajador instruct o sin razonamiento de 4B a 7B, con el razonamiento desactivado.
- [ ] Con 8 a 16 GB, mantén el trabajador sin razonamiento, y prueba el razonamiento solo para el
      planificador o el verificador.
- [ ] Con 24 GB, mide el razonamiento activado y desactivado antes de convertirlo en el valor por defecto.
- [ ] Ejecuta `pedacito calibrate` una vez por modelo y cuantización.
- [ ] Compara los modelos candidatos con los seis benchmarks, en el mismo hardware.
- [ ] Ejecuta `pedacito eval` antes y después de un cambio, y usa `--baseline` como compuerta.
- [ ] Juzga por first-apply-clean y por los números de tokens y tiempo, no solo por la tasa de aprobación.
- [ ] Mantén simple el enrutamiento de roles en una sola GPU. Cada cambio de modelo puede recargar.
- [ ] Haz tus primeras doce tareas reales sin `--apply`, y luego usa `--review`.
- [ ] Ejecuta `pedacito config --doctor` una vez por servidor para confirmar su nivel de JSON.
- [ ] Con dos GPU, define `scout_profile` con un perfil de 4B a 9B, y mantén el modelo de 27B a
      35B como el arquitecto activo.
