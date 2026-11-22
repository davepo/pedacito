🌐 [English](../INSTALLATION.md) | **Español**

# Instalación

Esta guía cubre la instalación de Pedacito (`pedacito` o el alias corto `pdc`) desde PyPI,
la elección de extras opcionales y la conexión de un servidor de inferencia en GPU. Si solo quieres la
versión de 30 segundos, consulta el [README](../../README.es.md#inicio-rápido-en-30-segundos).

Pedacito necesita **Python 3.11 o superior**. No necesita Node.js para el arnés en sí.

## Contenido

- [Instalar desde PyPI](#instalar)
- [Actualización](#actualización)
- [Elegir un modelo para tu GPU](#qué-modelo-se-ajusta-a-tu-gpu)
- [Conectar un servidor de modelos](#configuración-del-servidor-de-modelos)
- [Servidores de inferencia en GPU: ROCm y CUDA](#servidores-de-inferencia-en-gpu-rocm-y-cuda)

## Instalar

Requiere Python 3.11+.

Instalación rápida desde PyPI (0.3.1):

```bash
pip install pedacito                     # Windows: py -m pip install pedacito
pipx install pedacito                    # isolated CLI install, recommended for a tool
pip install "pedacito[browser,vision]"   # browser verification (Playwright) + screenshot analysis (Pillow)
python -m playwright install chromium    # one-time, after the browser extra
```

Con pipx, instala los extras con `pipx install "pedacito[browser,vision]"` y luego ejecuta `pipx run --spec playwright playwright install chromium` para descargar el navegador.

Esto coloca `pedacito` y `pdc` en tu `PATH` e instala el núcleo completo:

- `tree-sitter` con las gramáticas de `tree-sitter-language-pack`, que impulsan la
  fragmentación por AST y la verificación exacta de sintaxis que se ejecuta antes de escribir una edición.
- `pytest` y `ruff`, las herramientas de verificación. Pedacito ejecuta `ruff check` para
  los diagnósticos y las verificaciones posteriores a la edición, y `pytest` para la rúbrica de evaluación
  de los benchmarks y las pruebas que una edición puede romper.
- `requests`, el cliente HTTP para cada servidor de modelos y API alojada.

No necesitas un paso aparte para ninguno de estos. Si no se encuentra el comando
(algo común con instalaciones por usuario en Windows), `python -m pedacito`
(`py -m pedacito`) es idéntico.

`pip install -r requirements.txt` instala el mismo conjunto núcleo sin instalar
Pedacito en sí. Úsalo cuando solo quieras las dependencias.

Funciones opcionales: `pip install -e ".[full]"` agrega capturas de pantalla (Pillow) y el
navegador headless (playwright). O elige una: `pip install -e ".[vision]"` o
`pip install -e ".[browser]"`. Tras instalar el extra del navegador, ejecuta
`python -m playwright install chromium`.

<details>
<summary>Instalar desde el código fuente (desarrollo)</summary>

```bash
git clone https://github.com/davepo/pedacito.git
cd pedacito
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

</details>

Desinstala con `pip uninstall pedacito`. Tu configuración y la carpeta `.pedacito/` de cada
proyecto quedan intactas.

## Actualización

Pedacito busca en PyPI versiones más recientes de forma discreta: **una sola petición GET anónima** de los
metadatos públicos de `pedacito` al iniciar, en un hilo en segundo plano con un tiempo límite de
2 segundos, en caché por 24 horas junto a tu configuración de usuario (`update_check.json`).
Sin cuenta, sin telemetría, sin datos del proyecto; y sin conexión, cada verificación es una
operación nula silenciosa. Por defecto solo se ofrecen versiones estables: las etiquetas de prerrelease
(`dev`, `rc`, `alpha`, `beta`) se ignoran, las versiones retiradas (yanked) se ignoran y
nunca se te dice que "actualices" a la versión que ya ejecutas. Define
`update_channel = "any"` en `[universal]` de tu configuración (o
`PEDACITO_UPDATE_CHANNEL=any`) para incluir prerreleases.

- **GUI**: cuando existe una versión más reciente, aparece un banner delgado bajo la
  barra superior: *Pedacito vX.Y.Z is available (you have A.B.C)*. **Update** ejecuta el
  comando de actualización de tu método de instalación en segundo plano y transmite su
  salida a un pequeño panel; **Stop** cancela una ejecución atascada (el trabajador también
  impone un límite de 10 minutos de reloj), y una ejecución fallida o cancelada muestra la
  salida más el comando manual. Si tiene éxito, **Restart Pedacito** reabre la
  GUI en el mismo puerto con la nueva versión. **Dismiss** oculta el banner
  hasta que aparezca una versión aún más nueva. El banner es parte del flujo: nunca cubre
  un control. Sin esperar al banner, **Settings → Updates** ofrece una
  verificación manual **Check for updates** que muestra la versión instalada frente a la última y si
  la respuesta vino de la caché de 24 h o de una verificación nueva; con
  `check_for_updates = false` explica la desactivación y no hace ninguna llamada
  de red.
- **CLI**: `pedacito update` imprime la versión instalada frente a la última y el
  método de instalación detectado; `pedacito update --run` ejecuta la actualización y devuelve su
  código de salida.
- **TUI**: `/update` muestra la misma comparación y ofrece la actualización;
  `/update run` omite la confirmación.

El comando de actualización se elige según cómo se instaló Pedacito, y siempre se ejecuta
como una lista de argumentos sin shell (seguro en Windows):

| Método de instalación | Actualización |
|---|---|
| pipx | `pipx upgrade pedacito` |
| virtualenv | `<venv python> -m pip install --upgrade pedacito` |
| pip --user | `<python> -m pip install --upgrade --user pedacito`: en los Python "externally managed" de PEP 668 (el Python del sistema en Debian/Ubuntu/Homebrew) pip puede negarse; ahí prefiere pipx o un venv |
| editable / copia del código fuente | sin comando automático: ejecuta `git pull && pip install -e .` en tu copia |

**Nota sobre la plataforma al reiniciar.** La garantía "seguro en Windows" de arriba se refiere a la
lista de argumentos del comando de actualización. El **Restart Pedacito** de la GUI es un
relanzamiento de proceso aparte: en Windows, Pedacito inicia el reemplazo
(`python -m pedacito gui …`) con `subprocess.Popen` y cierra el proceso
anterior, porque `os.execv` no reemplaza un proceso allí; en macOS y
Linux se vuelve a ejecutar en el lugar. En ambos casos la GUI regresa en el mismo puerto.

**Desactivarlo por completo**: define `check_for_updates = false` en `[universal]` de tu
configuración (o `PEDACITO_CHECK_FOR_UPDATES=false`). Eso desactiva cada verificación de versión
(cero llamadas de red, cero subprocesos de actualización, sin banner) en la
GUI, la TUI y `pedacito update` por igual; todo lo demás se comporta exactamente como
antes. Consulta la [referencia de configuración](CONFIG.md#referencia-de-configuración).

Después de actualizar, `pedacito config --upgrade` agrega a tu configuración las opciones
recién documentadas, comentadas (consulta [QUICKSTART B1](../../QUICKSTART.es.md)).

## ¿Qué modelo se ajusta a tu GPU?

| VRAM | Nivel de Pedacito | Modelos (Q4 a Q8) | En qué destaca |
|---|---|---|---|
| **6–8 GB** | `tiny` / `small` | 4B a 8B | Ediciones quirúrgicas de un solo archivo con `pedacito ask`, un `scout_profile` rápido, o el trabajador local combinado con un modelo `planner` o `escalate` más grande |
| **12–16 GB** | `medium` | 9B a 14B | Tareas de varios archivos, redacción de pruebas y flujos de `pedacito auto` con una ventana de contexto de 16k–32k |
| **24 GB+** (p. ej. RX 7900 XTX, RTX 3090, RTX 4090) | `large` | 24B a 35B | Andamiaje autónomo de varios archivos, enrutamiento automático por especialista entre especialistas de 24B–30B y ventanas de contexto de 32k |
| **Dos GPU o dos servidores** | scout + arquitecto | scout de 4B–9B, arquitecto de 24B–35B | Combinación de `scout_profile` sin intercambios entre puertos separados |

Los tamaños son solo de los pesos; suma la caché KV y unos 1.5 a 2.5 GB de
sobrecarga. La tabla completa, con ejemplos y consejos de razonamiento por nivel, está en
[../../MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md#niveles-de-hardware).

## Configuración del servidor de modelos

Inicia cualquier servidor que hable la API de chat-completions de OpenAI y luego dale a
Pedacito su URL base y un nombre de modelo.

| Servidor | Cómo iniciarlo | URL base |
|---|---|---|
| **LM Studio** | Carga un modelo y luego inicia el servidor (pestaña Developer). Para otra máquina, activa "Serve on Local Network". | `http://localhost:1234/v1` |
| **Ollama** | `ollama serve` y `ollama pull <model>`. Para otra máquina, define primero `OLLAMA_HOST=0.0.0.0`. | `http://localhost:11434/v1` |
| **Lemonade** | Inicia el servidor y carga un modelo. | La URL que imprime (termina en `/v1` o `/api/v1`) |
| **Strata / Avalon** | Inicia el servidor MoE por niveles y carga un modelo, siguiendo su propia documentación. | Su URL compatible con OpenAI, p. ej. `http://127.0.0.1:8000/v1` |
| **API alojada** | Obtén una clave de API. | p. ej. `https://api.deepseek.com/v1` |

Los puertos por defecto de arriba son indicios, no reglas: cualquier puerto funciona, y el escaneo
identifica un servidor por sus respuestas, así que dos servidores pueden coexistir en un mismo host
(por ejemplo Lemonade en 11434 junto a LM Studio en 1234).

Strata y Avalon son servidores de mezcla de expertos por niveles: mantienen residente la mayor parte de la
VRAM de la GPU y razonan antes de responder. Dale al perfil
`provider = "strata"` (o `"avalon"`), o un nombre que contenga `strata`/`avalon`,
y Pedacito aplica el nivel de ejecución `local-moe`: presupuestos de respuesta mayores, un
tiempo límite de al menos 600 segundos para la carga de pesos y exactamente un stream
concurrente para que un segundo prefill no pueda expulsar los pesos residentes. Pedacito también maneja su
ciclo de carga/descarga de modelos entre ejecuciones. Consulta
[Perfiles de ejecución y el nivel local-moe](ARCHITECTURE.md#perfiles-de-ejecución-y-el-nivel-local-moe)
para ver los presupuestos y un ejemplo de perfil.

Para las APIs alojadas define `base_url`, `api_key` y `model` (o `api_key_env`, el
nombre de una variable de entorno que contiene la clave, para que el secreto quede fuera del
archivo). Pedacito detecta un host remoto y cambia a `json_mode = "object"`
automáticamente.

Verifica la conexión desde la máquina donde corre Pedacito:

```bash
pedacito config --check
```

Distingue "servidor inalcanzable" de "modelo no encontrado o que no carga"
(normalmente una discrepancia de nombre o falta de VRAM libre).

`pedacito config --doctor` va más allá y calibra la configuración: ejecuta la
misma verificación, luego sondea qué modos de salida estructurada admite realmente el servidor
(`json_schema`, `json_object`, prompts simples), imprime un `json_mode`
recomendado, informa la ventana de contexto real del servidor y el tamaño de los pesos del modelo,
calibra la aritmética de tokens contra el propio tokenizador del servidor e informa si
el análisis con tree-sitter está activo (con su versión, o por qué está desactivado).
Los veredictos se recuerdan por servidor y modelo durante una semana, de modo que un servidor sin
soporte de gramática recibe automáticamente un ejemplo JSON few-shot en el prompt en
ejecuciones posteriores.

Pedacito también le pide al servidor que restrinja cada paso de búsqueda al JSON de acción
(`enforce_json_schema = true`, activado por defecto). Los servidores que admiten un esquema JSON
lo imponen durante el muestreo, de modo que no se puede producir un paso mal formado. Un servidor que
rechaza la petición se degrada automáticamente, primero a `json_object` y luego
a prompts simples, y la elección se recuerda por modelo. Cómo se comporta cada servidor
está en [../../MODEL_GUIDE.es.md](../../MODEL_GUIDE.es.md#json-impuesto-por-el-motor-según-el-servidor-enforce_json_schema).

Una vez por modelo, ejecuta `pedacito calibrate`. Ejecuta una minisuite programada:
elección de herramientas, adherencia a JSON, formato de edición, verificaciones de inyección plantada y
honestidad, recuperación en contexto largo, continuación, un A/B de razonamiento,
capacidades, y guarda una **tarjeta de modelo**. Cada sonda se ejecuta 3 veces (1 con
`--quick`; `--repeats N` lo anula) y la tarjeta registra la media y la dispersión,
de modo que las decisiones provienen de una media, no de una muestra ruidosa; una tarjeta anterior
(versión 1) se muestra con una nota para recalibrar y no se aplica. Los ajustes que dejaste en sus valores por defecto
siguen entonces a la tarjeta (paquete de prompts, modo de respuesta, verificador, best-of-N, profundidad de
pliegue, el interruptor de razonamiento); lo que definas explícitamente sigue ganando.
`pedacito config` muestra la tarjeta.

La calibración es **solo para modelos locales**: un perfil de proveedor alojado se rechaza
(salida 2 de la CLI, botón de la GUI deshabilitado) porque los modelos alojados usan
automáticamente los valores por defecto de la tabla del proveedor, y `models scan` nunca calibra las importaciones alojadas.
Para los modelos locales la tarjeta también registra los **valores por defecto de arquitectura** de una
biblioteca integrada (temperatura, sampler y ventana de contexto recomendados por
familia: Qwen coder, Qwen3, DeepSeek coder/R1, Nemotron, Gemma, Mistral,
Granite, Phi-4, Llama-3), limitados por la ventana que informa el servidor y tu
VRAM, con la aguja de contexto largo reduciendo la ventana (16384 y luego
8192) cuando falla. `pedacito calibrate --all` recorre cada perfil local por
turno, descargando cada modelo antes de que cargue el siguiente; `-y` omite la confirmación
y las tarjetas vigentes se omiten a menos que se indique `--recalibrate`. En la GUI el mismo
lote está en **Project ▾ → Calibrate All Local Models…**, y cada
fila de perfil lleva una insignia de calibración: ✓ Calibrated (con la fecha),
⚠ Calibration stale (más de 30 días o una versión de tarjeta antigua),
○ Not calibrated o ☁ Hosted API.

## Servidores de inferencia en GPU: ROCm y CUDA

Pedacito nunca ejecuta el modelo por sí mismo. Se comunica con un servidor que lo hace, así que el backend
de GPU es una propiedad del servidor que instales. Elige el servidor para tu tarjeta y luego
apunta Pedacito hacia él.

### Elegir un backend

| Tu GPU | Buenas opciones de servidor | Backend |
|---|---|---|
| **AMD Radeon** en Linux (revisa las notas de versión de ROCm de AMD para el soporte de tu tarjeta) | Lemonade Server, llama.cpp compilado con HIP, Ollama, LM Studio | ROCm (HIP). Vulkan es una alternativa que funciona en más tarjetas. |
| **AMD Radeon** en Windows | Lemonade Server, LM Studio, llama.cpp compilado con Vulkan | Vulkan o ROCm, según la compilación del servidor |
| **NVIDIA GeForce o RTX** | Ollama, LM Studio, llama.cpp compilado con CUDA, vLLM | CUDA |
| **Sin GPU dedicada** | Ollama o llama.cpp compilado para CPU | CPU. Espera solo modelos pequeños. |

Si un modelo no carga o produce basura en un backend, prueba con el otro. El propio texto de error de Pedacito
sugiere esto cuando detecta una discrepancia de runtime.

### AMD: ROCm

1. Instala el runtime de ROCm para tu distribución, siguiendo la propia guía de instalación de AMD para
   tu versión. Confirma que la tarjeta es visible:

   ```bash
   rocm-smi --showmeminfo vram
   ```

2. Inicia un servidor. Con **llama.cpp**, compila con soporte HIP y luego ejecuta el servidor. Las
   opciones de abajo son un punto de partida. Consulta la documentación de compilación de llama.cpp para tu
   versión y define `AMDGPU_TARGETS` con la arquitectura de tu tarjeta (por ejemplo `gfx1100`
   para la RX 7900 XTX).

   ```bash
   cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100
   cmake --build build --config Release -j
   ./build/bin/llama-server -m ./models/your-model.gguf \
     --host 127.0.0.1 --port 8080 -ngl 99 -c 16384
   ```

3. Apunta Pedacito hacia él. El endpoint compatible con OpenAI de llama.cpp es `/v1`:

   ```toml
   [universal]
   vram_gb = 24                              # your card's VRAM, in GB

   [profiles.rocm]
   base_url = "http://127.0.0.1:8080/v1"
   model    = "your-model.gguf"              # the id the server reports
   ```

La vista de memoria entre servidores de Pedacito lee la VRAM de AMD de los contadores ROCm del kernel
(`mem_info_vram_total` y `mem_info_vram_used`). Por eso la insignia de memoria muestra números
reales en ROCm. Si faltan los contadores, la insignia muestra solo la RAM del sistema.

### NVIDIA: CUDA

1. Instala el controlador de NVIDIA y el toolkit de CUDA para tu distribución. Confirma que la tarjeta es
   visible:

   ```bash
   nvidia-smi
   ```

2. Inicia un servidor. Con **llama.cpp**, compila con soporte CUDA:

   ```bash
   cmake -B build -DGGML_CUDA=ON
   cmake --build build --config Release -j
   ./build/bin/llama-server -m ./models/your-model.gguf \
     --host 127.0.0.1 --port 8080 -ngl 99 -c 16384
   ```

   **Ollama** y **LM Studio** detectan CUDA automáticamente. Inícialos como se describe en
   [Configuración del servidor de modelos](#configuración-del-servidor-de-modelos).

3. Define `vram_gb` a mano si es necesario. Pedacito lee la VRAM de NVIDIA mediante `nvidia-smi` cuando
   está disponible, así que la vista de memoria ve tu tarjeta automáticamente. Aun así puedes definir
   `vram_gb` manualmente; el valor dimensiona la ventana de contexto, así que usa la VRAM real de tu tarjeta.

### Caché KV cuantizada y contexto

Si tu runtime cuantiza la caché KV, define `k_cache_quant` y `v_cache_quant` en el
perfil (por ejemplo `"q8_0"`), para que la estimación de VRAM de Pedacito no sea demasiado pesimista. Si
el servidor informa una ventana de contexto desactualizada, define `context_tokens` explícitamente. Consulta
[CONFIG.md](CONFIG.md#perfiles-niveles-y-presupuestos-de-tokens) para ver la lista completa de claves de perfil.

### Verificar la ruta de GPU de extremo a extremo

```bash
pedacito config --check     # reachable? model found? generation works?
pedacito config --doctor    # also probes structured output and the real context window
pedacito models scan        # lists the models the server reports and writes a profile for each
```

Luego ejecuta `pedacito calibrate` una vez por modelo. Guarda una tarjeta de modelo que ajusta los valores por defecto
a lo que ese modelo hace en tu hardware.
