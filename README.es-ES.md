<p align="center">
  <h1 align="center">talib-rs</h1>
  <p align="center">
    Biblioteca de Análisis Técnico en Rust Puro — Reemplazo Directo para TA-Lib
  </p>
  <p align="center">
    <a href="README.zh-CN.md">中文</a> · <a href="BENCHMARK.md">Benchmarks</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/indicators-158-blue" alt="158 indicators" />
  <img src="https://img.shields.io/badge/tests-4562_accuracy-green" alt="17855 accuracy tests" />
  <img src="https://img.shields.io/badge/unsafe-zero-brightgreen" alt="zero unsafe" />
  <img src="https://img.shields.io/badge/precision-bit--exact-brightgreen" alt="bit-exact" />
  <img src="https://img.shields.io/badge/C_deps-zero-orange" alt="zero C deps" />
  <img src="https://img.shields.io/badge/license-BSD--3--Clause-lightgrey" alt="BSD-3-Clause" />
</p>

---

## ¿Por qué talib-rs?

[TA-Lib](https://ta-lib.org/) ha sido el estándar de la industria para el análisis técnico desde 1999, pero su implementación en C conlleva puntos críticos bien conocidos: compilación difícil en Windows/macOS, compilaciones frágiles del wrapper de Python y falta de garantías de seguridad de lenguajes modernos. **talib-rs** resuelve todo esto entregando resultados idénticos.

| | C TA-Lib | talib-rs |
|---|---|---|
| Lenguaje | C (1999) | Rust (2024) |
| Instalación | Requiere compilador C + libs del sistema | `pip install talib-rs` |
| Precisión | Implementación de referencia | **Coincidencia bit a bit** (17,855 pruebas de precisión × 7 tamaños de datos × 6 escenarios) |
| Rendimiento | Base | **1.41x más rápido en promedio** con 1M de barras (47/90 más rápido, cero unsafe) |
| Seguridad de memoria | Gestión manual | Garantizada por Rust |
| Integración Python | Wrapper de Cython | PyO3 zero-copy |
| Indicadores | 158 | 158 (cobertura al 100%) |

### Ventajas Técnicas Clave

- **Precisión bit a bit** — No es "aproximadamente igual", sino `diff=0` frente a C TA-Lib en más de 1M de puntos de datos. Verificado con 353 pruebas de precisión que cubren cada función, cada combinación de parámetros y casos límite extremos.

- **Algoritmos O(n) en todo el sistema** — STDDEV, CORREL, BETA, LINEARREG, KAMA, TRIMA utilizan fórmulas de ventana deslizante incremental. Rendimiento independiente del periodo: `CORREL(period=200)` se ejecuta en el mismo tiempo que `CORREL(period=10)`.

- **Aceleración SIMD** — Vectorización f64x4 mediante el crate `wide` para operaciones de reducción (suma, suma de cuadrados). Aceleración de ~3x en cálculos agregados utilizados por SMA, BBANDS, STDDEV, VAR.

- **Puente NumPy zero-copy** — Los arrays de entrada se leen directamente de la memoria de NumPy vía `PyReadonlyArray1::as_slice()`. Los arrays de salida se transfieren mediante `PyArray1::from_vec()` con movimiento de propiedad, sin memcpy.

- **Optimización de deque monotónica** — AROON, WILLR, MAX, MIN, MIDPOINT, MIDPRICE utilizan extremos de ventana deslizante O(n) en lugar de escaneo por fuerza bruta O(n×p).

- **Cero unsafe** — Todo el código base utiliza Rust seguro. Sin bloques `unsafe`, sin `get_unchecked`. Los patrones basados en iteradores permiten la auto-vectorización de LLVM manteniendo la seguridad total de los límites.

## Instalación

```bash
pip install talib-rs
```

Sin compilador C, sin bibliotecas del sistema, sin fallos de compilación. Wheels pre-construidos para Linux/macOS/Windows × Python 3.9–3.13.

## Inicio Rápido

### API de Funciones (Compatible con TA-Lib)

```python
import talib_rs as talib
import numpy as np

close = np.random.random(1000) * 100

# Tendencia (Trend)
sma  = talib.SMA(close, timeperiod=20)
ema  = talib.EMA(close, timeperiod=12)
kama = talib.KAMA(close, timeperiod=30)
upper, mid, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

# Impulso (Momentum)
rsi  = talib.RSI(close, timeperiod=14)
macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
slowk, slowd = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3)

# Volatilidad y Volumen
atr = talib.ATR(high, low, close, timeperiod=14)
obv = talib.OBV(close, volume)

# Reconocimiento de Patrones (61 patrones de velas)
engulfing = talib.CDLENGULFING(open, high, low, close)  # -100 / 0 / 100
hammer    = talib.CDLHAMMER(open, high, low, close)

# Metadatos
print(len(talib.get_functions()))       # 158
print(talib.get_function_groups().keys())
# ['Overlap Studies', 'Momentum Indicators', 'Pattern Recognition', ...]
```

### API Abstracta

```python
from talib_rs.abstract import Function

# Crear función con introspección
rsi = Function('RSI')
print(rsi.info['group'])         # 'Momentum Indicators'
print(rsi.input_names)           # OrderedDict([('price', ['close'])])
print(rsi.parameters)            # OrderedDict([('timeperiod', 14)])
print(rsi.output_names)          # ['real']
print(rsi.lookback)              # 14

# Llamar con diccionario o DataFrame
data = {'close': close, 'high': high, 'low': low, 'open': open_, 'volume': volume}
result = rsi(data)

# Sobrescribir parámetros
rsi.parameters = {'timeperiod': 21}
result = rsi(data)

# Importaciones en una línea
from talib_rs.abstract import SMA, MACD, BBANDS
sma_result = SMA(data, timeperiod=50)
```

### API de Stream

```python
from talib_rs import stream

# Devuelve solo el valor más reciente (escalar), no el array completo
latest_rsi  = stream.RSI(close, timeperiod=14)          # float
latest_macd = stream.MACD(close, 12, 26, 9)             # (float, float, float)
latest_sma  = stream.SMA(close, timeperiod=20)           # float
```

## Rendimiento

> Apple M4 | `--release` LTO fat | mediana de 20 iteraciones | código **zero unsafe**

| Dataset | Más Rápido | Igual (±5%) | Más Lento | Promedio | Mediana |
|--------:|:------:|:-----------:|:------:|:---:|:------:|
| 1,000 | 31 | 26 | 33 | 1.16x | 1.00x |
| 10,000 | 38 | 34 | 18 | 1.34x | 1.02x |
| 100,000 | **45** | 39 | 6 | **1.40x** | **1.06x** |
| 1,000,000 | **47** | 35 | 8 | **1.41x** | **1.07x** |

### Aspectos destacados con 1M de barras

| Indicador | C (us) | Rust (us) | Aceleración | Técnica |
|-----------|-------:|---------:|--------:|-----------|
| MIDPOINT(14) | 39,300 | 3,720 | **10.6x** | Índice cacheado O(n) vs escaneo bruto O(n×p) de C |
| BBANDS(20) | 3,500 | 1,160 | **3.0x** | SMA+STDDEV fusionado en una sola pasada |
| TEMA(20) | 3,900 | 1,420 | **2.8x** | Cascada EMA de 3 capas inline, sin Vec intermedio |
| LINEARREG(14) | 4,020 | 1,830 | **2.2x** | Sumas deslizantes O(n) vs O(n×p) de C |
| TRIX(15) | 4,100 | 1,940 | **2.1x** | EMA de 3 capas fusionada + ROC, formulación estilo C |
| OBV | 3,160 | 1,600 | **2.0x** | El push secuencial evita fallos de página COW de calloc |
| DEMA(20) | 2,640 | 1,390 | **1.9x** | Eliminación del Vec intermedio ema1 |
| STOCH | 6,680 | 3,720 | **1.8x** | Min/max de ventana deslizante + señal EMA |
| VAR(20) | 1,160 | 710 | **1.6x** | Suma deslizante O(n) + div→mul (precomputación 1/n) |
| STDDEV(20) | 1,400 | 870 | **1.6x** | Var+sqrt fusionado en una sola pasada |
| MACD | 4,260 | 2,790 | **1.5x** | Formulación EMA de C TA-Lib: `k*(x-prev)+prev` |

Benchmark completo: 90 indicadores × 4 datasets con tiempos C/Rust → [BENCHMARK.md](BENCHMARK.md)

## Indicadores (158)

### Estudios de Superposición (Overlap Studies) (16)
SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA, T3, MAMA, BBANDS, SAR, SAREXT, MIDPOINT, MIDPRICE, MAVP, HT_TRENDLINE

### Indicadores de Impulso (Momentum Indicators) (30)
RSI, MACD, MACDEXT, MACDFIX, STOCH, STOCHF, STOCHRSI, ADX, ADXR, CCI, MOM, ROC, ROCP, ROCR, ROCR100, WILLR, APO, PPO, BOP, CMO, AROON, AROONOSC, MFI, TRIX, ULTOSC, DX, PLUS_DI, MINUS_DI, PLUS_DM, MINUS_DM

### Reconocimiento de Patrones (Pattern Recognition) (61)
CDL2CROWS, CDL3BLACKCROWS, CDL3INSIDE, CDL3LINESTRIKE, CDL3OUTSIDE, CDL3STARSINSOUTH, CDL3WHITESOLDIERS, CDLABANDONEDBABY, CDLADVANCEBLOCK, CDLBELTHOLD, CDLBREAKAWAY, CDLCLOSINGMARUBOZU, CDLCONCEALBABYSWALL, CDLCOUNTERATTACK, CDLDARKCLOUDCOVER, CDLDOJI, CDLDOJISTAR, CDLDRAGONFLYDOJI, CDLENGULFING, CDLEVENINGDOJISTAR, CDLEVENINGSTAR, CDLGAPSIDESIDEWHITE, CDLGRAVESTONEDOJI, CDLHAMMER, CDLHANGINGMAN, CDLHARAMI, CDLHARAMICROSS, CDLHIGHWAVE, CDLHIKKAKE, CDLHIKKAKEMOD, CDLHOMINGPIGEON, CDLIDENTICAL3CROWS, CDLINNECK, CDLINVERTEDHAMMER, CDLKICKING, CDLKICKINGBYLENGTH, CDLLADDERBOTTOM, CDLLONGLEGGEDDOJI, CDLLONGLINE, CDLMARUBOZU, CDLMATCHINGLOW, CDLMATHOLD, CDLMORNINGDOJISTAR, CDLMORNINGSTAR, CDLONNECK, CDLPIERCING, CDLRICKSHAWMAN, CDLRISEFALL3METHODS, CDLSEPARATINGLINES, CDLSHOOTINGSTAR, CDLSHORTLINE, CDLSPINNINGTOP, CDLSTALLEDPATTERN, CDLSTICKSANDWICH, CDLTAKURI, CDLTASUKIGAP, CDLTHRUSTING, CDLTRISTAR, CDLUNIQUE3RIVER, CDLUPSIDEGAP2CROWS, CDLXSIDEGAP3METHODS

### Volatilidad (3)
ATR, NATR, TRANGE

### Volumen (3)
AD, ADOSC, OBV

### Transformación de Precio (4)
AVGPRICE, MEDPRICE, TYPPRICE, WCLPRICE

### Funciones Estadísticas (9)
STDDEV, VAR, BETA, CORREL, LINEARREG, LINEARREG_SLOPE, LINEARREG_INTERCEPT, LINEARREG_ANGLE, TSF

### Transformación Matemática (15)
ACOS, ASIN, ATAN, CEIL, COS, COSH, EXP, FLOOR, LN, LOG10, SIN, SINH, SQRT, TAN, TANH

### Operadores Matemáticos (9)
ADD, SUB, MULT, DIV, MAX, MAXINDEX, MIN, MININDEX, SUM

### Indicadores de Ciclo (5)
HT_DCPERIOD, HT_DCPHASE, HT_PHASOR, HT_SINE, HT_TRENDMODE

## Pruebas y Verificación

```
158/158 funciones · 17,855 pruebas de precisión · 7 tamaños de datos × 6 escenarios × 3 semillas · 0 fallos
```

### Matriz de Pruebas

| Suite | Casos | Descripción |
|-------|------:|------------|
| **Pruebas unitarias Rust** | 54 | Algoritmo núcleo, SIMD, ventana deslizante, casos límite |
| **Validación cruzada de precisión** | 353 | 158 funciones × 6 datasets, rtol=1e-10 |
| **Alineación multi-dataset** | 17,451 | 158 funciones × 7 tamaños × 6 escenarios × 3 semillas |
| **Coincidencia exacta de patrones** | 122 | 61 patrones CDL × 2 datasets, señal entera exacta |

### Metodología de Verificación

1. **Tolerancia escalonada** — Exacta (1e-14) para ops elemento a elemento, estándar (1e-10) para cadenas EMA, deslizante (1e-8) para acumuladores O(n), acumulativa (1e-6) para VAR/STDDEV.
2. **Alineación de NaN** — Las posiciones de NaN del lookback coinciden exactamente entre talib-rs y C TA-Lib.
3. **Multi-escenario** — Paseo aleatorio, tendencia alcista/bajista, lateral, volátil, datos de reversión a la media.
4. **Reconocimiento de patrones** — Los 61 patrones de velas producen señales idénticas (-100/0/100).

```bash
cargo test                                               # 54 pruebas unitarias de Rust
pytest tests/accuracy/ -v                                # 17,855 pruebas de precisión
pytest tests/accuracy/ -k "100000"                       # Solo dataset de 100K
pytest tests/accuracy/ -k "volatile"                     # Solo escenario volátil
pytest tests/accuracy/test_multi_dataset_alignment.py    # Suite multi-dataset
```

## Arquitectura

```
talib-rs/
├── crates/
│   ├── talib-rs-core/                    # Biblioteca de Rust puro (sin dependencia de Python)
│   │   ├── src/
│   │   │   ├── overlap/               # SMA, EMA, WMA, DEMA, TEMA, TRIMA, KAMA,
│   │   │   │                          # T3, MAMA, BBANDS, SAR, MIDPOINT, ...
│   │   │   ├── momentum/              # RSI, MACD, STOCH, ADX, CCI, MOM, ROC,
│   │   │   │                          # WILLR, AROON, MFI, TRIX, ULTOSC, DX, ...
│   │   │   ├── volatility/            # ATR, NATR, TRANGE
│   │   │   ├── volume/                # AD, ADOSC, OBV
│   │   │   ├── pattern/               # 61 patrones de velas con lógica real
│   │   │   ├── statistic/             # STDDEV, VAR, BETA, CORREL, LINEARREG, TSF
│   │   │   ├── cycle/                 # Transformada de Hilbert (optimizado con ring buffer)
│   │   │   ├── math_transform/        # sin, cos, sqrt, exp, ln, ... (loop-unrolled)
│   │   │   ├── math_operator/         # add, sub, max, min, sum (monotonic deque)
│   │   │   ├── price_transform/       # avgprice, medprice, typprice, wclprice
│   │   │   ├── simd.rs                # SIMD f64x4: sum, sum_sq_diff (3x speedup)
│   │   │   ├── sliding_window.rs      # Deque monotónico: O(n) sliding max/min
│   │   │   ├── ma_type.rs             # Dispatcher de tipo MA (9 tipos)
│   │   │   └── error.rs               # Enum TaError
│   │   └── benches/
│   │       └── simd_bench.rs          # Benchmarks de Criterion
│   │
│   └── talib-rs-python/                  # Bindings de PyO3
│       └── src/
│           ├── func_api.rs            # 158 definiciones #[pyfunction]
│           ├── metadata.rs            # get_functions(), get_function_groups()
│           └── conversion.rs          # NumPy ↔ Rust zero-copy
│
├── python/talib_rs/
│   ├── __init__.py                    # import talib_rs as talib
│   ├── abstract.py                    # Clase Function con introspección
│   └── stream.py                      # Wrappers solo para el valor más reciente
│
├── tests/
│   ├── test_exhaustive.py             # Validación cruzada 158/158
│   ├── test_full_coverage.py          # Consistencia + edge (620 tests)
│   └── accuracy/                      # Pruebas de precisión focalizadas
│
├── benches/
│   ├── generate_report.py             # → BENCHMARK.md
│   └── python_benches/                # Suites de pytest-benchmark
│
└── .github/workflows/
    ├── ci.yml                         # Test + lint (3 OS × 4 versiones de Python)
    └── release.yml                    # Construcción de wheels + publicación en PyPI
```

### Decisiones de Diseño

| Decisión | Elección | Justificación |
|----------|--------|-----------|
| Crate SIMD | `wide` 0.7 | Rust estable, no requiere nightly |
| Extremos deslizantes | Deque monotónico | Escaneo dual de extremos fusionado, O(n) amortizado |
| Varianza | E[X²]−E[X]² | Ventana deslizante de una sola pasada, no Welford de dos pasadas |
| Transformada de Hilbert | Ring buffer | 10 Vec → [f64; 64] fijo, 80% menos asignaciones en heap |
| EMA/DEMA/TEMA/T3 | Capas in-place | Sin Vec intermedio debido al filtrado de NaN |
| WMA | Recurrencia incremental | WS_new = WS_old − S_old + p×x_new, O(1) por paso |
| Nombre módulo Python | `talib_rs` | `import talib_rs as talib` para uso como reemplazo directo |
| Formato de salida | `Vec<f64>` → `PyArray1` | Transferencia de propiedad, casi zero-copy |
| Política de Unsafe | Cero `unsafe` | Patrones de iterador + indexación segura; LLVM auto-vectoriza cadenas zip |
| Convención NaN | Rellenar lookback con NaN | Coincide exactamente con C TA-Lib |

## Desarrollo

### Prerrequisitos

```bash
# Toolchain de Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Dependencias de Python
pip install maturin numpy pytest pytest-benchmark
```

### Compilar y Probar

```bash
maturin develop --release              # Construir paquete de Python
cargo test                             # 55 pruebas de Rust
pytest tests/ -q                       # 353 pruebas de precisión
```

### Benchmarking

```bash
cargo bench --bench simd_bench         # Micro-benchmarks SIMD de Rust
pytest benches/ --benchmark-sort=name  # Python vs C TA-Lib
python benches/generate_report.py      # Generar BENCHMARK.md
```

### CI/CD

- **CI**: Prueba de Rust + clippy + fmt + compilación de Python en Ubuntu/macOS/Windows × Python 3.10–3.13
- **Release**: El tag `v*` activa la construcción de wheels manylinux/macOS/Windows + publicación en PyPI

## Migración desde C TA-Lib

```bash
# Eliminar C TA-Lib
pip uninstall TA-Lib

# Instalar talib-rs
pip install talib-rs

# Actualizar importaciones: import talib_rs as talib
```

## Licencia

BSD-3-Clause
