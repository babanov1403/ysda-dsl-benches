# План экспериментов

Первый этап ограничен Triton, CuTe DSL и CUDA C++.

| Задача | Назначение | Состояние |
|---|---|---|
| 00-vector-add | установка, индексы, маски хвостов, stream interop | 3 реализации |
| 01-softmax | reductions, регистры, repeated reads | 3 baseline, без тюнинга |
| 02-prefixsum | scan, межблочная координация | спецификация, реализаций пока нет |
| 03-reduction | многоуровневое reduction, размер блока | план |
| 04-transpose | coalescing, shared memory bank conflicts | план |
| 05-gemm | Tensor Cores, layout, async movement, pipelining | план |
| 06-rmsnorm | fusion и bandwidth | план |

Для каждого case study записывать:

- shape/dtype/layout и численную семантику;
- алгоритм, tiling, scheduling, число warps/stages, registers/shared memory;
- какие решения принимает компилятор и какие доступны программисту;
- корректность, latency distributions и profiler evidence;
- бюджет тюнинга, compile time и объём поддерживаемого кода;
- что пришлось поменять при переходе на другую архитектуру.

После устойчивых базовых задач: persistent/grouped GEMM, fused normalization +
linear, attention-like kernels и quantized GEMM. Architecture-specific варианты
держать рядом с portable baseline и явно указывать минимальный SM/feature set.
LLM-эксперименты и новые DSL добавлять после стабилизации протокола.
