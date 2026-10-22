# Протокол измерений

`bench.run` создаёт одинаковые contiguous FP32 inputs с фиксированным seed и
переиспользует их во всех реализациях. FP64 reference, аллокации, подготовка,
JIT и первый запуск происходят до измерения. Выход предварительно заполнен NaN:
незаписанный хвост не сможет случайно пройти проверку.

Измеряются CUDA events вокруг replay графа с 32 вызовами kernel. По умолчанию
30 независимых replay samples, результат каждого делится на 32. Сохраняются
все samples, median/p10/p90. Это steady-state latency с прогретыми кэшами и
переиспользованием одних адресов. Python launch overhead, JIT, передачи CPU/GPU
и end-to-end latency приложения этим числом не описываются. Replay overhead
остаётся в измерении, хотя и амортизируется. Для cold-cache bandwidth нужен
отдельный явно описанный режим; его здесь пока нет.

Общий non-default CUDA stream используется всеми реализациями. CuTe и C++
получают его явно; графовые запуски проверяются тестами. Порядок backend
перемешивается детерминированно по seed. Для серьёзной серии повторяй процесс с
несколькими seed и чередуй порядки; одного короткого прогона недостаточно.

JSON содержит git revision и dirty status, хеш lock, Python/compiler/packages,
GPU name/UUID/SM, driver, clocks/power/temperature/utilization, compute processes
до/после. Эти снимки не доказывают эксклюзивность GPU на всём интервале.
На общем сервере прогоны годятся для проверки harness; итоговые числа для статьи
снимай на выделенной GPU с записанными clocks/power policy и без соседней нагрузки.
Автоматического изменения clocks, power limit или завершения чужих процессов нет.

Текущие softmax baseline используют разные decomposition/memory strategies.
Нельзя интерпретировать их отношение latency как границу выразительности языков.
Следующие сравнения: одинаковый алгоритм/precision; одинаковый бюджет тюнинга;
затем лучшие специфичные реализации с объяснением использованных механизмов.
Регистрируй failures, unsupported cases и время компиляции отдельно от latency.

## Профилировщики

`source scripts/env.sh` добавляет инструменты установленного CUDA Toolkit в PATH.
`bench.doctor` показывает доступность `ncu`, `nsys`, `compute-sanitizer`.

```bash
# Без Python: проверить CUDA C++ memory access и runtime errors
compute-sanitizer --tool memcheck ./build/sm_86/cuda_smoke

# Только один прогретый kernel внутри profiler API range
ncu --profile-from-start off --set basic -o results/cute-softmax \
  python -m bench.profile --language cute --task 01-softmax
nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o results/triton-softmax python -m bench.profile --language triton --task 01-softmax
```

Nsight Compute hardware counters могут быть закрыты администратором
(`ERR_NVGPUCTRPERM`). Это не ошибка kernel; скрипты не меняют настройку драйвера.
Не сравнивай profiler-instrumented latency с обычным бенчмарком. Для анализа
собирай register count, occupancy, DRAM/L2 traffic, achieved bandwidth,
Tensor Core utilization и stall reasons, когда они релевантны задаче.
