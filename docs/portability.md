# Переносимость

Зафиксирован профиль Linux x86_64 / CPython 3.12 / CUDA 13.3+:
PyTorch 2.14.1, Triton 3.8.0, nvidia-cutlass-dsl[cu13] 4.8.1.
PyTorch использует собственный CUDA runtime из wheel; системный `nvcc` собирает
отдельную библиотеку через C ABI, без C++ ABI зависимости от PyTorch. Все Python
транзитивные зависимости и их хеши записаны в `uv.lock`.

Внешние prerequisites — Linux, NVIDIA driver, CUDA Toolkit и host compiler.
Они не устанавливаются lock-файлом. `bench.doctor` и JSON бенчмарка фиксируют
их фактические версии. Для точного воспроизведения компиляции нужны также те же
Toolkit/compiler/driver; для переносимости на другую GPU допускаются совместимые
версии, но это уже другая конфигурация эксперимента. Python patch version тоже
фиксируется в отчёте, а файл `.python-version` ограничивает minor version.

`bash scripts/build.sh` получает compute capability через PyTorch у выбранной
GPU и создаёт `build/sm_<arch>`. CMake компилирует код для этой архитектуры.
Triton и CuTe компилируют для текущей GPU при первом вызове. Не переноси `.venv`,
`build/` и JIT-кэши на другую машину — переноси исходники и lock.

```bash
export CUDA_VISIBLE_DEVICES=0
source scripts/env.sh
bash scripts/build.sh
python -m bench.doctor
python -m pytest -q
```

Если меняется Toolkit/host compiler на той же архитектуре, создай свежий каталог
сборки (старый можно переименовать) перед запуском `scripts/build.sh`; CMake
сохраняет пути компилятора в cache. Для предварительной сборки без GPU:
`YSDA_CUDA_ARCH=90 bash scripts/build.sh`. Такой override не меняет JIT-архитектуру
и не заставляет runtime загружать несовместимый бинарник.

| Семейство | Типичные SM | Статус этого проекта |
|---|---|---|
| Ampere | 80, 86 | исходная проверка на RTX A4000 / SM86 |
| Ada | 89 | базовые операции должны переноситься; нужен прогон тестов |
| Hopper | 90 | базовые операции должны переноситься; нужен прогон тестов |
| Blackwell | 100, 103, 120 и др. | зависит от конкретной GPU и поддержки toolchain; нужен прогон тестов |

Таблица не является заявлением о проверке на всех GPU. Bootstrap и JIT решают
выбор target, но не гарантируют одинаковую скорость. TMA, WGMMA, tcgen05,
warp specialization должны добавляться отдельными вариантами с явными требованиями
к архитектуре. Нельзя молча заменять неподдерживаемый вариант PyTorch-реализацией.

CPU-only macOS/Windows и Linux aarch64 не входят в текущий lock-профиль.
При необходимости добавляется отдельный проверенный профиль, не подмена текущего.
Старый driver, который не поддерживает этот CUDA stack, требует обновления
администратором или отдельного профиля CUDA 12.9 со своим lock и тестами.

## Перенос через bundle

```bash
# На исходной машине после коммита:
git bundle create ysda-dsl.bundle --all
# На целевой машине:
git clone /path/to/ysda-dsl.bundle ysda-dsl
cd ysda-dsl
export CUDA_VISIBLE_DEVICES=0
bash scripts/bootstrap.sh
```

Обычный checkout можно клонировать по SSH. Для совместной работы и push лучше
позднее завести GitHub/GitLab remote или bare repository; push в активную ветку
этого рабочего каталога не нужен.
