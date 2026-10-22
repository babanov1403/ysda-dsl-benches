# YSDA GPU DSL experiments

Воспроизводимая площадка для **Triton, NVIDIA CuTe DSL и CUDA C++ Runtime API**.
Вопрос исследования: сколько контроля над GPU нужно программисту, чтобы получить
высокую производительность, и насколько переносимы эти решения между архитектурами?

Здесь уже есть установка окружения, сборка CUDA C++, одинаковые контракты задач,
проверки корректности, общий измеритель и примеры на всех трёх языках. Это старт
исследования: текущие kernels — базовые реализации, а не оптимизированные победители.

## Структура

```text
triton/                  # реализации Triton (не Python-пакет!)
  00-vector-add/kernel.py
  01-softmax/kernel.py
  02-prefixsum/README.md  # следующая задача, ещё не реализована
cute/                    # та же нумерация, NVIDIA CuTe DSL
cuda/                    # та же нумерация, CUDA C++ и CMakeLists.txt
  00-vector-add/main.cu   # самостоятельная программа без Python/PyTorch
bench/                   # общие данные, эталон, измерение, сведения об окружении
tests/                   # одинаковые тесты для трёх реализаций
scripts/                 # установка, окружение, сборка
docs/                    # протокол измерений, переносимость, развитие
pyproject.toml           # фиксированные прямые зависимости
uv.lock                  # полное дерево зависимостей, URL и хеши пакетов
results/                 # локальные JSON, не включены в Git
build/sm_<arch>/          # локальная сборка для выбранной архитектуры, не в Git
```

Языки находятся на верхнем уровне, задачи внутри них. Общие данные и протокол
не дублируются между языками. Зависимости описаны в корне, поскольку сейчас все
реализации используют один PyTorch для данных и измерений. Отдельные окружения
имеет смысл вводить, когда появятся несовместимые требования DSL.

## Быстрый старт

Требования: **Linux x86_64**, NVIDIA GPU **Ampere / SM80 или новее**, совместимый
NVIDIA driver, **CUDA Toolkit 13.3+** с `nvcc`, C++ compiler и `python3-venv`.
Загрузка Python-пакетов занимает несколько GB. `sudo` скриптам не нужен;
драйвер и системный Toolkit они не изменяют.

```bash
# На другой машине с доступом по SSH к исходному серверу:
git clone babanovil@185.185.59.19:/home/babanovil/ysda-dsl
cd ysda-dsl

# Выбрать разрешённую/свободную GPU; на однокарточной машине это обычно 0.
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda    # либо путь к установленному Toolkit 13.3+
bash scripts/bootstrap.sh
source scripts/env.sh
python -m bench.doctor
python -m pytest -q
make smoke
make bench
```

Псевдоним `shadgpu` существует только на машине, где он был настроен. Для другой
машины используй адрес из команды выше и свой SSH-ключ/пароль. Также репозиторий
можно перенести через `git bundle`; окружение и бинарники нужно собрать заново.

На текущем сервере рабочий каталог — `/home/babanovil/ysda-dsl`. Для каждого
нового shell достаточно `source scripts/env.sh` и выбора `CUDA_VISIBLE_DEVICES`.
Выбор GPU не зашит в репозиторий. `CUDA_VISIBLE_DEVICES` принимает и UUID карты.

Установка использует `uv sync --locked`, Python 3.12 и версии из `uv.lock`.
`uv` устанавливается отдельно в `.tools/venv`, рабочие пакеты — в `.venv`.
Не нужно запускать `pip install --upgrade` внутри этого окружения: изменение
зависимостей оформляется правкой `pyproject.toml`, новым lock и повторными тестами.

## Что запускать

```bash
# Самостоятельная CUDA C++ программа (без Python)
bash scripts/build.sh
./build/sm_86/cuda_smoke             # на A4000; для другой GPU свой sm_<arch>
ctest --test-dir build/sm_86 --output-on-failure

# Все три реализации на одинаковых данных
python -m bench.run --task 00-vector-add --size 1048579 \
  --output results/vector-add.json
python -m bench.run --task 01-softmax --rows 1024 --cols 1024 \
  --output results/softmax.json

# Один backend или другая форма
python -m bench.run --languages cute --task 01-softmax --rows 17 --cols 1009
```

Поддерживаются contiguous FP32 tensors, `1 <= numel < 2**31`; у softmax
`1 <= cols <= 16384`, редукция по последней оси. Данные должны быть конечными.
Пустые tensors, strided layouts, FP16/BF16, NaN/Inf и in-place операции пока вне
контракта. Эталон вычисляется в FP64 и приводится к FP32. Допуски:
`rtol=2e-5, atol=2e-6`; дополнительно проверяется сумма строки softmax.

Triton softmax использует одну программу на строку и tile в регистрах;
CuTe/CUDA baseline — warp на строку и несколько проходов по данным. Это разные
исходные алгоритмы. Сравнение пределов DSL потребует выравнивания алгоритма,
тюнинга и анализа профилей. См. [протокол](docs/benchmarking.md).

## Добавление задачи

1. Зафиксировать форму, dtype, допуски и математическую семантику в `bench/tasks.py`.
2. Создать одинаково названные каталоги в `triton/`, `cute/`, `cuda/`.
3. Реализовать `prepare(inputs, out) -> launch()`: подготовка может компилировать,
   `launch()` только запускает kernel на текущем потоке. Входы/выход должны жить
   до завершения всех запусков. После подготовки поток не меняется.
4. Добавить задачу в реестр и общие тесты, затем прогнать их для всех backend.
5. Только после этого добавлять варианты оптимизаций и результаты экспериментов.

Не добавляй `triton/__init__.py`: имя каталога не должно перекрыть установленный
пакет `triton`. Реализации загружаются по пути, поэтому `01-softmax` не обязан
быть допустимым именем Python-пакета.

## Документация и первоисточники

- [Переносимость и зависимости](docs/portability.md)
- [Измерения и профилирование](docs/benchmarking.md)
- [План исследования](docs/roadmap.md)
- [Triton installation](https://triton-lang.org/main/getting-started/installation.html)
- [CuTe DSL quick start](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
- [CUTLASS / CuTe DSL](https://github.com/NVIDIA/cutlass)
- [CUDA Linux installation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)

Версии latest в онлайн-документации могут отличаться от зафиксированных здесь.
Источником воспроизводимой установки служит `uv.lock`, а не плавающая ветка main.
