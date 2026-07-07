# capburn

Captcha recognizer built on [Burn](https://burn.dev). One Rust workspace with
two deliverables:

- **`capburn`** — a trainer CLI (GPU/CPU) that learns to read captchas.
- **`capburn_php`** — a native PHP extension that loads a trained model and
  recognizes captchas from PHP, with no Composer wrapper required.

The character set and captcha length are **runtime parameters** — you can train
a model for digits only, Cyrillic + digits, a fixed length, etc., or let the
tool detect the format automatically from your dataset.

📖 **[Русская версия — ниже](#русская-версия)**

---

## Table of contents

- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Dataset format](#dataset-format)
- [Training](#training)
  - [Backends (GPU / CPU / CUDA)](#backends)
  - [Charset and length](#charset-and-length)
  - [Auto-detection](#auto-detection)
  - [Training on a remote server](#training-on-a-remote-server)
- [Inference from the CLI](#inference-from-the-cli)
- [PHP extension](#php-extension)
- [Model artifacts](#model-artifacts)
- [CI / Releases](#ci--releases)

---

## Project layout

```
capburn/
├── crates/
│   ├── capburn-core/     # model, charset, image preprocessing, CPU inference (backend-agnostic)
│   ├── capburn-train/    # trainer CLI (bin: `capburn`), wgpu/cpu/cuda backends
│   └── capburn-php/      # PHP extension (cdylib), CPU inference via ndarray
├── data/                 # datasets (git-ignored) — one subfolder per dataset
├── artifacts/            # trained models (git-ignored)
└── .github/workflows/    # CI + release builds
```

`capburn-core` is the shared heart: the network definition and inference code
are backend-independent, so the trainer and the PHP extension load **exactly the
same model**.

## Requirements

- **Rust** stable (1.85+). Install via [rustup](https://rustup.rs).
- For the PHP extension: **PHP 8.1–8.5** with development headers
  (`php-config` on `PATH`). On Debian/Ubuntu: `apt install php-dev`.
- For GPU training: a Metal (macOS) or Vulkan (Linux/Windows) capable GPU —
  no extra SDK needed thanks to `wgpu`. For NVIDIA CUDA, the CUDA toolkit.

## Dataset format

- Each dataset lives in its **own subfolder** under `data/`, e.g.
  `data/digits5/`, `data/cyrillic4/`.
- One image per captcha (PNG/JPEG/…). The **label is taken from the file name**,
  specifically the part **before the first `_`**:

  ```
  12345.png              → label "12345"
  12345_a1b2c3.png       → label "12345"   (suffix after "_" is ignored — useful for de-duplication)
  ```

- Images are converted to grayscale and resized to **128×32** internally, so
  the source size doesn't matter.
- Files whose label length or characters don't match the training config are
  skipped (with a count printed), so mixing stray files is harmless.

## Training

Build the trainer once:

```bash
cargo build --release -p capburn-train
# binary: ./target/release/capburn
```

Basic run (auto-detects charset and length from the dataset):

```bash
./target/release/capburn train --data ./data --dataset digits5 --epochs 30
```

Common flags:

| Flag           | Default        | Meaning                                                        |
|----------------|----------------|----------------------------------------------------------------|
| `--data`       | `./data`       | Root folder with datasets.                                     |
| `--dataset`    | —              | Subfolder name inside `--data` (omit to use `--data` directly).|
| `--out`        | `./artifacts`  | Where to write the model and logs.                             |
| `--charset`    | `auto`         | Charset spec (see below).                                      |
| `--num-chars`  | auto           | Length: `5` (fixed) or `4-7` (range); detected if omitted.     |
| `--arch`       | `auto`         | `fixed`, `ctc`, or `auto` (fixed for one length, ctc for a range). |
| `--backend`    | `wgpu`         | `wgpu` (GPU), `cpu`, or `cuda`.                                 |
| `--epochs`     | `30`           | Number of training epochs.                                     |
| `--batch-size` | `64`           | Batch size.                                                    |
| `--lr`         | `0.0005`       | Adam learning rate.                                            |
| `--no-augment` | off            | Disable training-image augmentation (affine + photometric).    |

### Architectures (`--arch`)

Both heads share one CNN backbone (input is resized to grayscale **128×32**);
no RNN is used, so training and CPU inference stay fast.

| `--arch` | Objective                         | Best for                                              |
|----------|-----------------------------------|-------------------------------------------------------|
| `fixed`  | Positional slots + cross-entropy  | Fixed-length captchas (e.g. exactly 5 digits). Strong, stable, handles repeats like `00075`. |
| `ctc`    | Per-column classifier + CTC loss  | Variable length / shifting positions (e.g. 4–7 chars). |
| `auto`   | picks `fixed` or `ctc`            | A single detected length → `fixed`; a range → `ctc`.  |

Training augments each image (small rotation/scale/shift, brightness/contrast,
noise) to improve generalization on small datasets; disable with `--no-augment`.
Each epoch reports training loss, per-character accuracy and full-captcha
accuracy on a held-out split, and only the best epoch's model is saved.

### Backends

| `--backend` | Hardware                              | Notes                                            |
|-------------|---------------------------------------|--------------------------------------------------|
| `wgpu`      | GPU: Metal / Vulkan / DX12            | Default. No vendor SDK required.                 |
| `cpu`       | CPU (ndarray)                         | Always works; slow — good for smoke tests.       |
| `cuda`      | NVIDIA GPU                            | Requires building with `--features cuda`.        |

CUDA build:

```bash
cargo build --release -p capburn-train --features cuda
./target/release/capburn train --backend cuda --data ./data --dataset digits5 --epochs 30
```

### Charset and length

`--charset` accepts named sets, combinations (joined by `+`), or literal
characters:

| Spec                | Expands to                                   |
|---------------------|----------------------------------------------|
| `digits`            | `0-9`                                         |
| `lower`             | `a-z`                                         |
| `upper`             | `A-Z`                                         |
| `letters`           | `a-z` + `A-Z`                                 |
| `cyrillic`          | `а-я` + `ё` (lowercase Cyrillic)              |
| `cyrillic_upper`    | `А-Я` + `Ё` (uppercase Cyrillic)             |
| `cyrillic+digits`   | lowercase Cyrillic + digits                   |
| `ABCDEF0123456789`  | exactly those characters (e.g. hex)           |

Examples:

```bash
# 5 digits, GPU
./target/release/capburn train --dataset digits5 --charset digits --num-chars 5

# 4 characters, lowercase Cyrillic + digits
./target/release/capburn train --dataset cyr4 --charset cyrillic+digits --num-chars 4

# 6 uppercase Latin letters
./target/release/capburn train --dataset codes --charset upper --num-chars 6
```

### Auto-detection

With `--charset auto` (the default) the trainer scans the file names and:

- picks the **most common label length** as `--num-chars` (unless you set it),
- builds the charset **by families**: if any digit appears, all digits are
  included; if any lowercase Latin letter appears, all of `a-z`; if any Russian
  letter appears, all of that case's Cyrillic — and so on.

In other words: *one Russian letter in the dataset means "all Russian letters";
one English letter means "all English letters".* The detected format is printed
before training starts.

### Training on a remote server

The trainer runs fully **headless** — when stdout is not a terminal it
automatically switches from the TUI dashboard to plain line logging, so it works
under `nohup`, `tmux`, `systemd`, CI, etc.

Typical workflow on a training box:

```bash
# 1. Get the code and build
git clone https://github.com/eav93/capburn.git
cd capburn
cargo build --release -p capburn-train        # add --features cuda for NVIDIA

# 2. Copy your dataset into data/<name>/ (datasets are not in git)
mkdir -p data/digits5
rsync -a you@host:/path/to/images/ data/digits5/

# 3. Train in the background, detached from the SSH session
nohup ./target/release/capburn train \
  --data ./data --dataset digits5 \
  --backend wgpu --epochs 40 --batch-size 128 \
  --out ./artifacts/digits5 > train.log 2>&1 &

# 4. Watch progress
tail -f train.log
#   ...or the detailed log:
tail -f artifacts/digits5/experiment.log

# 5. When done, fetch the model (just two files)
scp you@host:'.../artifacts/digits5/model.mpk' .../artifacts/digits5/model.json  ./
```

> **Note:** the `wgpu` backend needs a real GPU on the server. On a headless
> Linux GPU box that means a working Vulkan driver. If the machine has no GPU,
> use `--backend cpu` (slower) or `--backend cuda` with an NVIDIA card.

## Inference from the CLI

Quickly test a trained model (CPU inference, no GPU needed):

```bash
./target/release/capburn infer ./captcha.png --artifacts ./artifacts/digits5
# Predicted: 12345
```

## PHP extension

The extension exposes a single class, `Capburn\Ext\Recognizer`, that loads a
trained model and recognizes captchas. No Composer package is needed.

### Build

```bash
cargo build --release -p capburn_php
# Linux:  target/release/libcapburn_php.so
# macOS:  target/release/libcapburn_php.dylib
```

Prebuilt binaries for each PHP version × platform are attached to every
[GitHub Release](https://github.com/eav93/capburn/releases).

### Quick install

Download the prebuilt extension for your PHP version and platform in one line —
the script prints the exact `-d extension=…` command to run:

```bash
curl -fsSL https://raw.githubusercontent.com/eav93/capburn/main/install-capburn.sh | bash
# → ./capburn/capburn_php.so, then:
#   php -d extension=/abs/path/capburn/capburn_php.so your-script.php
```

Pick a directory or version with flags:

```bash
curl -fsSL https://raw.githubusercontent.com/eav93/capburn/main/install-capburn.sh \
  | bash -s -- --dest /usr/local/lib/php --version v0.1.0
```

In a Dockerfile:

```dockerfile
RUN curl -fsSL https://raw.githubusercontent.com/eav93/capburn/main/install-capburn.sh \
      | bash -s -- --dest /tmp/capburn \
    && cp /tmp/capburn/capburn_php.so "$(php-config --extension-dir)/capburn_php.so" \
    && docker-php-ext-enable capburn_php
```

### Install manually

Either load it ad-hoc:

```bash
php -d extension=/abs/path/to/libcapburn_php.so your-script.php
```

…or copy it into PHP's extension directory and enable it in `php.ini`:

```bash
cp target/release/libcapburn_php.so "$(php-config --extension-dir)/capburn_php.so"
echo "extension=capburn_php.so" >> "$(php --ini | grep 'Loaded' | awk '{print $NF}')"
php -m | grep -i capburn     # should print the module
```

### Usage

```php
<?php
use Capburn\Ext\Recognizer;

// Point at the folder that contains model.json + model.mpk
$rec = new Recognizer('/srv/models/digits5');

echo $rec->numChars();                 // 5
echo Recognizer::extensionVersion();   // e.g. 0.1.0

// From a file
echo $rec->recognize('/tmp/captcha.png');

// From raw bytes (e.g. an HTTP response body)
$bytes = file_get_contents('/tmp/captcha.png');
echo $rec->recognizeBytes($bytes);

// From base64 (plain or a data-URL)
echo $rec->recognizeBase64(base64_encode($bytes));
echo $rec->recognizeBase64('data:image/png;base64,' . base64_encode($bytes));
```

A runnable example lives in
[`crates/capburn-php/examples/basic.php`](crates/capburn-php/examples/basic.php),
and IDE stubs in
[`crates/capburn-php/stubs/Capburn.stubs.php`](crates/capburn-php/stubs/Capburn.stubs.php).

### Deploying the model

The PHP extension only needs the two model files — not the dataset:

```
model.json    # charset + captcha length + architecture config
model.mpk     # trained weights (MessagePack)
```

Copy them to the server, point `Recognizer` at their folder, done.

## Model artifacts

After training, `--out` contains:

| File               | Purpose                                                       |
|--------------------|---------------------------------------------------------------|
| `model.mpk`        | Trained weights (used by the trainer and the PHP extension).  |
| `model.json`       | Charset, length and architecture — read on load.              |
| `config.json`      | Full training config (for reproducibility).                   |
| `experiment.log`   | Detailed training log.                                        |
| `checkpoint/`      | Per-epoch checkpoints.                                        |

Only `model.mpk` + `model.json` are needed to run inference.

## CI / Releases

- **`.github/workflows/ci.yml`** — on every push/PR: `cargo fmt`, `clippy`,
  builds the trainer and the extension on Linux + macOS across PHP 8.1/8.5, and
  smoke-tests that the extension loads.
- **`.github/workflows/release.yml`** — on a `v*` tag: builds the trainer
  binary (Linux x86_64/aarch64, macOS aarch64) **and** the PHP extension for
  each PHP 8.1–8.5 × platform, then attaches everything to the GitHub Release.

Cut a release:

```bash
git tag v0.1.0
git push origin v0.1.0
```

---
---

# Русская версия

Распознаватель капчи на [Burn](https://burn.dev). Один Rust-workspace с двумя
результатами:

- **`capburn`** — CLI-обучалка (GPU/CPU), которая учится читать капчу.
- **`capburn_php`** — нативное PHP-расширение, которое загружает обученную
  модель и распознаёт капчу прямо из PHP. Composer-обёртка не нужна.

Алфавит и длина капчи — **параметры времени выполнения**: можно обучать модель
на цифрах, кириллице + цифрах, фиксированной длине и т. д., либо позволить
инструменту определить формат по датасету автоматически.

## Структура проекта

```
capburn/
├── crates/
│   ├── capburn-core/     # модель, алфавит, препроцессинг, CPU-инференс (без привязки к бэкенду)
│   ├── capburn-train/    # CLI-обучалка (bin: `capburn`), бэкенды wgpu/cpu/cuda
│   └── capburn-php/      # PHP-расширение (cdylib), CPU-инференс через ndarray
├── data/                 # датасеты (в git не коммитятся) — по подпапке на датасет
├── artifacts/            # обученные модели (в git не коммитятся)
└── .github/workflows/    # CI + сборка релизов
```

`capburn-core` — общее ядро: определение сети и инференс не зависят от бэкенда,
поэтому обучалка и PHP-расширение загружают **одну и ту же модель**.

## Требования

- **Rust** stable (1.85+). Установка через [rustup](https://rustup.rs).
- Для PHP-расширения: **PHP 8.1–8.5** с заголовками разработчика (`php-config`
  в `PATH`). На Debian/Ubuntu: `apt install php-dev`.
- Для обучения на GPU: видеокарта с поддержкой Metal (macOS) или Vulkan
  (Linux/Windows) — отдельный SDK не нужен благодаря `wgpu`. Для NVIDIA CUDA —
  CUDA toolkit.

## Формат датасета

- Каждый датасет лежит в **своей подпапке** внутри `data/`, например
  `data/digits5/`, `data/cyrillic4/`.
- Одно изображение на капчу (PNG/JPEG/…). **Метка берётся из имени файла** —
  часть **до первого `_`**:

  ```
  12345.png              → метка "12345"
  12345_a1b2c3.png       → метка "12345"   (суффикс после "_" игнорируется — удобно для дедупликации)
  ```

- Изображения внутри приводятся к серому и размеру **128×32**, так что исходный
  размер не важен.
- Файлы, у которых длина метки или символы не совпадают с конфигом обучения,
  пропускаются (с выводом количества) — посторонние файлы не мешают.

## Обучение

Собрать обучалку один раз:

```bash
cargo build --release -p capburn-train
# бинарник: ./target/release/capburn
```

Базовый запуск (алфавит и длина определяются по датасету автоматически):

```bash
./target/release/capburn train --data ./data --dataset digits5 --epochs 30
```

Основные флаги:

| Флаг           | По умолчанию   | Значение                                                       |
|----------------|----------------|----------------------------------------------------------------|
| `--data`       | `./data`       | Корневая папка с датасетами.                                    |
| `--dataset`    | —              | Имя подпапки в `--data` (без него используется сам `--data`).   |
| `--out`        | `./artifacts`  | Куда сохранять модель и логи.                                   |
| `--charset`    | `auto`         | Спецификация алфавита (см. ниже).                               |
| `--num-chars`  | авто           | Длина: `5` (фикс.) или `4-7` (диапазон); иначе по датасету.     |
| `--arch`       | `auto`         | `fixed`, `ctc` или `auto` (fixed для одной длины, ctc для диапазона). |
| `--backend`    | `wgpu`         | `wgpu` (GPU), `cpu` или `cuda`.                                 |
| `--epochs`     | `30`           | Число эпох обучения.                                            |
| `--batch-size` | `64`           | Размер батча.                                                   |
| `--lr`         | `0.0005`       | Скорость обучения Adam.                                         |
| `--no-augment` | выкл           | Отключить аугментацию тренировочных изображений.               |

### Архитектуры (`--arch`)

Оба режима используют общий CNN-backbone (вход приводится к серому **128×32**);
RNN не используется — обучение и CPU-инференс быстрые.

| `--arch` | Что делает                        | Для чего лучше                                        |
|----------|-----------------------------------|-------------------------------------------------------|
| `fixed`  | Позиционные слоты + cross-entropy | Фиксированная длина (например ровно 5 цифр). Стабильно, держит повторы вроде `00075`. |
| `ctc`    | Классификатор по столбцам + CTC   | Переменная длина / плавающие позиции (например 4–7 символов). |
| `auto`   | выбирает `fixed` или `ctc`        | Одна длина → `fixed`; диапазон → `ctc`.               |

Каждое изображение аугментируется (небольшие поворот/масштаб/сдвиг, яркость/
контраст, шум) для лучшей генерализации на маленьких датасетах (`--no-augment`
отключает). Каждая эпоха печатает loss, per-char и full-captcha точность на
отложенной выборке; на диск сохраняется только лучшая эпоха.

### Бэкенды (GPU / CPU / CUDA)

| `--backend` | Железо                          | Примечания                                       |
|-------------|---------------------------------|--------------------------------------------------|
| `wgpu`      | GPU: Metal / Vulkan / DX12      | По умолчанию. Не требует SDK производителя.       |
| `cpu`       | CPU (ndarray)                   | Работает всегда; медленно — для проверок.         |
| `cuda`      | NVIDIA GPU                      | Требует сборки с `--features cuda`.               |

Сборка с CUDA:

```bash
cargo build --release -p capburn-train --features cuda
./target/release/capburn train --backend cuda --data ./data --dataset digits5 --epochs 30
```

### Алфавит и длина

`--charset` принимает именованные наборы, комбинации (через `+`) или явные
символы:

| Спецификация        | Разворачивается в                            |
|---------------------|----------------------------------------------|
| `digits`            | `0-9`                                         |
| `lower`             | `a-z`                                         |
| `upper`             | `A-Z`                                         |
| `letters`           | `a-z` + `A-Z`                                 |
| `cyrillic`          | `а-я` + `ё` (строчная кириллица)              |
| `cyrillic_upper`    | `А-Я` + `Ё` (заглавная кириллица)            |
| `cyrillic+digits`   | строчная кириллица + цифры                    |
| `ABCDEF0123456789`  | ровно эти символы (например hex)              |

Примеры:

```bash
# 5 цифр, GPU
./target/release/capburn train --dataset digits5 --charset digits --num-chars 5

# 4 символа, строчная кириллица + цифры
./target/release/capburn train --dataset cyr4 --charset cyrillic+digits --num-chars 4

# 6 заглавных латинских букв
./target/release/capburn train --dataset codes --charset upper --num-chars 6
```

### Автоопределение

При `--charset auto` (по умолчанию) обучалка сканирует имена файлов и:

- берёт **самую частую длину метки** как `--num-chars` (если вы её не задали);
- собирает алфавит **по семействам**: встретилась цифра — включаются все цифры;
  встретилась строчная латинская буква — весь `a-z`; встретилась русская буква —
  вся кириллица нужного регистра, и т. д.

Иными словами: *одна русская буква в датасете означает «все русские буквы»; одна
английская — «все английские»*. Определённый формат печатается перед стартом
обучения.

### Обучение на удалённом сервере

Обучалка работает полностью **headless** — если stdout не терминал, она
автоматически переключается с TUI-дашборда на построчный лог, поэтому работает
под `nohup`, `tmux`, `systemd`, в CI и т. д.

Типовой сценарий на сервере для обучения:

```bash
# 1. Забрать код и собрать
git clone https://github.com/eav93/capburn.git
cd capburn
cargo build --release -p capburn-train        # для NVIDIA добавьте --features cuda

# 2. Скопировать датасет в data/<name>/ (датасеты не в git)
mkdir -p data/digits5
rsync -a you@host:/path/to/images/ data/digits5/

# 3. Запустить обучение в фоне, отвязав от SSH-сессии
nohup ./target/release/capburn train \
  --data ./data --dataset digits5 \
  --backend wgpu --epochs 40 --batch-size 128 \
  --out ./artifacts/digits5 > train.log 2>&1 &

# 4. Следить за прогрессом
tail -f train.log
#   ...или подробный лог:
tail -f artifacts/digits5/experiment.log

# 5. По завершении забрать модель (всего два файла)
scp you@host:'.../artifacts/digits5/model.mpk' .../artifacts/digits5/model.json  ./
```

> **Важно:** бэкенду `wgpu` нужна реальная GPU на сервере. На headless-сервере с
> Linux это значит рабочий драйвер Vulkan. Если GPU нет — используйте
> `--backend cpu` (медленнее) или `--backend cuda` с картой NVIDIA.

## Инференс из CLI

Быстро проверить обученную модель (инференс на CPU, GPU не нужна):

```bash
./target/release/capburn infer ./captcha.png --artifacts ./artifacts/digits5
# Predicted: 12345
```

## PHP-расширение

Расширение регистрирует один класс `Capburn\Ext\Recognizer`, который загружает
обученную модель и распознаёт капчу. Composer-пакет не нужен.

### Сборка

```bash
cargo build --release -p capburn_php
# Linux:  target/release/libcapburn_php.so
# macOS:  target/release/libcapburn_php.dylib
```

Готовые бинарники под каждую версию PHP и платформу прикладываются к каждому
[релизу на GitHub](https://github.com/eav93/capburn/releases).

### Установка

Либо разово при запуске:

```bash
php -d extension=/abs/path/to/libcapburn_php.so your-script.php
```

…либо скопировать в каталог расширений PHP и включить в `php.ini`:

```bash
cp target/release/libcapburn_php.so "$(php-config --extension-dir)/capburn_php.so"
echo "extension=capburn_php.so" >> "$(php --ini | grep 'Loaded' | awk '{print $NF}')"
php -m | grep -i capburn     # должен показать модуль
```

### Использование

```php
<?php
use Capburn\Ext\Recognizer;

// Папка, где лежат model.json + model.mpk
$rec = new Recognizer('/srv/models/digits5');

echo $rec->numChars();                 // 5
echo Recognizer::extensionVersion();   // например 0.1.0

// Из файла
echo $rec->recognize('/tmp/captcha.png');

// Из сырых байтов (например тело HTTP-ответа)
$bytes = file_get_contents('/tmp/captcha.png');
echo $rec->recognizeBytes($bytes);

// Из base64 (обычного или data-URL)
echo $rec->recognizeBase64(base64_encode($bytes));
echo $rec->recognizeBase64('data:image/png;base64,' . base64_encode($bytes));
```

Готовый пример — в
[`crates/capburn-php/examples/basic.php`](crates/capburn-php/examples/basic.php),
стабы для IDE — в
[`crates/capburn-php/stubs/Capburn.stubs.php`](crates/capburn-php/stubs/Capburn.stubs.php).

### Развёртывание модели

PHP-расширению нужны только два файла модели (датасет не нужен):

```
model.json    # алфавит + длина капчи + конфиг архитектуры
model.mpk     # обученные веса (MessagePack)
```

Скопируйте их на сервер, укажите папку в `Recognizer` — готово.

## Артефакты модели

После обучения в `--out` лежат:

| Файл               | Назначение                                                    |
|--------------------|---------------------------------------------------------------|
| `model.mpk`        | Обученные веса (нужны обучалке и PHP-расширению).             |
| `model.json`       | Алфавит, длина и архитектура — читаются при загрузке.         |
| `config.json`      | Полный конфиг обучения (для воспроизводимости).               |
| `experiment.log`   | Подробный лог обучения.                                       |
| `checkpoint/`      | Чекпоинты по эпохам.                                          |

Для инференса нужны только `model.mpk` + `model.json`.

## CI / Релизы

- **`.github/workflows/ci.yml`** — на каждый push/PR: `cargo fmt`, `clippy`,
  сборка обучалки и расширения на Linux + macOS для PHP 8.1/8.5 и smoke-тест
  загрузки расширения.
- **`.github/workflows/release.yml`** — на тег `v*`: сборка бинарника обучалки
  (Linux x86_64/aarch64, macOS aarch64) **и** PHP-расширения под каждую пару
  PHP 8.1–8.5 × платформа, всё прикладывается к GitHub Release.

Выпустить релиз:

```bash
git tag v0.1.0
git push origin v0.1.0
```
