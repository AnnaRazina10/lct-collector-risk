# Настройка проекта

## Требования

Рекомендуемое окружение:

- macOS или Linux;
- Python 3.11+;
- Git;
- `bsdtar` или `7z` для просмотра и распаковки `.7z` архивов;
- PostgreSQL 12+ для промышленной версии или SQLite/DuckDB для локального прототипа.

## Подготовка репозитория

```bash
cd /Users/annarazina/Desktop/CT2/lct-collector-risk
git status
```

Папка `data/raw/` должна быть в `.gitignore`. Проверка:

```bash
cat .gitignore
git status
```

В `git status` не должны отображаться большие файлы из `data/raw/`.

## Локальные данные

Если данные нужно скопировать заново:

```bash
mkdir -p data/raw
cp -R "/Users/annarazina/Desktop/CT2/LCT/dataset/"* data/raw/
```

Проверка:

```bash
ls -lh data/raw
```

Ожидаемые файлы:

```text
ext-journal-2019.7z
ext-journal-2020.7z
ext-journal-2021.7z
ext-journal-2022.7z
ext-journal-2023.7z
ext-journal-2024.7z
ext-journal-2025.7z
ext-journal-2026.7z
журнал_событий_пример.csv
справочник_каналов_датчиков.csv
справочник_объектов_диспетчер.csv
```

## Просмотр содержимого архивов

```bash
bsdtar -tf data/raw/ext-journal-2026.7z
```

Чтение первых строк CSV внутри архива без распаковки:

```bash
bsdtar -xOf data/raw/ext-journal-2026.7z ext-journal-2026.csv | head
```

Если `bsdtar` не умеет читать архив на конкретной машине, установите `p7zip` и
используйте `7z`.

## Рекомендуемая Python-среда

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Первый набор библиотек для EDA и baseline:

```bash
python -m pip install pandas polars pyarrow scikit-learn catboost lightgbm matplotlib seaborn fastapi uvicorn
```

После появления стабильного списка зависимостей нужно зафиксировать его в
`requirements.txt` или `pyproject.toml`.

## GitHub

Большие данные не коммитятся. Для первого коммита:

```bash
git add .gitignore README.md docs
git commit -m "Add competition documentation"
```

Если нужно сохранить пустые папки в Git:

```bash
touch src/.gitkeep api/.gitkeep web/.gitkeep notebooks/.gitkeep data/samples/.gitkeep
git add src api web notebooks data/samples
git commit -m "Add project directories"
```

Создание приватного репозитория через GitHub CLI:

```bash
gh repo create lct-collector-risk --private --source=. --remote=origin
git push -u origin main
```

Если `gh` не настроен, создайте приватный репозиторий на GitHub вручную и выполните
команды, которые GitHub покажет для existing repository.
