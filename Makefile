LINTING_DIRS_WITHOUT_DWARN := ./src       # Linterの対象ディレクトリまたはファイル(Docstringチェックを行わない)
LINTING_DIRS_WITH_DWARN := ./src/main.py  # Linterの対象ディレクトリまたはファイル(Docstringチェックを行う)
EXCLUED_LINTING_DIRS := ./src/clip        # CLIPは既存のコードなので、lintingしない

.PHONY: black-check
black-check:
	poetry run black --check $(LINTING_DIRS_WITH_DWARN)

.PHONY: black
black:
	poetry run black $(LINTING_DIRS_WITH_DWARN)

.PHONY: flake8
flake8:
	poetry run flake8 --config=.flake8 --extend-ignore=D102,D103 $(LINTING_DIRS_WITHOUT_DWARN) --exclude $(EXCLUED_LINTING_DIRS)
	poetry run flake8 --config=.flake8 $(LINTING_DIRS_WITH_DWARN)

.PHONY: isort-check
isort-check:
	poetry run isort --check-only $(LINTING_DIRS_WITH_DWARN)

.PHONY: isort
isort:
	poetry run isort $(LINTING_DIRS_WITH_DWARN)

.PHONY: mypy
mypy:
	poetry run mypy $(LINTING_DIRS_WITH_DWARN)

.PHONY: format
format:
	$(MAKE) black
	$(MAKE) isort

.PHONY: lint
lint:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) flake8
	$(MAKE) mypy

.PHONY: lint-min
lint-min:
	$(MAKE) black-check
	$(MAKE) isort-check
	$(MAKE) flake8
