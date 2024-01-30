.PHONY: install
install :
	@pip install poetry
	@poetry install --no-root

.PHONY: run
run :
	@poetry run streamlit run main.py
