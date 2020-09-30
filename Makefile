root_dir ?= algorithmselection

lint:
	pylint $(root_dir)
mypy:
	mypy $(root_dir)

install:
	pip install -e ".[test]"



