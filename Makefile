SHELL := /bin/bash

venv:
	python3 -m venv rl_learn && \
	. /rl_learn/bin/activate

install:
	pip install -r requirements.txt