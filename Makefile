all: clean build run

build: clean venv
venv: venv/bin/activate
venv/bin/activate: requirements.txt
	test -d venv || virtualenv venv
	venv/bin/pip install -Ur requirements.txt
	touch venv/bin/activate

run: demo
demo: main.py
	venv/bin/python main.py

clean:
	rm -rf venv
	find . -name '*.pyc' -delete