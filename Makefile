VENV = .venv
PYTHON = python3

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt

run:
	. $(VENV)/bin/activate && python -m flask --app app.py --debug run --host=0.0.0.0 --port=8000

clean:
	rm -rf $(VENV)
