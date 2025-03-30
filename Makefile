.PHONY: venv install run clean

venv:
	python3 -m venv venv

install: venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt

run: install
	./venv/bin/python extreme_sprocket_overlay.py

clean:
	rm -rf venv
