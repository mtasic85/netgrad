# netgrad

A (not even a toy) autograd engine.


## Development

Create venv:
```bash
python -v venv venv
```

For bash/sh shell:
```bash
source venv/bin/activate
```

For fish shell:
```bash
source venv/bin/activate.fish
```

Install requirements:
```bash
pip install -r requirements.txt
```

Run demos:
```bash
python -B netgrad/tensor.py
```

Run tests:
```bash
DEBUG=1 python -B -m pytest -v
```
