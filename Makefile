PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

.PHONY: help venv test contract plan deploy docker clean

help:
	@echo "Laptop:"
	@echo "  make venv      create .venv, install dattn + this package"
	@echo "  make test      correctness suite (no GPU required)"
	@echo "  make contract  anti-decoration suite - fails if a technology is decorative"
	@echo ""
	@echo "  Most contract checks are GPU-gated and will SKIP here. Skips are"
	@echo "  reported, never silently passed - see tests/test_contract.py."
	@echo ""
	@echo "GPU box:"
	@echo "  make deploy PORT=.. HOST=.. [VERB=all]   sync, run, pull results back"
	@echo "  make docker                              build the NGC-based image"

deploy:
	@test -n "$(PORT)" || { echo "usage: make deploy PORT=16094 HOST=1.2.3.4 [VERB=all]"; exit 1; }
	@test -n "$(HOST)" || { echo "usage: make deploy PORT=16094 HOST=1.2.3.4 [VERB=all]"; exit 1; }
	deploy/remote.sh $(PORT) $(HOST) $(or $(VERB),all)

docker:
	docker build -f deploy/Dockerfile -t lis:latest .

venv:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) -e ../Distributed_Attention00 -e ".[dev]"

test:
	$(PY) -m pytest tests/test_rope.py tests/test_lse.py -q

# -rs prints the reason for every skip, so a green CPU run cannot be mistaken
# for a validated stack.
contract:
	$(PY) -m pytest tests/test_contract.py -q -rs

clean:
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__
