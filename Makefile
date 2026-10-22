SHELL := /bin/bash
.PHONY: setup build doctor test smoke bench
setup:
	bash scripts/bootstrap.sh
build:
	bash scripts/build.sh
doctor:
	source scripts/env.sh && python -m bench.doctor
test:
	source scripts/env.sh && python -m pytest -q
smoke:
	source scripts/env.sh && python -m bench.run --task 00-vector-add --size 4099 --samples 5 --output results/smoke.json
bench:
	source scripts/env.sh && python -m bench.run --task 01-softmax --rows 256 --cols 1024 --output results/softmax.json
