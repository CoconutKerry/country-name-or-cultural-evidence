PYTHON ?= python3

.PHONY: reproduce analyze figure metadata verify test validate-inference hashes clean-generated

reproduce: analyze figure metadata verify test

analyze:
	PYTHONPATH=src $(PYTHON) scripts/reproduce_analysis.py

figure:
	PYTHONPATH=src $(PYTHON) scripts/make_figure2.py

metadata:
	$(PYTHON) scripts/export_release_metadata.py

verify:
	PYTHONPATH=src $(PYTHON) scripts/verify_release.py

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

validate-inference:
	$(PYTHON) scripts/validate_inference_sources.py

hashes:
	$(PYTHON) scripts/generate_hash_manifests.py

clean-generated:
	rm -rf results/processed results/figures results/latex
	rm -f results/summary/RESULTS.md results/summary/analysis_manifest.json
