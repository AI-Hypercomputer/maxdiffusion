.PHONY: deps_table_update deps_table_check_updated style quality test

# Make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

# Update src/maxdiffusion/dependency_versions_table.py from the generated requirements.

deps_table_update:
	@python3 utils/update_dependency_table.py

deps_table_check_updated:
	@md5sum src/maxdiffusion/dependency_versions_table.py > md5sum.saved
	@python3 utils/update_dependency_table.py
	@md5sum -c --quiet md5sum.saved || (rm -f md5sum.saved; printf "\nError: the version dependency table is outdated.\nPlease run 'make deps_table_update' and commit the changes.\n\n"; exit 1)
	@rm -f md5sum.saved

# Format source code with pyink and lint with pylint (same tools as CI).

style:
	bash code_style.sh

# Check formatting/lint without modifying files.

quality:
	bash code_style.sh --check
	ruff check .

# Run the unit tests (CI additionally skips kernels/ and a few TPU-only tests; see .github/workflows/UnitTests.yml).

test:
	python3 -m pytest src/maxdiffusion/tests
