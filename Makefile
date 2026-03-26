# Setting default
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

message ?=
level ?=
push ?=
branch ?=
cleaning ?= false

# Default help command to list available Sphinx options
help:
	@echo "Available commands:"
	@echo "  help       - Show this help message"
	@echo "  install    - Install the package in editable mode (pip install -e .) and sphinx dependencies (sphinx, pydata-sphinx-theme, sphinx-copybutton, sphinx-gallery, sphinx-design)"
	@echo "  clean      - Clean the documentation build directory docs/build/"
	@echo "  html       - Generate HTML documentation with sphinx-build output at docs/build/html/ (use cleaning=true to clean before building)"
	@echo "  bump       - Update the version of the package (use level=major/minor/patch)"
	@echo "  git        - Commit and push changes to given branch (use message='Your commit message' and branch='branch_name' and push=true/false to push)"
	@echo "  test       - Run the tests of the package with pytest"

.PHONY: help Makefile

# Install the package in editable mode and sphinx dependencies
install:
	@echo "Installing the package in editable mode and sphinx dependencies..."
	pip install -e .
	pip install sphinx pydata-sphinx-theme sphinx-copybutton sphinx-gallery sphinx-design

# Check the Tests
test:
	@. $(VENV)/bin/activate && pytest tests

# Update the version of the package
bump:
	@if [ -z "$(level)" ]; then \
		echo "Error: level variable is not set. Use 'make bump level=patch' (or major/minor)"; \
		exit 1; \
	fi
	@echo "Updating version with level: $(level)"
	bumpver update --$(level) --no-fetch

# Clean the documentation
clean:
	@echo "Cleaning up generated files at docs/source/_autosummary/"
	@rm -rf docs/source/_autosummary
	@echo "Cleaning up generated files at docs/source/_gallery/"
	@rm -rf docs/source/_gallery
	@rm -rf docs/source/_gallery_backreferences
	@rm -rf docs/source/sg_execution_times.rst
	@echo "Cleaning up generated build documentation at docs/build/"
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O);
	@echo "Recreating necessary directories..."
	cd $(BUILDDIR); mkdir -p html; mkdir -p latex
	@echo "Clean complete."

# Generate HTML documentation
html:
	@if [ "$(cleaning)" = "true" ]; then \
		echo "Cleaning before building..."; \
		$(MAKE) clean; \
	fi
	@echo "Generating HTML documentation at $(BUILDDIR)/html/"
	$(SPHINXBUILD) -v -b html $(SOURCEDIR) $(BUILDDIR)/html

# Git Push origin Branch
git:
	@if [ -z "$(message)" ]; then \
		echo "Error: message variable is not set. Use 'make git message=\"Your commit message\"'"; \
		exit 1; \
	fi
	@if [ -z "$(branch)" ]; then \
		echo "Error: branch variable is not set. Use 'make git branch=\"branch_name\"'"; \
		exit 1; \
	fi
	@if [ -z "$(push)" ]; then \
		echo "Error: push variable is not set. Use 'make git push=true' to enable pushing or 'make git push=false' to disable pushing"; \
		exit 1; \
	fi
	@echo "Committing changes with message: $(message) on branch: $(branch)"
	git checkout $(branch)
	git add -A .
	git commit -m "$(message)"
	@if [ "$(push)" = "true" ]; then \
		@echo "Pushing changes to origin $(branch)..."; \
		git push origin $(branch); \
	else \
		@echo "Push is disabled. Skipping git push."; \
	fi