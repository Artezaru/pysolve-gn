# Setting default
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build

# variables
venv ?= venv

message ?=
push ?=
branch ?=

level ?=

cleaning ?= false
generating ?= false

# Default help command to list available Sphinx options
help:
	@echo "Available commands:"
	@echo "  help       - Show this help message"
	@echo "  install    - [venv=venv] Install the package in editable mode (pip install -e .) and sphinx dependencies"
	@echo "  clean      - Clean the documentation build directory $(BUILDDIR)"
	@echo "  html       - [cleaning=false, venv=venv] Generate HTML documentation with sphinx-build output at $(BUILDDIR)/html/"
	@echo "  open       - [generating=false, venv=venv] Open the generated HTML at $(BUILDDIR)/html/index.html in the default web browser"
	@echo "  bump       - [level={major/minor/patch}] Update the version of the package"
	@echo "  git        - [message, branch, push:{true/false}] Commit and push changes to given branch with given message. Example: 'make git message=\"Update docs\" branch=main push=true'"
	@echo "  test       - [venv=venv] Run the tests of the package with pytest"

.PHONY: help Makefile

# Install the package in editable mode and sphinx dependencies
install:
	@\
	echo "Connecting to the virtual environment at $(venv)" && \
	. $(venv)/bin/activate && \
	echo "Virtual environment activated." && \
	echo "Installing the package in editable mode..." && \
	pip install -e . && \
	echo "Package installed in editable mode." && \
	echo "Installing sphinx dependencies..." && \
	pip install sphinx pydata-sphinx-theme sphinx-copybutton sphinx-gallery sphinx-design && \
	echo "Sphinx dependencies installed successfully."

# Check the Tests
test:
	@\
	echo "Connecting to the virtual environment at $(venv)" && \
	. $(venv)/bin/activate && \
	echo "Virtual environment activated." && \
	echo "Running tests with pytest..." && \
	pytest tests && \
	echo "Tests completed successfully."

# Update the version of the package
bump:
	@\
	echo "Checking for version update level..." && \
	if [ -z "$(level)" ]; then \
		echo "Error: level variable is not set. Use 'make bump level=patch' (or major/minor)"; \
		exit 1; \
	fi && \
	echo "Updating version with level: $(level)" && \
	bumpver update --$(level) --no-fetch && \
	echo "Version updated successfully."

# Clean the documentation
clean:
	@\
	echo "Cleaning up generated files and build artifacts..." && \
	rm -rf $(SOURCEDIR)/_autosummary && \
	rm -rf $(SOURCEDIR)/_gallery && \
	rm -rf $(SOURCEDIR)/_gallery_backreferences && \
	rm -rf $(SOURCEDIR)/sg_execution_times.rst && \
	$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) && \
	cd $(BUILDDIR) && mkdir -p html && mkdir -p latex && \
	echo "Clean complete."

# Generate HTML documentation
html:
	@\
	echo "Checking if cleaning is required before building..." && \
	if [ "$(cleaning)" = "true" ]; then \
		echo "Cleaning before building..."; \
		$(MAKE) clean; \
	else \
		echo "Skipping cleaning before building."; \
	fi && \
	echo "Connecting to the virtual environment at $(venv)" && \
	. $(venv)/bin/activate && \
	echo "Virtual environment activated." && \
	echo "Generating HTML documentation at $(BUILDDIR)/html/" && \
	$(SPHINXBUILD) -v -b html $(SOURCEDIR) $(BUILDDIR)/html && \
	echo "HTML documentation generated successfully at $(BUILDDIR)/html/"

# Open the generated HTML documentation in the default web browser
open:
	@\
	echo "Checking if generating is required before opening..." && \
	if [ "$(generating)" = "true" ]; then \
		echo "Generating HTML documentation before opening..."; \
		$(MAKE) html cleaning=true venv=$(venv); \
	else \
		echo "Skipping HTML generation before opening."; \
	fi && \
	echo "Checking if HTML documentation exists at $(BUILDDIR)/html/index.html..." && \
	if [ ! -f "$(BUILDDIR)/html/index.html" ]; then \
		echo "Error: HTML documentation not found at $(BUILDDIR)/html/index.html. Please run 'make html' to generate the documentation first."; \
		exit 1; \
	fi && \
	echo "HTML documentation found. Attempting to open in the default web browser..." && \
	xdg-open $(BUILDDIR)/html/index.html || open $(BUILDDIR)/html/index.html && \
	echo "Documentation opened successfully."

# Git Push origin Branch
git:
	@\
	echo "Checking for required variables: message, branch, push..." && \
	if [ -z "$(message)" ]; then \
		echo "Error: message variable is not set. Use 'make git message=\"Your commit message\"'"; \
		exit 1; \
	fi && \
	if [ -z "$(branch)" ]; then \
		echo "Error: branch variable is not set. Use 'make git branch=\"branch_name\"'"; \
		exit 1; \
	fi && \
	if [ -z "$(push)" ]; then \
		echo "Error: push variable is not set. Use 'make git push=true' to enable pushing or 'make git push=false' to disable pushing"; \
		exit 1; \
	fi && \
	echo "Committing changes with message: $(message) on branch: $(branch)" && \
	git checkout $(branch) && \
	git add -A . && \
	git commit -m "$(message)" && \
	if [ "$(push)" = "true" ]; then \
		echo "Pushing changes to origin $(branch)..."; \
		git push origin $(branch); \
	else \
		echo "Push is disabled. Skipping git push."; \
	fi && \
	echo "Git operations completed successfully."
