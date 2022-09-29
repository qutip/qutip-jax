# Sphinx flags are:
#   -W : turn warnings into errors
#   --keep-going : do not stop after the first error
#   -T : display a full traceback if a Python exception occurs
SPHINXOPTS    ?= -W --keep-going -T
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Default target just prints the help (also lists all targets).
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

.PHONY: help Makefile

# Every target is just passed directly to sphinx-build to do whatever it wants.
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)
