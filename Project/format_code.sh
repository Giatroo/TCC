#!/bin/bash

# This script formats all the code inside this directory and any
# subdirectories recursively using the black command.

echo Formatting with black...
python -m black . --line-length=80

echo
echo Sorting imports with isort...
python -m isort .
