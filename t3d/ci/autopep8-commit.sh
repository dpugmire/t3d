#!/bin/bash
set -e

if [[ `git status --porcelain --untracked-files=no` ]]; then
    git add .
    git commit -m "[skip ci] Formatting update from autopep8"
    git push
fi
