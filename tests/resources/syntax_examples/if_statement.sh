#!/usr/bin/env bash
VARIABLE="${VARIABLE:-0}"

if [[ "$VARIABLE" -eq 0 ]]; then
	# comments should not break branch coverage
	echo "if.statements[0]"
	# comments should not break branch coverage
	# comments should not break branch coverage
	echo "if.statements[1]"
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
	echo "if.statements[2]"
fi

