#!/usr/bin/env bash

VARIABLE="${VARIABLE:-Hello, World!}"

# comment
if [ "${VARIABLE?}" != "Hello, World!" ]; then
	# comments should not break branch coverage
	echo "Unreachable, should not match"
	# comments should not break branch coverage
elif false; then
	# comments should not break branch coverage
	# comments should not break branch coverage
	echo "Unreachable, always false"
	# comments should not break branch coverage
elif true; then
	# comments should not break branch coverage
	echo "Hello, World!"
	echo "Hello, World!"
	# comments should not break branch coverage
	# comments should not break branch coverage
else
	# comments should not break branch coverage
	echo "Unreachable"
	# comments should not break branch coverage
	# comments should not break branch coverage
	echo "Unreachable"
	# comments should not break branch coverage
fi

true # pragma: no cover

