#!/usr/bin/env bash

i=0
# comments should not break branch coverage
while (( i < 5 )); do
	# comments should not break branch coverage
	echo "Hello World $i"
	# comments should not break branch coverage
	(( i += 5 ))	
	# comments should not break branch coverage
done

