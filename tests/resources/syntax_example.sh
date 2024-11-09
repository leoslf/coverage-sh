#!/usr/bin/env bash

#
# SPDX-License-Identifier: MIT
# Copyright (c) 2023-2024 Kilian Lackhove
#

# This is an extended Bash script with some common syntax elements including a case statement. It was created by ChatGPT
# with minor manual modifications

# Variable assignment
variable="Hello, World!"

# Printing variables
echo $variable

# Conditionals
if [ "$variable" == "Hello, World!" ]; then
	# comments should not break branch coverage
	echo "Variable is set to 'Hello, World!'"
	# comments should not break branch coverage
	# comments should not break branch coverage
elif false; then
	# comments should not break branch coverage
	# comments should not break branch coverage
	echo "Unreachable"
	# comments should not break branch coverage
	echo "Unreachable"
	echo "Unreachable"
	echo "Unreachable"
	# comments should not break branch coverage
else
	# comments should not break branch coverage
	echo "Variable is not set to 'Hello, World!'"
	# comments should not break branch coverage
fi

if [ "$variable" != "Hello, World!" ]; then
	# comments should not break branch coverage
	echo "Unreachable"
	# comments should not break branch coverage
	# comments should not break branch coverage
elif [ "$variable" == "Hello, World!" ]; then
	# comments should not break branch coverage
	# comments should not break branch coverage
	echo "Variable is set to 'Hello, World!'"
	# comments should not break branch coverage
	# comments should not break branch coverage
else
	# comments should not break branch coverage
	echo "Variable is not set to 'Hello, World!'"
	# comments should not break branch coverage
fi

if true; then
	echo "always true"
fi

if false; then
	echo "unreachable" # pragma: no cover
fi

# C-style for loop
for ((i=0; i<10; ++i)); do
	# comments should not break branch coverage
	echo "Iteration $i"
	# comments should not break branch coverage
done

# C-style for loop - branch missing
for ((i=0; i<0; ++i)); do
	# comments should not break branch coverage
	echo "Should not enter this block - Iteration $i"
	# comments should not break branch coverage
done

# for loop
for i in {1..5}; do
	# comments should not break branch coverage
	echo "Iteration $i"
	# comments should not break branch coverage
done

shopt -s nullglob
# for loop - branch missing
for i in /non-existent/*; do
	# comments should not break branch coverage
	echo "Should not enter this block - Iteration $i"
	# comments should not break branch coverage
done


# while loop
i=0
while (( i < 10 )); do
	# comments should not break branch coverage
	echo "Iteration $i"
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
	(( ++i ))
	# comments should not break branch coverage
done

# while loop - branch missing
i=0
# comments should not break branch coverage
while (( i < 0 )); do
	# comments should not break branch coverage
	echo "Should not enter this block - Iteration $i"
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
	(( ++i ))
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
done

# while loop - branch missing
while false; do
	# comments should not break branch coverage
	echo "Should not enter this block"
	# comments should not break branch coverage
done

# Functions
function say_hello() {
	# comments should not break branch coverage
	echo "Hello from a function!"
	# comments should not break branch coverage
}

# comments should not break branch coverage
say_hello
# comments should not break branch coverage
# comments should not break branch coverage

say_bye() {
	# comments should not break branch coverage
	echo "Bye from a function!"
	# comments should not break branch coverage
	# comments should not break branch coverage
	# comments should not break branch coverage
}

say_bye

# Command substitution
os=$(uname)
echo "Current OS is: $os"

# Arithmetic operations
result=$((5 + 3))
echo "5 + 3 = $result"

# Ternary expression
(( result = result == 8 ? 1 : 0 ))
echo "result == 8 ? 1 : 0 = $result"

# Negated command
! test 1 -eq 0

# compound statement
[[ 1 == 0 ]] || [ 2 = 0 ] || echo "testing"

# File operations
touch example_file.txt
echo "This is a sample file." > example_file.txt
cat example_file.txt
rm -f example_file.txt

# Case statement
fruit="banana"
case $fruit in
	"apple")
		# comments should not break branch coverage
		echo "You selected an apple."
		# comments should not break branch coverage
		;;
	"banana")
		# comments should not break branch coverage
		echo "You selected a banana."
		# comments should not break branch coverage
		;;
	"orange")
		# comments should not break branch coverage
		echo "You selected an orange."
		# comments should not break branch coverage
		# comments should not break branch coverage
		;;
	"fallthrough-;&")
		# comments should not break branch coverage
		;&
	"fallthrough-;;&")
		# comments should not break branch coverage
		;;&
	*)
		# comments should not break branch coverage
		echo "Unknown fruit."
		# comments should not break branch coverage
		# comments should not break branch coverage
		;;
esac
# comments should not break branch coverage


# for loop
i=0
for ((;;)); do
	if (( i > 10 )); then
		break
	fi
	(( ++i ))
done
