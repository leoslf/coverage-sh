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
	echo "Variable is set to 'Hello, World!'"
else
	echo "Variable is not set to 'Hello, World!'" # pragma: no cover
fi

# C-style for loop
for ((i=0; i<10; ++i)); do
	echo "Iteration $i"
done

# for loop
for i in {1..5}; do
	echo "Iteration $i"
done

# while loop
i=0
while (( i < 10 )); do
	echo "Iteration $i"
	(( ++i ))
done

# Functions
function say_hello() {
	echo "Hello from a function!"
}

say_hello

say_bye() {
	echo "Bye from a function!"
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
		echo "You selected an apple."
		;;
	"banana")
		echo "You selected a banana."
		;;
	"orange")
		echo "You selected an orange."
		;;
	"fallthrough-;&")
		;&
	"fallthrough-;;&")
		;;&
	*)
		echo "Unknown fruit."
		;;
esac

