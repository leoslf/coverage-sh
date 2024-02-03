#!/bin/sh

#
# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Kilian Lackhove
#

# Variable assignment
variable="Hello, World!"

# Printing variables
echo $variable

# invalid shell syntax
a = b

# invalid syntax mixed with valid syntax
a = b echo $variable