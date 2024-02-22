#  SPDX-License-Identifier: MIT
#  Copyright (c) 2023 Kilian Lackhove

from __future__ import annotations

from typing import Any

from .plugin import ShellPlugin


def coverage_init(reg, options: dict[str, Any]) -> None:  # type: ignore[no-untyped-def] # noqa: ANN001
    shell_plugin = ShellPlugin(options)
    reg.add_file_tracer(shell_plugin)
    reg.add_configurer(shell_plugin)
