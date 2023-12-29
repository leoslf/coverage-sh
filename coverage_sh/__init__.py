from __future__ import annotations

from typing import Any

from .plugin import ShellPlugin


def coverage_init(reg, options: dict[str, Any]) -> None:
    shell_plugin = ShellPlugin(options)
    reg.add_file_tracer(shell_plugin)
    reg.add_configurer(shell_plugin)
