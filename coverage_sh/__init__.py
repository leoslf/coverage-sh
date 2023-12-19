from .plugin import ShellPlugin


def coverage_init(reg, options):
    plugin = ShellPlugin(options)
    reg.add_file_tracer(plugin)
    reg.add_configurer(plugin)
