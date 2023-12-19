from .plugin import ShPlugin

def coverage_init(reg, options):
    plugin = ShPlugin(options)
    reg.add_file_tracer(plugin)
    reg.add_configurer(plugin)