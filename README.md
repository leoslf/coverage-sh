# Coverage.sh

A  [Coverage.py](https://github.com/nedbat/coveragepy) plugin to measure code coverage of shell (sh or bash) scripts
executed from python.

## Installation

```shell
pip install coverage-sh
```

## Usage

In your `pyproject.toml`, set

```toml
[tool.coverage.run]
plugins = ["coverage_sh"]
```

and run

```shell
coverage run main.py
coverage combine
coverage html
```

to measure coverage of all shell scripts executed via
the [subprocess](https://docs.python.org/3/library/subprocess.html) module, e.g.:

```python
subprocess.run(["bash", "test.sh"])
```

The resulting coverage is then displayed alongside the coverage of the python files:

![coverage.sh report screenshot](doc/media/screenshot_html-report.png)

## Caveats

The plugin works by patching the `subprocess.Popen` class to alter `sh` and `bash` calls to produce trace files. This
approach comes with a few caveats:

- It will only cover shell scripts that are executed via the subprocess module.
- Only commands that start with    `sh`, `/bin/sh`, `/usr/bin/sh` and the path returned by `which sh` as well as their
  bash equivalents are supported. Custom interpreter paths and other shells are not supported.
- Depending on the shell scripts, their trace files may become very large.

## License

Licensed under the [MIT License](LICENSE.txt).
