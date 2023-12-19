import subprocess

import coverage

coverage.process_startup()

def main():
    print("starting")

    subprocess.run(["bash","-x", "test.sh"])

    print("done")


if __name__ == "__main__":
    main()