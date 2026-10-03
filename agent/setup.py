"""Include the maintained sibling UI in the installed Agent package."""
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        source = Path(__file__).parent / "frontend"
        target = Path(self.build_lib) / "agent" / "frontend"
        names = (Path(__file__).parent / "frontend-files.txt").read_text().splitlines()
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Agent frontend inventory requires relative paths")
            path = source / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("Agent frontend build requires regular source files: "+name)
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            self.copy_file(str(path), str(destination))


setup(cmdclass={"build_py": BuildPy})
