"""Import probes retain dependency paths while avoiding ambient startup hooks."""
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace


def test_registered_dependency_path_reaches_the_isolated_child(monkeypatch, tmp_path):
    helper = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/test_subprocess.py'))
    dependencies = tmp_path / 'site-packages'
    dependencies.mkdir()
    (dependencies / 'synthetic_dependency.py').write_text('value = 317\n')
    (dependencies / '_rexgraph_editable_loader.py').write_text("raise RuntimeError('unexpected editable hook')\n")
    monkeypatch.setattr(sys, 'path', [*sys.path, str(dependencies)])
    result = helper['run_isolated']('import synthetic_dependency; assert synthetic_dependency.value == 317',
                                    capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_wheel_probe_does_not_activate_an_ambient_editable_loader(monkeypatch, tmp_path):
    helper = runpy.run_path(str(Path(__file__).resolve().parents[2] / 'scripts/test_subprocess.py'))
    dependency = tmp_path / 'site-packages'
    dependency.mkdir()
    package = dependency / 'rexgraph'
    package.mkdir()
    (package / '__init__.py').write_text('')
    (dependency / '_rexgraph_editable_loader.py').write_text("raise RuntimeError('unexpected editable hook')\n")
    monkeypatch.setenv('REXGRAPH_TEST_INSTALLED', '1')
    monkeypatch.setitem(helper['run_isolated'].__globals__, 'distribution',
                        lambda name: SimpleNamespace(read_text=lambda filename: None))
    monkeypatch.setattr(sys, 'path', [*sys.path, str(dependency)])
    monkeypatch.setitem(helper['run_isolated'].__globals__, 'site',
                        type('Site', (), {'getsitepackages': staticmethod(lambda: [str(dependency), *sys.path])}))
    code = ('from pathlib import Path; import rexgraph; '
            f'assert str(Path(rexgraph.__file__).resolve()) == {str(package / "__init__.py")!r}')
    result = helper['run_isolated'](code, packages=('rexgraph',), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
