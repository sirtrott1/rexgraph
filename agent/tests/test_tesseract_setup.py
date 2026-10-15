"""Tesseract needs its language data and must report extraction failures."""
from types import SimpleNamespace
import subprocess

import pytest

from agent.integrations.unlimited_ocr import OfflineOCRClient


@pytest.mark.parametrize('languages,expected', [('eng\nosd\n', True), ('afr\nosd\n', False), ('', False)])
def test_english_data_is_required_for_availability(monkeypatch, languages, expected):
    from agent.cli.setup import has_tesseract
    def run(args, **kwargs):
        if '--list-langs' in args:
            return SimpleNamespace(returncode=0, stdout=languages, stderr='')
        return SimpleNamespace(returncode=0, stdout='tesseract 5', stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('shutil.which', lambda name: '/synthetic/tesseract')
    assert OfflineOCRClient().is_available() is expected
    assert has_tesseract() is expected


def test_missing_language_data_cannot_produce_a_successful_empty_result(monkeypatch):
    monkeypatch.setattr(OfflineOCRClient, '_check_tesseract', staticmethod(lambda: False))
    with pytest.raises(RuntimeError, match='Tesseract.*English'):
        OfflineOCRClient().ocr_image('synthetic.png')


@pytest.mark.parametrize('failure', ['exit', 'timeout'])
def test_native_extraction_failure_reaches_the_caller(monkeypatch, failure):
    monkeypatch.setattr(OfflineOCRClient, '_check_tesseract', staticmethod(lambda: True))
    def run(args, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired(args, 60)
        return SimpleNamespace(returncode=1, stdout='', stderr='synthetic native OCR failure')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='Tesseract'):
        OfflineOCRClient().ocr_image('synthetic.png')


@pytest.mark.parametrize('manager,language_package', [
    ('apt', 'tesseract-ocr-eng'), ('dnf', 'tesseract-langpack-eng'), ('pacman', 'tesseract-data-eng')])
def test_setup_installs_the_language_data_with_the_engine(monkeypatch, manager, language_package):
    from agent.cli import setup
    availability = iter([False, True])
    monkeypatch.setattr(setup, 'has_tesseract', lambda: next(availability))
    commands = []
    def run(args, **kwargs):
        commands.append(args)
        return SimpleNamespace(returncode=0, stderr='')
    monkeypatch.setattr(subprocess, 'run', run)
    assert setup.install_tesseract(SimpleNamespace(package_manager=manager, has_sudo=True), interactive=False)
    assert language_package in commands[0]
