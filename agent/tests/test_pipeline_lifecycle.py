"""Pipeline options reach processing and stream failures remain readable."""
import asyncio
import io
import json
import queue
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile

from agent.server.routes import pipeline


@pytest.mark.parametrize('options', ['broken', '[]', 'null', '1'])
def test_invalid_reader_options_fail_before_upload_reads(options):
    class Source:
        filename = 'synthetic.txt'
        async def read(self, size):
            pytest.fail('invalid options reached the upload reader')
    with pytest.raises(HTTPException) as error:
        asyncio.run(pipeline.stream_pipeline(files=[Source()], workspace='default',
                    reader_options=options, face_selection='none'))
    assert error.value.status_code == 400


def test_reader_options_reach_the_analysis_worker(monkeypatch):
    import agent.pipeline_runner as runners
    received = []
    class Runner:
        def on_phase(self, callback):
            pass
        def run(self, **kwargs):
            received.append(kwargs)
            return SimpleNamespace(documents=[], elapsed=0, corpus_summary={}, temporal={},
                chunks=[], query_result=None, model_response=None, hallucination_report=None, ontology=None)
    monkeypatch.setattr(runners, 'PipelineRunner', Runner)
    messages = queue.Queue()
    pipeline._analysis_subprocess([], [], [], None, 0, 'quick', None, messages,
                                  face_selection='none', reader_options={'backbone': False})
    assert received[0]['face_selection'] == 'none'
    assert received[0]['reader_options'] == {'backbone': False}
    assert messages.get()[0] == 'done'


def test_requested_face_rule_changes_the_constructed_document(tmp_path):
    from agent.pipeline_runner import PipelineRunner
    path = tmp_path / 'triangle.csv'
    path.write_text('source,target\na,b\nb,c\nc,a\n')
    results = []
    for rule in ('none', 'auto'):
        runner = PipelineRunner()
        runner.run(files=[str(path)], depth='quick', face_selection=rule)
        results.append(runner._last_corpus.documents[0].rex.nF_hodge)
    assert results == [0, 1]


def test_prebuilt_edges_use_the_requested_face_rule():
    from agent.corpus import CorpusBuilder
    from agent.adapters import EdgeConstruction
    import numpy as np
    counts = []
    for rule in ('none', 'auto'):
        corpus = CorpusBuilder(face_selection=rule)
        edges = EdgeConstruction(sources=np.array([0,1,2], np.int32),
                                 targets=np.array([1,2,0], np.int32),
                                 vertex_labels=['a','b','c'], weights=np.ones(3),
                                 signs=np.ones(3), type_labels=np.zeros(3, np.int32),
                                 n_types=1, type_names=['synthetic'])
        corpus.add_document('<synthetic>', edge_construction=edges)
        corpus.build(depth='quick')
        counts.append(corpus.documents[0].rex.nF_hodge)
    assert counts == [0, 1]


def test_stream_errors_escape_newlines_and_backslashes(monkeypatch):
    def fail(*args):
        raise RuntimeError('synthetic\\path\nsecond line "quoted"')
    monkeypatch.setattr(pipeline, '_get_ocr_client', fail)
    async def read():
        response = await pipeline.stream_pipeline(
            files=[UploadFile(io.BytesIO(b'synthetic'), filename='input.png')],
            workspace='default', reader_options='{}', face_selection='none')
        return ''.join([part async for part in response.body_iterator])
    text = asyncio.run(read())
    payload = next(line[6:] for line in text.splitlines() if line.startswith('data: '))
    assert '\\path\nsecond line' in json.loads(payload)['error']


def test_cancelled_ingestion_keeps_files_until_the_reader_finishes(monkeypatch):
    from pathlib import Path
    from threading import Event
    started, release = Event(), Event()
    paths = []
    def ingest(files, *args, **kwargs):
        paths.extend(files)
        started.set()
        assert release.wait(5)
        assert all(Path(path).is_file() for path in files)
        return [], [], []
    monkeypatch.setattr(pipeline, '_ingest_files', ingest)
    async def exercise():
        response = await pipeline.stream_pipeline(
            files=[UploadFile(io.BytesIO(b'synthetic'), filename='input.txt')],
            workspace='default', reader_options='{}', face_selection='none')
        pending = asyncio.create_task(anext(response.body_iterator))
        try:
            assert await asyncio.to_thread(started.wait, 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert paths and all(Path(path).is_file() for path in paths)
        finally:
            release.set()
        for _ in range(100):
            if all(not Path(path).exists() for path in paths):
                break
            await asyncio.sleep(.01)
        assert all(not Path(path).exists() for path in paths)
    asyncio.run(exercise())


@pytest.mark.parametrize('failure', ['queue', 'quiet_exit'])
def test_worker_failures_emit_an_error_and_close_the_queue(monkeypatch, failure):
    class Messages:
        closed = False
        def get(self, **kwargs):
            if failure == 'queue':
                raise OSError('synthetic broken queue')
            raise queue.Empty
        def empty(self):
            return True
        def close(self):
            self.closed = True
        def join_thread(self):
            pass
    class Process:
        exitcode = 0
        alive = failure == 'queue'
        def start(self):
            pass
        def is_alive(self):
            return self.alive
        def terminate(self):
            self.alive = False
        def join(self, **kwargs):
            pass
    messages, process = Messages(), Process()
    monkeypatch.setattr(pipeline, '_analysis_ctx', lambda: SimpleNamespace(
        Queue=lambda: messages, Process=lambda **kwargs: process))
    monkeypatch.setattr(pipeline, '_ingest_files', lambda files, *a, **kw: ([], [], files))
    async def exercise():
        response = await pipeline.stream_pipeline(
            files=[UploadFile(io.BytesIO(b'synthetic'), filename='input.txt')],
            workspace='default', reader_options='{}', face_selection='none')
        return ''.join([part async for part in response.body_iterator])
    text = asyncio.run(exercise())
    assert 'event: error\n' in text and 'event: done\n' not in text
    assert messages.closed and not process.alive
    assert ('broken queue' if failure == 'queue' else 'without a result') in text


def test_workspace_write_failure_reaches_the_caller(monkeypatch):
    def fail(*args):
        raise OSError('synthetic storage failure')
    monkeypatch.setattr('agent.server.persistence.save_document_rex', fail)
    runner = SimpleNamespace(_last_corpus=SimpleNamespace(documents=[
        SimpleNamespace(rex=object(), doc_id='synthetic', text='')]))
    with pytest.raises(OSError, match='storage failure'):
        pipeline._save_to_workspace(runner, 'default')


def test_session_write_failure_is_not_a_successful_registration(monkeypatch):
    from rexgraph import RexGraph
    def fail(*args, **kwargs):
        raise OSError('synthetic session failure')
    session = SimpleNamespace(_metadata={}, add_snapshot=fail)
    monkeypatch.setattr('agent.server.app.get_store', lambda: SimpleNamespace(create=lambda **kw: session))
    monkeypatch.setattr('agent.server.persistence.load_document_rex',
                        lambda *args: RexGraph.from_graph([0], [1]))
    with pytest.raises(OSError, match='session failure'):
        pipeline._register_documents({'documents':[{'doc_id':'synthetic'}]}, 'default')


def test_ocr_cleanup_preserves_unicode_short_lines_and_long_documents():
    from agent.pipeline_runner import PipelineRunner
    text = 'Résumé\n東京\n12.50\n' + 'synthetic text ' * 6000
    assert PipelineRunner._sanitize_text(text) == text
    assert PipelineRunner._sanitize_text('first\r\nsecond\rthird\x00line') == 'first\nsecond\nthird line'


@pytest.mark.parametrize('texts,ids', [(['one'], None), (None, ['one']),
    (['one', 'two'], ['one']), (['one'], ['one', 'two']), ([], ['one'])])
def test_mismatched_text_ids_fail_before_processing(texts, ids, monkeypatch):
    from agent.pipeline_runner import PipelineRunner
    runner = PipelineRunner()
    monkeypatch.setattr(runner, '_build_corpus', lambda *a, **kw: pytest.fail('invalid inputs reached processing'))
    with pytest.raises(ValueError, match='texts.*doc_ids'):
        runner.run(texts=texts, doc_ids=ids, files=['synthetic.csv'])


def test_a_new_run_does_not_retain_the_previous_corpus():
    from agent.pipeline_runner import PipelineRunner
    runner = PipelineRunner()
    runner._last_corpus = object()
    assert runner.run().documents == []
    assert runner._last_corpus is None


def test_interrupted_workspace_text_write_preserves_existing_text(monkeypatch, tmp_path):
    from pathlib import Path
    target = tmp_path / 'synthetic.txt'
    target.write_text('previous complete text', encoding='utf-8')
    monkeypatch.setattr('agent.server.persistence.save_document_rex', lambda *a: None)
    monkeypatch.setattr('agent.server.persistence.doc_path', lambda *a, **kw: target)
    original = Path.write_text
    def fail(self, *a, **kw):
        original(self, 'partial', encoding='utf-8')
        raise OSError('synthetic interrupted write')
    monkeypatch.setattr(Path, 'write_text', fail)
    runner = SimpleNamespace(_last_corpus=SimpleNamespace(documents=[
        SimpleNamespace(rex=object(), doc_id='synthetic', text='replacement complete text')]))
    with pytest.raises(OSError, match='interrupted write'):
        pipeline._save_to_workspace(runner, 'default')
    assert target.read_text(encoding='utf-8') == 'previous complete text'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['synthetic.txt']


def test_analysis_uses_spawn_when_forkserver_is_unavailable(monkeypatch):
    monkeypatch.setattr(pipeline, '_ANALYSIS_CTX', None)
    monkeypatch.setattr(pipeline.multiprocessing, 'get_all_start_methods', lambda: ['spawn'])
    assert pipeline._analysis_ctx().get_start_method() == 'spawn'


def test_unreadable_document_is_reported_instead_of_an_empty_result(tmp_path):
    from agent.pipeline_runner import PipelineRunner
    path = tmp_path / 'empty.txt'
    path.write_text('', encoding='utf-8')
    with pytest.raises(ValueError, match='No document relations'):
        PipelineRunner().run(files=[str(path)], depth='quick')


def test_missing_saved_document_is_not_a_successful_registration(monkeypatch):
    monkeypatch.setattr('agent.server.persistence.load_document_rex', lambda *a: None)
    with pytest.raises(FileNotFoundError, match='synthetic'):
        pipeline._register_documents({'documents':[{'doc_id':'synthetic'}]}, 'default')


def test_interrupted_session_link_write_preserves_existing_links(monkeypatch, tmp_path):
    from pathlib import Path
    from agent.server import persistence
    target = tmp_path / 'doc_sessions.json'
    target.write_text('{"previous":"session_previous"}', encoding='utf-8')
    monkeypatch.setattr(persistence, '_links_path', lambda workspace: target)
    original = Path.write_text
    def fail(self, *a, **kw):
        original(self, '{"partial":', encoding='utf-8')
        raise OSError('synthetic interrupted link write')
    monkeypatch.setattr(Path, 'write_text', fail)
    with pytest.raises(OSError, match='interrupted link write'):
        persistence.link_doc_session('default', 'new', 'session_new')
    assert json.loads(target.read_text()) == {'previous': 'session_previous'}


def test_concurrent_session_links_are_retained(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from agent.server import persistence
    target = tmp_path / 'doc_sessions.json'
    monkeypatch.setattr(persistence, '_links_path', lambda workspace: target)
    def write(index):
        persistence.link_doc_session('default', f'doc_{index}', f'session_{index}')
    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(write, range(20)))
    assert json.loads(target.read_text()) == {f'doc_{i}': f'session_{i}' for i in range(20)}


def test_same_file_names_keep_distinct_stable_documents(monkeypatch, tmp_path):
    from agent.pipeline_runner import PipelineRunner
    from agent.server.persistence import load_document_rex
    paths = []
    for folder, rows in [('first', 'a,b\nb,c\nc,a'), ('second', 'x,y\ny,z')]:
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / 'same.csv'
        path.write_text('source,target\n' + rows + '\n', encoding='utf-8')
        paths.append(str(path))
    runner = PipelineRunner()
    result = runner.run(files=paths, depth='quick')
    ids = [doc['doc_id'] for doc in result.documents]
    assert len(set(ids)) == 2
    reverse = PipelineRunner().run(files=paths[::-1], depth='quick')
    assert [doc['doc_id'] for doc in reverse.documents] == ids[::-1]
    monkeypatch.setenv('REXGRAPH_CONFIG_DIR', str(tmp_path / 'configuration'))
    pipeline._save_to_workspace(runner, 'default')
    assert [load_document_rex('default', doc_id).nE for doc_id in ids] == [3, 2]


def test_duplicate_text_ids_fail_before_processing(monkeypatch):
    from agent.pipeline_runner import PipelineRunner
    runner = PipelineRunner()
    monkeypatch.setattr(runner, '_build_corpus', lambda *a, **kw: pytest.fail('duplicate IDs reached processing'))
    with pytest.raises(ValueError, match='doc_ids.*unique'):
        runner.run(texts=['alpha beta', 'gamma delta'], doc_ids=['same', 'same'])


def test_ocr_failure_reaches_the_pipeline_caller(tmp_path):
    from agent.pipeline_runner import PipelineRunner
    source = tmp_path / 'synthetic.png'
    source.write_bytes(b'synthetic image')
    def fail(path):
        raise RuntimeError('synthetic missing English data')
    runner = PipelineRunner(ocr_client=SimpleNamespace(ocr_image=fail))
    with pytest.raises(RuntimeError, match='synthetic.png.*missing English data'):
        runner.run(files=[str(source)], depth='quick')


def test_blank_ocr_output_is_reported_to_the_pipeline_caller(tmp_path):
    from agent.pipeline_runner import PipelineRunner
    source = tmp_path / 'synthetic.png'
    source.write_bytes(b'synthetic image')
    client = SimpleNamespace(ocr_image=lambda path: SimpleNamespace(text=' \n '))
    with pytest.raises(ValueError, match='No readable text.*synthetic.png'):
        PipelineRunner(ocr_client=client).run(files=[str(source)], depth='quick')


def test_ocr_identity_preserves_content_aliases_and_short_text(tmp_path):
    from pathlib import Path
    from agent.pipeline_runner import PipelineRunner
    from agent.corpus import CorpusBuilder
    paths = []
    for folder, text in [('first', 'alpha connects beta'),
                         ('second', 'gamma connects delta'),
                         ('copy', 'alpha connects beta'), ('short', 'a b c')]:
        directory = tmp_path / folder
        directory.mkdir()
        path = directory / 'same.png'
        path.write_text(text)
        paths.append(str(path))
    client = SimpleNamespace(ocr_image=lambda path: SimpleNamespace(text=Path(path).read_text()))
    corpus = CorpusBuilder()
    runner = PipelineRunner(ocr_client=client)
    for path in paths:
        runner._add_file_to_corpus(corpus, path)
    assert len(corpus.documents) == 4
    ids = [doc.doc_id for doc in corpus.documents]
    assert ids[0] != ids[1]
    assert ids[0] == ids[2]
    assert corpus.documents[-1].text == 'a b c'


@pytest.mark.parametrize('method', ['add_text', 'add_layered_document'])
def test_corpus_rejects_duplicate_ids_before_building(method, monkeypatch):
    from agent.corpus import CorpusBuilder
    corpus = CorpusBuilder()
    corpus.add_text('alpha connects beta', doc_id='same')
    monkeypatch.setattr('rexgraph.document.build_document', lambda *a, **kw: pytest.fail('duplicate IDs reached construction'))
    with pytest.raises(ValueError, match='document ID.*already'):
        if method == 'add_text':
            corpus.add_text('beta connects gamma', doc_id='same')
        else:
            corpus.add_layered_document('<synthetic>', text='beta connects gamma', doc_id='same')
    assert len(corpus.documents) == 1
