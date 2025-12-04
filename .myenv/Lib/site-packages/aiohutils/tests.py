import atexit
from asyncio import new_event_loop
from itertools import cycle
from typing import Iterator, NotRequired, get_args, get_origin, is_typeddict
from unittest.mock import patch

from decouple import config
from pytest import fixture

from aiohutils.session import ClientSession, SessionManager

RECORD_MODE = OFFLINE_MODE = TESTS_PATH = REMOVE_UNUSED_TESTDATA = None


def init_tests():
    global RECORD_MODE, OFFLINE_MODE, TESTS_PATH, REMOVE_UNUSED_TESTDATA

    config.search_path = TESTS_PATH = config._caller_path()

    RECORD_MODE = config('RECORD_MODE', False, cast=bool)
    OFFLINE_MODE = config('OFFLINE_MODE', False, cast=bool) and not RECORD_MODE
    REMOVE_UNUSED_TESTDATA = (
        config('REMOVE_UNUSED_TESTDATA', False, cast=bool) and OFFLINE_MODE
    )


class EqualToEverything:
    def __eq__(self, other):
        return True


class FakeResponse:
    files: Iterator = None
    url = EqualToEverything()
    history = ()

    @property
    def file(self) -> str:
        return next(self.files)

    async def read(self):
        with open(self.file, 'rb') as f:
            content = f.read()
        return content


@fixture(scope='session', autouse=True)
async def session():
    if OFFLINE_MODE:

        class FakeSession:
            @staticmethod
            async def get(*_, **__):
                return FakeResponse()

        SessionManager.session = FakeSession()
        yield
        del SessionManager.session
        return

    if RECORD_MODE:
        original_get = ClientSession.get

        async def recording_get(*args, **kwargs):
            resp = await original_get(*args, **kwargs)
            content = await resp.read()
            with open(FakeResponse().file, 'wb') as f:
                f.write(content)
            return resp

        ClientSession.get = recording_get

        yield
        ClientSession.get = original_get
        return

    yield
    return


@fixture(scope='session')
def event_loop():
    loop = new_event_loop()
    yield loop
    loop.close()


def remove_unused_testdata():
    if REMOVE_UNUSED_TESTDATA is not True:
        return
    import os

    unused_testdata = (
        set(os.listdir(f'{TESTS_PATH}/testdata/')) - USED_FILENAMES
    )
    if not unused_testdata:
        print('REMOVE_UNUSED_TESTDATA: no action required')
        return
    for filename in unused_testdata:
        os.remove(f'{TESTS_PATH}/testdata/{filename}')
        print(f'REMOVE_UNUSED_TESTDATA: removed {filename}')


USED_FILENAMES = set()
atexit.register(remove_unused_testdata)


def file(filename: str):
    if REMOVE_UNUSED_TESTDATA is True:
        USED_FILENAMES.add(filename)
    return patch.object(
        FakeResponse,
        'files',
        cycle([f'{TESTS_PATH}/testdata/{filename}']),
    )


def files(*filenames: str):
    if REMOVE_UNUSED_TESTDATA is True:
        for filename in filenames:
            USED_FILENAMES.add(filename)
    return patch.object(
        FakeResponse,
        'files',
        (f'{TESTS_PATH}/testdata/{filename}' for filename in filenames),
    )


def assert_dict_type(d: dict, td: callable):
    assert td.__optional_keys__ >= (d.keys() - td.__required_keys__)
    annotations = td.__annotations__
    for k, v in d.items():
        expected_type = annotations[k]
        if is_typeddict(expected_type):
            assert_dict_type(v, expected_type)
            continue
        if get_origin(expected_type) is NotRequired:
            expected_type = get_args(expected_type)
        assert isinstance(v, expected_type), f'{k=} {v=} {expected_type=}'
