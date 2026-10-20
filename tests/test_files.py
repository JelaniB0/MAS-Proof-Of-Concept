import ctypes
from ctypes import wintypes
import errno
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from agentbalance.files import write_json


class FileWriteTests(unittest.TestCase):
    def test_temporary_windows_lock_preserves_old_report_until_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'report.json'
            target.write_text('{"calls": 6}')
            original = Path.replace
            attempts = []
            def locked_then_released(source, destination):
                attempts.append(source)
                if len(attempts) < 3:
                    self.assertEqual(json.loads(target.read_text()), {'calls': 6})
                    error = PermissionError('File temporarily locked')
                    error.winerror = 5
                    raise error
                return original(source, destination)
            with patch.object(Path, 'replace', locked_then_released), patch('agentbalance.files.time.sleep'):
                write_json(target, {'calls': 12})
            self.assertEqual(json.loads(target.read_text()), {'calls': 12})
            self.assertFalse(target.with_suffix('.json.tmp').exists())

    def test_persistent_lock_keeps_previous_report_and_new_recovery_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'report.json'
            target.write_text('{"calls": 6}')
            error = PermissionError('File locked')
            error.winerror = 32
            with patch.object(Path, 'replace', side_effect=error) as replace, patch('agentbalance.files.time.sleep'):
                with self.assertRaises(PermissionError):
                    write_json(target, {'calls': 12})
            self.assertEqual(replace.call_count, 9)
            self.assertEqual(json.loads(target.read_text()), {'calls': 6})
            self.assertEqual(json.loads(target.with_suffix('.json.tmp').read_text()), {'calls': 12})

    def test_unrelated_io_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            with (patch.object(Path, 'replace', side_effect=OSError(errno.ENOSPC, 'Disk full')) as replace,
                  patch('agentbalance.files.time.sleep') as sleep):
                with self.assertRaises(OSError):
                    write_json(Path(directory)/'report.json', {'calls': 12})
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()

    @unittest.skipUnless(os.name == 'nt', 'Real Windows sharing-lock check')
    def test_real_windows_reader_lock_releases_without_losing_report(self):
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                      wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        kernel.CreateFileW.restype = wintypes.HANDLE
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'report.json'
            target.write_text('{"calls": 6}')
            # Allow reads/writes but deny delete/rename while this reader holds the file.
            handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 0x80, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            released = threading.Event()
            def release():
                kernel.CloseHandle(handle)
                released.set()
            timer = threading.Timer(.15, release)
            timer.start()
            try:
                write_json(target, {'calls': 12})
                self.assertTrue(released.is_set())
                self.assertEqual(json.loads(target.read_text()), {'calls': 12})
            finally:
                timer.join(1)


if __name__ == '__main__':
    unittest.main()
