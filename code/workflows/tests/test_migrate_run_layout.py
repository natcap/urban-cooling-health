"""Filesystem-only tests: no model data or cloud access required."""
import tempfile
import unittest
from pathlib import Path
from code.workflows.migrate_run_layout import migrate


class RunLayoutTest(unittest.TestCase):
    def test_nested_moves_aliases_idempotence_and_rollback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'UCM_official_runs'
            (root / 'old/child').mkdir(parents=True)
            (root / 'old/child/evidence').write_bytes(b'unchanged')
            rows = [{'old': 'old', 'new': 'production/new'},
                    {'old': 'production/new/child', 'new': 'production/new/derived'}]
            migrate(root, rows)
            self.assertFalse((root / 'production').exists())
            migrate(root, rows, apply=True)
            self.assertEqual((root / 'old/child/evidence').read_bytes(), b'unchanged')
            migrate(root, rows, apply=True)
            migrate(root, rows, apply=True, rollback=True)
            self.assertFalse((root / 'old').is_symlink())
            self.assertEqual((root / 'old/child/evidence').read_bytes(), b'unchanged')

    def test_occupied_destination_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'UCM_official_runs'
            (root / 'old').mkdir(parents=True)
            (root / 'occupied').mkdir()
            for destination in ('occupied', '../outside'):
                with self.assertRaises(ValueError):
                    migrate(root, [{'old': 'old', 'new': destination}], apply=True)
            self.assertFalse((root / 'old').is_symlink())


if __name__ == '__main__':
    unittest.main()
