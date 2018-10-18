import unittest

from topf import PersistenceTransformer


class EmptyDiagram(unittest.TestCase):
    def test(self):
        pt = PersistenceTransformer()
        self.assertRaises(RuntimeError, pt.fit_transform, [])
