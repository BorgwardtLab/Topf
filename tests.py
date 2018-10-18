import unittest
import numpy as np

from topf import PersistenceTransformer


class EmptyDiagram(unittest.TestCase):
    def test(self):
        pt = PersistenceTransformer()
        self.assertRaises(RuntimeError, pt.fit_transform, [])


class DiagramBeketavey(unittest.TestCase):
    '''
    Tests two of my preferred functions for checking persistence
    diagrams. They were originally introduced by Beketavey et al.
    in their paper *Measuring the Distance between Merge Trees*.
    '''

    def test(self):
        a = [(0, 3), (1, 1), (2, 6), (3, 5), (4, 8), (5, 2), (6, 7), (7, 4)]
        b = [(0, 3), (1, 1), (2, 8), (3, 2), (4, 7), (5, 5), (6, 6), (7, 4)]

        persistence_a = PersistenceTransformer().fit_transform(a)
        persistence_b = PersistenceTransformer().fit_transform(b)

        self.assertTrue(len(persistence_a) == len(persistence_b))
        self.assertFalse(np.all(persistence_a == persistence_b))
        self.assertTrue(sorted(persistence_a) == sorted(persistence_b))
