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
        self.assertFalse(np.all(persistence_a[:, 1] == persistence_b[:, 1]))
        self.assertTrue(np.all(np.sort(persistence_a[:, 1]) == np.sort(persistence_b[:, 1])))

        transformer1 = PersistenceTransformer(calculate_persistence_diagram=False)
        transformer2 = PersistenceTransformer(calculate_persistence_diagram=True)

        transformer1.fit_transform(a)
        transformer2.fit_transform(b)

        self.assertIsNone(transformer1.persistence_diagram)
        self.assertIsNotNone(transformer2.persistence_diagram)

        # Checks that the persistence calculation is in line with the
        # reported diagram.
        for point, transformed_point in zip(transformer2.persistence_diagram, persistence_b):
            self.assertEqual(point[0] - point[1], transformed_point[1])

        self.assertEqual(transformer2.persistence_diagram.total_persistence(), 15.0)

        tp1 = transformer2.persistence_diagram.total_persistence()
        tp2 = transformer2.persistence_diagram.total_persistence(1.0)

        self.assertEqual(tp1, tp2)


class DiagramRegularPoints(unittest.TestCase):
    '''
    Checks that the addition of regular points to a series does not
    create any new points in the persistence diagram.
    '''

    def test(self):
        a = [(0, 3), (1, 1), (2, 6), (3, 5), (4, 8), (5, 2), (6, 7), (7, 4)]
        b = [(0.5, 1.5), (1.5, 4.5), (2.5, 5.5), (3.5, 6.5), (5.5, 3.5), (6.5, 5.75)]
        x = sorted(a + b, key=lambda x: x[0])

        transformer = PersistenceTransformer(calculate_persistence_diagram=True)
        transformer.fit_transform(x)
        diagram = transformer.persistence_diagram

        self.assertEqual(diagram.total_persistence(), 15.0)


class DiagramPlateauPoints(unittest.TestCase):
    '''
    Checks that the existence of points at equal levels as their
    neighbours does not change the points in the diagram.
    '''

    def test(self):
        a = [(0, 3), (1, 1), (2, 6), (3, 5), (4, 8), (5, 2), (6, 7), (7, 4)]
        b = [(1.5, 1), (2.5, 6), (3.5, 5)]
        c = [(0, 0), (1, 0), (2, 1), (3, 0), (4, 1), (5, 0), (6, 0), (7, 1)]

        x = sorted(a + b, key=lambda x: x[0])
        y = c

        transformer = PersistenceTransformer(calculate_persistence_diagram=True)

        transformer.fit_transform(x)
        diagram1 = transformer.persistence_diagram

        transformer.fit_transform(y)
        diagram2 = transformer.persistence_diagram

        self.assertEqual(diagram1.total_persistence(), 15.0)
        self.assertEqual(diagram2.total_persistence(), 3.0)
