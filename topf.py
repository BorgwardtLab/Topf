#!/usr/bin/env python3


import numpy as np
import collections.abc


class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default and uses integers internally for storing a
    disjoint set.

    The class requires the vertices to form a contiguous sequence, so
    no gaps are allowed. The vertices also have to be zero-indexed.
    '''

    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.parent = {x: x for x in range(num_vertices)}

    def find(self, u):
        if self.parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self.parent[u] = self.find(self.parent[u])
            return self.parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''
        if u != v:
            self.parent[self.find(u)] = self.find(v)


class PersistenceDiagram(collections.abc.Sequence):
    '''
    Simple class for storing the pairs of a persistence diagram. This is
    nothing but a light-weight wrapper for additional convenience.
    '''

    def __init__(self, pairs):
        self._pairs = pairs

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        return self._pairs[index]


class PersistenceTransformer:
    '''
    Transforms a function that is represented as an input array of (x,y)
    pairs into (x, persistence(x)) pairs. Optionally, all tuples of the
    corresponding persistence diagram are returned as well.
    '''

    def __init__(self, calculate_persistence_diagram=False):
        self._calculate_persistence_diagram = calculate_persistence_diagram
        self._persistence_diagram = None

    def fit_transform(self, a):
        a = np.asarray(a)

        if len(a.shape) != 2 or a.shape[1] != 2:
            raise RuntimeError('Unexpected array format')

        indices = np.argsort(a[:,1])[::-1]

        # Optionally, the function can also return a proper persistence
        # diagram, i.e. a set of tuples that describe the merges.
        if self._calculate_persistence_diagram:
            b = np.zeros_like(a)
            b[:,0] = a[:,1]  # y
            b[:,1] = a[:,1]  # y (this is correct; everything is paired with itself)
        else:
            b = None

        # Prepare Union--Find data structure; by default, every vertex
        # is initialized to be its own parent.
        num_vertices = len(a)
        uf = UnionFind(num_vertices)

        # By default, all points that are not explicitly handled will be
        # assigned a persistence value of zero.
        persistence = np.zeros(num_vertices)

        for index in indices:
            left_index = index - 1
            right_index = index + 1

            x = a[index, 0]
            y = a[index, 1]

            # Inner point: both neighbours are defined; this is easy to
            # handle because we just have to check both of them.
            if left_index >= 0 and right_index <= num_vertices - 1:
                y_left = a[left_index, 1]
                y_right = a[right_index, 1]

                # The point is a local minimum, so we have to merge the
                # two neighbours.
                if y_left >= y and y <= y_right:

                    # The left neighbour is the younger neighbour, so it
                    # will be merged into the right one.
                    if a[uf.find(left_index), 1] < a[uf.find(right_index), 1]:
                        persistence[uf.find(left_index)] = a[uf.find(left_index), 1] - y

                        if self._calculate_persistence_diagram:
                            b[uf.find(left_index), 1] = y

                        uf.merge(left_index, right_index)
                    else:
                        persistence[uf.find(right_index)] = a[uf.find(right_index), 1] - y

                        if self._calculate_persistence_diagram:
                            b[uf.find(right_index), 1] = y

                        uf.merge(right_index, left_index)

                # The point is a regular point, i.e. one neighbour
                # has a higher function value, the other one has a
                # lower function value.
                elif not (y > y_left and y > y_right):

                    # Always merge the lower neighbour into the current
                    # point. This merge does not give rise to a pair.
                    if y_left < y_right:
                        uf.merge(left_index, right_index)
                    else:
                        uf.merge(right_index, left_index)

        # Assign the persistence value to the global maximum of the
        # function to ensure that all tuples have been paired.
        if num_vertices > 0:
            global_maximum_index = indices[0]
            global_minimum_index = indices[-1]

            persistence[global_maximum_index] = a[global_maximum_index, 1] - a[global_minimum_index, 1]

            if self._calculate_persistence_diagram:
                b[global_maximum_index, 1] = a[global_minimum_index, 1]

        # Only create a persistence diagram if we have some persistence
        # tuples to store.
        if b is not None:
            self._persistence_diagram = PersistenceDiagram(b)

        return persistence

    @property
    def persistence_diagram(self):
        '''
        :return: Returns the persistence diagram that was optionally
        calculated by calling :func:`fit_transform`. The diagram, if
        available, will be returned as a 2D ``numpy.array`.
        '''

        return self._persistence_diagram


if __name__ == '__main__':
    pt = PersistenceTransformer()
    pt.fit_transform([(0,1),(1,7),(2,4),(3,5),(4,2),(5,8),(6,0)])
