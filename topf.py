# Copyright (c) 2018 Bastian Rieck
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

__author__ = 'Bastian Rieck'
__version__ = '0.1'


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

    def __str__(self):
        return str(self._pairs)

    def total_persistence(self, p=1.0):
        '''
        Calculates the sum of all persistence values in the diagram,
        weighted by the specified power.
        '''

        assert p > 0.0

        persistence_values = self._pairs[:, 0] - self._pairs[:, 1]
        persistence_values = np.power(persistence_values, p)

        return np.power(np.sum(persistence_values), 1.0 / p)


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

        # This way of sorting ensures that points with the same
        # y value will be sorted according to their x value. It
        # ensures that left-most points are detected first.
        indices = np.argsort(-a[:, 1], kind='stable')

        # Optionally, the function can also return a proper persistence
        # diagram, i.e. a set of tuples that describe the merges.
        if self._calculate_persistence_diagram:
            b = np.zeros_like(a)
            b[:, 0] = a[:, 1]  # y
            b[:, 1] = a[:, 1]  # y (everything is paired with itself)
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
                        persistence[uf.find(left_index)] \
                            = a[uf.find(left_index), 1] - y

                        if self._calculate_persistence_diagram:
                            b[uf.find(left_index), 1] = y

                        uf.merge(left_index, index)
                        uf.merge(index, right_index)
                    else:
                        persistence[uf.find(right_index)] \
                            = a[uf.find(right_index), 1] - y

                        if self._calculate_persistence_diagram:
                            b[uf.find(right_index), 1] = y

                        uf.merge(right_index, index)
                        uf.merge(index, left_index)

                # The point is a regular point, i.e. one neighbour
                # has a higher function value, the other one has a
                # lower function value.
                elif not (y > y_left and y > y_right):

                    # Always merge the current point into the higher one
                    # of its neighbours. This merge does not result in a
                    # pair.
                    if a[uf.find(left_index), 1] < a[uf.find(right_index), 1]:
                        uf.merge(index, right_index)
                    else:
                        uf.merge(index, left_index)

        # Assign the persistence value to the global maximum of the
        # function to ensure that all tuples have been paired.
        if num_vertices > 0:
            global_maximum_index = indices[0]
            global_minimum_index = indices[-1]

            persistence[global_maximum_index] \
                = a[global_maximum_index, 1] - a[global_minimum_index, 1]

            if self._calculate_persistence_diagram:
                b[global_maximum_index, 1] = a[global_minimum_index, 1]

        # Only create a persistence diagram if we have some persistence
        # tuples to store.
        if b is not None:
            self._persistence_diagram = PersistenceDiagram(b)

        return np.vstack((a[:, 0], persistence)).T

    @property
    def persistence_diagram(self):
        '''
        :return: Returns the persistence diagram that was optionally
        calculated by calling :func:`fit_transform`. The diagram, if
        available, will be returned as a 2D ``numpy.array`.
        '''

        return self._persistence_diagram
