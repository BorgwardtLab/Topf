#!/usr/bin/env python3


import numpy as np


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


class PersistenceTransformer:
    '''
    Transforms a function that is represented as an input array of (x,y)
    pairs into (x, persistence(x)) pairs. Optionally, all tuples of the
    corresponding persistence diagram are returned as well.
    '''

    def fit_transform(self, a):
        a = np.asarray(a)
        indices = np.argsort(a[:,1])[::-1]

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
                    print('[-] at ({},{})'.format(x, y))

                    # The left neighbour is the younger neighbour, so it
                    # will be merged into the right one.
                    if a[uf.find(left_index), 1] < a[uf.find(right_index), 1]:
                        uf.merge(left_index, right_index)
                        persistence[left_index] = y
                    else:
                        uf.merge(right_index, left_index)
                        persistence[right_index] = y

                # The point is a regular point, i.e. one neighbour
                # has a higher function value, the other one has a
                # lower function value.
                elif not (y > y_left and y > y_right):
                    print('[=] at ({},{})'.format(x, y))

                    # Always merge the lower neighbour into the current
                    # point. This merge does not give rise to a pair.
                    if y_left < y_right:
                        uf.merge(left_index, right_index)
                    else:
                        uf.merge(right_index, left_index)

                else:
                    print('[+] at ({},{})'.format(x, y))

            # Boundary point: only one neighbour is defined
            else:
                if left_index < 0:
                    neighbour_index = right_index
                else:
                    neighbour_index = left_index

                print('[B] at ({},{})'.format(x, y))

        print(persistence)


if __name__ == '__main__':
    pt = PersistenceTransformer()
    pt.fit_transform([(0,1),(1,7),(2,4),(3,5)])
