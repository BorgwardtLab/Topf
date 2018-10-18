#!/usr/bin/env python3


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
