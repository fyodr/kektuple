"""
c3cbp_helpers.py
by Ted Morin

Edge, Face, and Vertex classes to use with c3cbp class
"""
import math

class Edge:
    def __init__(self,faces,verts):
        self.faces = tuple(faces)
        self.verts = tuple(verts) # This is an index into the graph's verts list
        
    def get_other_vert(self,v): # return the vertex besides v. v is an index into g.verts
        if self.verts[0] == v:
            return self.verts[1]
        elif self.verts[1] == v:
            return self.verts[0]
        # v is not a vertex of the edge
        return None

    # return the face besides f
    # f is an index into the face list
    def get_other_face(self,f): 
        if self.faces[0] == f:
            return self.faces[1]
        elif self.faces[1] == f:
            return self.faces[0]
        # f is not a face of the edge
        return None
    
    def repl_vert(self,v1,v2): # replace vertex v1 with vertex v2
        if self.verts[0] == v1:
            self.verts = (v2, self.verts[1])
        elif self.verts[1] == v1:
            self.verts = (self.verts[0], v2)
        else :
            # v1 is not a vertex of the edge
            return 1
        return 0
    
    def repl_face(self,f1,f2): # replace face f1 with face f2
        if self.faces[0] == f1:
            self.faces = (f2, self.faces[1])
        elif self.faces[1] == f1:
            self.faces = (self.faces[0], f2)
        else :
            # f1 is not a face of the edge
            return 1
        return 0
    
    def __cmp__(self,other):
        if self.faces[0] not in other.faces:
            return 1
        elif self.faces[1] not in other.faces:
            return 1
        elif self.verts[0] not in other.verts:
            return 1
        elif self.verts[1] not in other.verts:
            return 1
        else :
            return 0
        
    def permute(self,faces,verts):
        self.verts = tuple(verts[v] for v in self.verts)
        self.faces = tuple(faces[f] for f in self.faces)

class Vert:
    def __init__(self,edges,faces):
        self.faces = list(faces)
        self.edges = list(edges) # This is a  list of indices into the graph's edge list
        
    def repl_edge(self, e1, e2):
        if e1 not in self.edges:
            # passed bad edge number
            return 1
        self.edges[self.edges.index(e1)] = e2
        return 0
    
    def repl_face(self, f1, f2):
        if f1 not in self.faces:
            # passed bad face number
            return 1
        self.faces[self.faces.index(f1)] = f2
        return 0
    
    def __cmp__(self,other):
        if self.edges == other.edges and self.faces == other.faces:
            return 0
        else :
            return 1
        
    def permute(self,edges,faces):
        self.edges = [edges[e] for e in self.edges]
        self.faces = [faces[f] for f in self.faces]

class Face:
    def __init__(self,verts,edges):
        self.verts = list(verts) 
        self.edges = list(edges)

    # R4 expansion helper function - expects new vertices and edges
    def R4exp(self,v,newv,newe):
        if v not in self.verts:
            # this should never happen
            print("Vertex {} is not in the face!".format(v))
            return 1
        ix = self.verts.index(v)
        self.verts = self.verts[:ix]+newv+self.verts[ix+1:]
        self.edges = self.edges[:ix]+newe+self.edges[ix:]
        return 0

    # R0 expansion helper function - expects 3 edges and 2 vertices
    def R0exp(self,e,newe,newv):
        if e not in self.edges:
            # this should never happen
            print("Edge {} is not in the face!".format(e))
            return 1
        ix = self.edges.index(e)
        self.edges = self.edges[:ix]+newe+self.edges[ix+1:]
        self.verts = self.verts[:ix+1]+newv+self.verts[ix+1:]
        return 0

    def repl_edge(self, e1, e2):
        if e1 not in self.edges:
            # passed bad edge number
            return 1
        self.edges[self.edges.index(e1)] = e2
        return 0

    def repl_vert(self,v1,v2):
        if v1 not in self.verts:
            # passed bad vert number
            return 1
        self.verts[self.verts.index(v1)] = v2
        return 0

    def __cmp__(self,other):
        if self.edges == other.edges and self.verts == other.verts:
            return 0
        else :
            return 1

    def permute(self,verts,edges):
        self.verts = [verts[v] for v in self.verts]
        self.edges = [edges[e] for e in self.edges]


