"""
kektup.py
by Ted Morin

kektup <= c3c2p <= c3cbp

the base class of kektuple
simply represents a Barnette graph with certain attributes
"""

from kektup_helpers import *
import numpy as np
import random
from queue import Queue

class Kektup:
    # construct from instructions or lists of vertices, faces, edges.
    def __init__(self, hist=[], vef=()):
        # if given a tuple of verts, edges, faces, build the graph from there
        self.hist = []
        self.ifts = None # not sure what this is for?
        
        if vef :
            self.verts = [Vert(vert[0],vert[1]) for vert in vef[0]]
            self.edges = [Edge(edge[0],edge[1]) for edge in vef[1]]
            self.faces = [Face(face[0],face[1]) for face in vef[2]]
            self.edge_counts = []
            # check
            try :
                bad_edge_face, bad_face_vert, bad_vert_edge = \
                                                    self.check_agreement()
            except :
                print ("Bad graph! Could not perform check.")
                return
            if bad_edge_face or bad_face_vert or bad_vert_edge:
                print ("Bad graph! Check failed.")
                return
            
        # otherwise, start with the default
        else :
            self.edge_counts = []
            # make the cube - orders are ccw in embedding
            self.verts = [Vert([7,8,11],[3,4,5]), Vert([0,4,3],[0,1,4]),
                          Vert([1,5,0],[0,2,1]), Vert([2,6,1],[0,3,2]),
                          Vert([3,7,2],[0,4,3]), Vert([9,8,4],[1,5,4]),
                          Vert([5,10,9],[1,2,5]), Vert([6,11,10],[2,3,5])]
            #self.verts = {i: verts[i] for i in range(8)}
            self.faces = [Face([4,3,2,1],[2,1,0,3]),Face([2,6,5,1],[5,9,4,0]),
                          Face([2,3,7,6],[1,6,10,5]),Face([3,4,0,7],[2,7,11,6]),
                          Face([1,5,0,4],[4,8,7,3]),Face([6,7,0,5],[10,11,8,9])]
            #self.faces = {i: faces[i] for i in range(6)}
            self.edges = [Edge([0,1],[1,2]),Edge([0,2],[2,3]),Edge([0,3],[3,4]),
                          Edge([0,4],[4,1]),Edge([1,4],[1,5]),Edge([1,2],[2,6]),
                          Edge([2,3],[3,7]),Edge([3,4],[0,4]),Edge([4,5],[0,5]),
                          Edge([1,5],[5,6]),Edge([2,5],[6,7]),Edge([3,5],[0,7])]
            #self.edges = {i: edges[i] for i in range(12)}
            # identify a coloring of graph
            self.black = [0, 1, 3, 6]
            self.hist = []
            
            # parse history and construct graph
            for step in hist:
                if step[0].lower()[:5] == "r0exp": # R0 expansion
                    self.r0exp(step[1],step[2])
                elif step[0].lower()[:5] == "r4exp": # R4 expansion
                    self.r4exp(step[1])
                elif step[0].lower()[:5] == "r0red": # R0 reduction
                    self.R4red(step[1],step[2])
                elif step[0].lower()[:5] == "r4red": # R4 reduction
                    self.R4red(step[1])

        # identify a color of graph
        self.vertex_colors = self.make_vertex_coloring()
        self.face_colors = self.make_face_coloring()

    def get_neighbor_verts(self,v):
        return [self.edges[e].get_other_vert(v) for e in self.verts[v].edges]

    #Allows us to construct random graph at instantiation with little annoyance 
    @classmethod
    def random_graph(cls, n, seed):
        g = cls()
        rm = random.Random(seed)
        while len(g.verts) < n:
            if len(g.verts) >= n -8:
                operation = 'r0exp'
            else:
                operation = rm.choice(['r0exp', 'r4exp'])
            if operation == 'r0exp':
                # find an elligible edge pair
                e1, e2 = rm.choice(g.get_elligible_edge_pairs(
                                                order_matters=False))
                # apply operation
                g.r0exp(e1,e2)
            else : #elif operation == 'r4exp':
                v = rm.randrange(len(g.verts))
                g.r4exp(v)
        return g


    # return the face common to edges e1 and e2
    def get_common_face(g,e1,e2):
        e1faces = g.edges[e1].faces
        e2faces = g.edges[e2].faces
        if e1faces[0] in e2faces:
            return e1faces[0]
        elif e1faces[1] in e2faces:
            return e1faces[1]
        else:
            return None

    # get the neighbors of a vertex
    def get_vert_neighbors(g,v):
        return [g.edges[e].get_other_vert(v) for e in g.verts[v].edges]

    # get the neighbors of a face
    def get_vert_neighbors(g,f):
        return [g.edges[e].get_other_face(f) for e in g.faces[f].edges]

    # check if edges e1 and e2 are elligible for R0
    def r0exp_eli(g,e1,e2):
        # check that they are not the same edge
        if e1 == e2:
            return 0
        # check that edges exist
        nedges = len(g.edges)
        if e1 < 0 or e2 < 0 or e1 >= nedges or e2 >= nedges:
            return 0
        # check for common face
        f = g.get_common_face(e1,e2)
        if f == None:
            return 0
        # check for even spacing
        edge_dist = g.faces[f].edges.index(e1) - g.faces[f].edges.index(e2)
        if edge_dist % 2 == 1:
            return 0
        return 1

    # perform R0 expansion on g, expands on edges e1, e2
    def r0exp(g,e1,e2):
        # assumes g.r0exp_eli(e1,e2) == True
        f = g.get_common_face(e1,e2)
        # get indices of e1 and e2 in f check that they are an evenly spaced
        e1_f_ix = g.faces[f].edges.index(e1)
        e2_f_ix = g.faces[f].edges.index(e2)
        edge_dist = e2_f_ix - e1_f_ix
        # prepare to expand
        x = len(g.verts); y = len(g.faces); z = len(g.edges)
        # identify other faces involved
        f1 = g.edges[e1].get_other_face(f)
        f2 = g.edges[e2].get_other_face(f)
        # identify and connect vertices on either side of e1 and e2 to new edges
        deg = len(g.faces[f].verts)
        e1_before = g.faces[f].verts[e1_f_ix]
        e1_after  = g.faces[f].verts[(e1_f_ix+1)%deg]
        e2_before = g.faces[f].verts[e2_f_ix]
        e2_after  = g.faces[f].verts[(e2_f_ix+1)%deg]
        g.verts[e1_before].repl_edge(e1,z+0)
        g.verts[e1_after].repl_edge(e1,z+5)
        g.verts[e2_before].repl_edge(e2,z+3)
        g.verts[e2_after].repl_edge(e2,z+2)
        # add new faces
        if edge_dist > 0:
            new_f3_verts = g.faces[f].verts[e2_f_ix+1:] +\
                            g.faces[f].verts[:e1_f_ix+1]
            new_f3_edges = g.faces[f].edges[e2_f_ix+1:] +\
                            g.faces[f].edges[:e1_f_ix+1]
            new_f4_verts = g.faces[f].verts[e1_f_ix+1:e2_f_ix+1]
            new_f4_edges = g.faces[f].edges[e1_f_ix+1:e2_f_ix+1]
        else :
            new_f3_verts = g.faces[f].verts[e2_f_ix+1:e1_f_ix+1]
            new_f3_edges = g.faces[f].edges[e2_f_ix+1:e1_f_ix+1]
            new_f4_verts = g.faces[f].verts[e1_f_ix+1:] +\
                            g.faces[f].verts[:e2_f_ix+1]
            new_f4_edges = g.faces[f].edges[e1_f_ix+1:] +\
                            g.faces[f].edges[:e2_f_ix+1]
        g.faces.append(Face(new_f3_verts,new_f3_edges))
        g.faces.append(Face(new_f4_verts,new_f4_edges))
        # apply R0exp helper to faces
        g.faces[f1].R0exp(e1,[z+5,e1,z+0],[x+1,x+0])
        g.faces[f2].R0exp(e2,[z+2,e2,z+3],[x+3,x+2])
        g.faces[y+0].R0exp(g.faces[y+0].edges[-1],[z+0,z+1,z+2],[x+0,x+3])
        g.faces[y+1].R0exp(g.faces[y+1].edges[-1],[z+3,z+4,z+5],[x+2,x+1])
        # new face f
        g.faces[f] = Face([x+0,x+1,x+2,x+3],[e1,z+4,e2,z+1])
        # add new verts
        g.verts.append(Vert([z+1,z+0,e1],[f,y+0,f1]))
        g.verts.append(Vert([e1,z+5,z+4],[f,f1,y+1]))
        g.verts.append(Vert([z+4,z+3,e2],[f,y+1,f2]))
        g.verts.append(Vert([e2,z+2,z+1],[f,f2,y+0]))
        # modify edges affected
        g.edges[e1].verts = [x+0,x+1]
        g.edges[e2].verts = [x+2,x+3]
        # add new edges
        g.edges.append(Edge([f1,y+0],[e1_before,x+0]))
        g.edges.append(Edge([f, y+0],[x+0,x+3]))
        g.edges.append(Edge([f2,y+0],[x+3,e2_after]))
        g.edges.append(Edge([f2,y+1],[e2_before,x+2]))
        g.edges.append(Edge([f,y+1],[x+2,x+1]))
        g.edges.append(Edge([f1,y+1],[x+1,e1_after]))
        # assign verts and edges of old f to new faces
        for v in g.faces[y+0].verts:
            if v not in [x+0,x+3]:
                g.verts[v].repl_face(f, y+0)
        for e in g.faces[y+0].edges:
            if e not in [z+0,z+1,z+2]:
                g.edges[e].repl_face(f, y+0)
        for v in g.faces[y+1].verts:
            if v not in [x+2,x+1]:
                g.verts[v].repl_face(f, y+1)
        for e in g.faces[y+1].edges:
            if e not in [z+3,z+4,z+5]:
                g.edges[e].repl_face(f, y+1)
        # update coloring and history, return
        if e1_before in g.black:
            g.black += [x+1, x+3]
        else :
            g.black += [x+0, x+2]
        g.hist.append(('r0exp',e1,e2))
        g.face_colors = None
        g.vertex_colors = None
        return 0
        

    # perform R4 expansion on g, expand on vertex v
    def r4exp(g,v):
        x = len(g.verts); y = len(g.faces); z = len(g.edges)
        vedges = g.verts[v].edges # edges affected
        vfaces = g.verts[v].faces # faces affected
        # add new faces
        g.faces.append(Face([v,x+5,x+0,x+1],[z+2,z+8,z+3,z+0]))
        g.faces.append(Face([v,x+1,x+2,x+3],[z+0,z+4,z+5,z+1]))
        g.faces.append(Face([v,x+3,x+4,x+5],[z+1,z+6,z+7,z+2]))
        # add new edges
        g.edges.append(Edge((y+0,y+1),(v,x+1)))
        g.edges.append(Edge((y+1,y+2),(v,x+3)))
        g.edges.append(Edge((y+2,y+0),(v,x+5)))
        g.edges.append(Edge((vfaces[1],y+0),(x+0,x+1)))
        g.edges.append(Edge((vfaces[1],y+1),(x+1,x+2)))
        g.edges.append(Edge((vfaces[2],y+1),(x+2,x+3)))
        g.edges.append(Edge((vfaces[2],y+2),(x+3,x+4)))
        g.edges.append(Edge((vfaces[0],y+2),(x+4,x+5)))
        g.edges.append(Edge((vfaces[0],y+0),(x+5,x+0)))
        # add new vertices
        g.verts.append(Vert([vedges[0],z+3,z+8],[vfaces[0],vfaces[1],y+0]))
        g.verts.append(Vert([z+4,z+0,z+3],[vfaces[1],y+1,y+0]))
        g.verts.append(Vert([vedges[1],z+5,z+4],[vfaces[1],vfaces[2],y+1]))
        g.verts.append(Vert([z+6,z+1,z+5],[vfaces[2],y+2,y+1]))
        g.verts.append(Vert([vedges[2],z+7,z+6],[vfaces[2],vfaces[0],y+2]))
        g.verts.append(Vert([z+8,z+2,z+7],[vfaces[0],y+0,y+2]))
        # modify old faces (affected)
        g.faces[vfaces[0]].R4exp(v,[x+0,x+5,x+4],[z+8,z+7]) # formerly +7 +8
        g.faces[vfaces[1]].R4exp(v,[x+2,x+1,x+0],[z+4,z+3])
        g.faces[vfaces[2]].R4exp(v,[x+4,x+3,x+2],[z+6,z+5])
        # modify old edges (affected)
        g.edges[vedges[0]].repl_vert(v,x+0)
        g.edges[vedges[1]].repl_vert(v,x+2)
        g.edges[vedges[2]].repl_vert(v,x+4)
        # replace the target vertex
        g.verts[v] = Vert([z+0,z+1,z+2],[y+0,y+1,y+2])
        # update coloring and history, return
        if v in g.black:
            g.black += [x+0,x+2,x+4]
        else :
            g.black += [x+1,x+3,x+5]
        g.hist.append(('r4exp',v))
        g.face_colors = None
        g.vertex_colors = None
        return 0

    # check that all edges, faces, and vertices agree       
    def check_agreement(g):
        bad_edge_face = []
        bad_face_vert = []
        bad_vert_edge = []
        # edges
        for e, edge in enumerate(g.edges):
            # faces
            for f in edge.faces:
                if e not in g.faces[f].edges:
                    bad_edge_face.append((e,f))
            #verts
            for v in edge.verts:
                if e not in g.verts[v].edges:
                    bad_vert_edge.append((v,e))
        # faces
        for f, face in enumerate(g.faces):
            # verts
            for v in face.verts:
                if f not in g.verts[v].faces:
                    bad_face_vert.append((f,v))
            # edges
            for e in face.edges:
                if f not in g.edges[e].faces:
                    bad_edge_face.append((e,f))
        # verts
        for v, vert in enumerate(g.verts):
            # edges
            for e in vert.edges:
                if v not in g.edges[e].verts:
                    bad_vert_edge.append((v,e))
            # faces
            for f in vert.faces:
                if v not in g.faces[f].verts:
                    bad_face_vert.append((f,v))
        return (bad_edge_face, bad_face_vert, bad_vert_edge)

    # check that the graph is really c3cbp/Barnette/kektup
    def check_barnette(g):
        # check cubic
        # TODO
        # check bipartite
        # TODO
        # check 3-connected? - check if faces having two common edges
        # TODO
        # check planar? - check if, um... huh.
        # TODO
        return 1

    # return a list of edge pairs elligible for R0 expansion
    def get_elligible_edge_pairs(g, order_matters = True):
        pairs = []
        for e1 in range(len(g.edges)):
            for f in g.edges[e1].faces:
                for e2 in g.faces[f].edges:
                    if e2 <= e1 and not order_matters:
                        continue
                    if g.r0exp_eli(e1,e2):
                        pairs.append((e1,e2))
        return pairs

    # check if an edge set is a hamiltonian cycle on g
    def is_ham_cycle(g, edge_set):
        # make sure it has the right number of elements
        if len(edge_set) != len(g.verts):
            return False
        unused_edges = set(edge_set)
        # make sure that there are two edges for vertex 0
        zero_edges = unused_edges.intersection(g.verts[0].edges)
        if len(zero_edges) != 2:
            return False
        # remove and remember one of the 0-vertices
        last_edge = zero_edges.pop()
        unused_edges.remove(last_edge)
        v = 0
        # follow the cycle and make sure it goes back to vertex 0
        for _ in range(len(g.verts)-1):
            next_edges = unused_edges.intersection(g.verts[v].edges)
            if len(next_edges) != 1:
                return False
            next_edge = next_edges.pop()
            unused_edges.remove(next_edge)
            v = g.edges[next_edge].get_other_vert(v)
        if g.edges[last_edge].get_other_vert(0) != v:
            return False
        return True

    # converts a permeating tree to a hamiltonian cycle (edges)
    def permeating_tree_to_ham(g, face_set):
        nedges = len(g.edges)
        ecount = [0 for i in range(nedges)]
        for f in face_set:
            for e in g.faces[f].edges:
                ecount[e] += 1
        return set([e for e in range(nedges) if ecount[e] == 1])

    # converts a hamiltonian cycle (edges) to pair of permeating subtrees
    def ham_to_permeating_tree(g, ham):
        A, B = set([0]), set(range(len(g.faces)))
        B.remove(0)
        q = Queue()
        q.put(0)
        while not q.empty() :
            a0 = q.get()
            for e in g.faces[a0].edges:
                if e in ham :
                    continue
                else :
                    a1 = g.edges[e].get_other_face(a0)
                    if a1 in A:
                        continue
                    A.add(a1)
                    B.remove(a1)
                    q.put(a1)
        return A, B

    # re-color the graph
    def make_vertex_coloring(g):
        # set up
        half_nverts = len(g.verts)/2
        black = [0]
        black_set = set([0])
        white_set = set([])
        q = Queue()
        q.put(0)
        count = 0
        while not q.empty():
            v = q.get()
            # iterate over 1st neighbors
            for e1 in g.verts[v].edges:
                white = g.edges[e1].get_other_vert(v)
                if white in white_set:
                    continue
                white_set.add(white)
                # iterate over 2nd neighbors
                for e2 in g.verts[white].edges:
                    w = g.edges[e2].get_other_vert(white)
                    if w in black_set:
                        continue
                    black_set.add(w)
                    q.put(w)
        return black_set, white_set

    def get_vertex_coloring(g, doit = False):
        if g.vertex_colors is None or doit:
            g.vertex_colors = g.make_vertex_coloring()
        return g.vertex_colors

    # get a 3coloring on the dual of the graph
    # start0 defines color 0. start1 defines color 1, adjacent to start0
    def make_face_coloring(g, start0=0, start1=-1):
        if start1 == -1:
            start1 = g.edges[g.faces[start0].edges[0]].get_other_face(start0)
        colors = (set([start0]), set([start1]), set())
        colored = set([start0, start1])
        tovisit = Queue()
        tovisit.put(start0)
        while not tovisit.empty():
            face = tovisit.get()
            # get the color of the face
            for ix in range(3):
                if face in colors[ix]:
                    face_color = ix
                    break
            # get adjacent faces
            adjacent = [g.edges[e].get_other_face(face)
                            for e in g.faces[face].edges]
            # get the color of even and odd parity adjacent faces
            even_color = (face_color+1)%3; odd_color = (face_color+2)%3 # guess
            for ix, adj in enumerate(adjacent):
                if adj in colored:
                    if (ix&1) != (adj in colors[odd_color]): # check the guess
                            even_color, odd_color = odd_color, even_color
                    break
            # put the right color in all of the adjacent faces
            for ix, adj in enumerate(adjacent):
                if adj in colored:
                    continue
                if ix&1:
                    colors[odd_color].add(adj)
                else :
                    colors[even_color].add(adj)
                tovisit.put(adj)
                colored.add(adj)
        return colors

    def get_face_coloring(g, doit=False):
        if g.face_colors is None or doit:
            g.face_colors = g.make_face_coloring()
        return g.face_colors

    # build the graph from which g came
    def get_parent(g):
        return Kektup(hist = g.hist[-1:])

    # rename the vertices, edges, and faces of g
    def permute(g, verts, edges, faces):
        # ensure that permutations are the right length
        if len(verts) != len(g.verts):
            return 1
        if len(edges) != len(g.edges):
            return 1
        if len(faces) != len(g.faces):
            return 1
        # find inverse permutations and ensure that all values are present
        inv_verts = []
        for v in range(len(verts)):
            if v not in verts:
                return 1
            else :
                inv_verts.append(verts.index(v))
        inv_edges = []
        for e in range(len(edges)):
            if e not in edges:
                return 1
            else :
                inv_edges.append(edges.index(e))
        inv_faces = []
        for f in range(len(faces)):
            if f not in faces:
                return 1
            else :
                inv_faces.append(faces.index(f))
        # permute all of the verts, edges, faces
        for vert in g.verts:
            vert.permute(edges, faces)
        for edge in g.edges:
            edge.permute(faces, verts)
        for face in g.faces:
            face.permute(verts, edges)
        # reorder the verts, edges, faces
        new_verts = []
        for v in inv_verts:
            new_verts.append(g.verts[v])
        g.verts = new_verts
        new_edges = []
        for e in inv_edges:
            new_edges.append(g.edges[e])
        g.edges = new_edges
        new_faces = []
        for f in inv_faces:
            new_faces.append(g.faces[f])
        g.faces = new_faces
        # check that the renaming is consistent
        try :
            g.check_agreement()
        except :
            print ("Bad transformation!")
        return 0
