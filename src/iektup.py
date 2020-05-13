"""
iektup.py
by Ted Morin (and Rob Hochberg and Cameron Nottingham and Therese Aglialoro)

iektup <= interactive kektup

class for exploring Barnette graphs via various extensions of kektup
"""
from queue import Queue
import time
from kektup import *
import random
from collections import Counter
# import PolygonFuncs   # for VR?
from collections import OrderedDict
import csv 
from xml.etree.ElementTree import Element, SubElement, Comment, tostring 
from xml.etree import ElementTree
from xml.dom import minidom

class Iektup(Kektup):
    # init
    def __init__(g, hist=[], vef=()):
        Kektup.__init__(g, hist=hist, vef=vef)
        g.active_edges = set()
        g.holder = []  # for use with VR?
        g.hams = []
        g.hams_up_to_date = True

    def r0exp(g,e1,e2):
        val = Kektup.r0exp(g,e1,e2)
        g.hams_up_to_date = val
        return val

    def r4exp(g,v):
        val = Kektup.r4exp(g,v)
        g.hams_up_to_date = val
        return val

    #returns a Counter conatining the list of n edge faces and their counts
    def face_info(self):
        face_count = []
        for face in self.faces:
            face_count.append(len(face.edges))
            count = Counter(face_count)
        return count

    def edge_count(self):
        return len(self.edges)

    #returns count of the vertices
    def vert_count(self):
        return len(self.verts)

    #returns count of the faces, based on Euler's formula 
    def face_count(self):
        return (self.edge_count() + 2) - self.vert_count()

    #Pretty prints the XML for easy human reading
    def prettify(self, elem):
        rough_string = ElementTree.tostring(elem, 'utf-8')
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    #Generate XML document for use in VR environment 
    def generateXML(self):
        tree = ElementTree.ElementTree()
        root = Element("graphml")
        subroot = SubElement(root, "graph")
        for i in range(len(self.g.verts)):
            node = SubElement(subroot, 'node')

            x = round(self.vert_pos[i][1][0], 4)
            z = round(self.vert_pos[i][1][1], 4)
            y = round(((x ** 2) + (z ** 2)), 4)

            node.set('id', 'node_%d' % i)
            node.set('x', str(x))
            node.set('y', str(y))
            node.set('z', str(z))
        for i in range(len(self.g.edges)):
            edge = SubElement(subroot, 'edge')
            edge.set('id', 'link_%d' % i)
            edge.set('source', 'node_%d' % self.g.edges[i].verts[0])
            edge.set('target', 'node_%d' % self.g.edges[i].verts[1])
            if i in self.active_edges and not self.active_edges.empty():
                edge.set('color', 'green')
            else:
                edge.set('color', 'blue')
        print(self.prettify(root))
        tree._setroot(root)
        xmlstr = minidom.parseString(tostring(root)).toprettyxml(indent="   ")
        with open("barnette.xml", "w") as f:
            f.write(xmlstr)

    def face_has_alternating_edges(g, face_num, edge_set):
        # check even edges are the same
        even_edges = g.faces[face_num].edges[::2]
        val_even = (even_edges[0] in edge_set)
        for e in even_edges:
            if (e in edge_set) != val_even:
                return 0
        # check odd edges are the same
        odd_edges = g.faces[face_num].edges[1::2]
        val_odd = (odd_edges[0] in edge_set)
        for e in odd_edges:
            if (e in edge_set) != val_odd:
                return 0
        # check that they are different
        if val_even == val_odd:
            return 0
        return 1

    def face_has_all_edges(g, face_num, edge_set):
        for e in g.faces[face_num].edges:
            if e not in edge_set:
                return 0
        return 1

    # inverts the edges on one face of a Barnette graph.
    def flip_face_edges(g, face_num, edge_set):
        for e in g.faces[face_num].edges:
            if e in edge_set :
                edge_set.remove(e)
            else :
                edge_set.add(e)

    # groups unused edges into inside and outside for some ham cycle
    # inside contains face 0
    def ham_inside_outside_edges(g, ham):
        # ham = set(ham) # (in case ham is given as a LIST of edges)
        visited_faces = set([0])
        inside_edges = set()
        q = Queue()
        q.put(0)
        while not q.empty() :
            f0 = q.get()
            for e in g.faces[f0].edges:
                if e in ham :
                    continue
                else :
                    f1 = g.edges[e].get_other_face(f0)
                    if f1 in visited_faces:
                        continue
                    inside_edges.add(e)
                    visited_faces.add(f1)
                    q.put(f1)
        outside_edges = set(range(len(self.g.edges))).difference(ham)
        outside_edges = outside_edges.difference(inside_edges)
        return inside_edges, outside_edges

    # get a hamiltonian cycle on the graph (by desired technique if specified)
    def get_hams(self, technique = None, timed = 0, doit = False):
        if (len(self.hams)==0) or (doit==True) or (not self.hams_up_to_date):
            if timed:
                t1 = time.clock()
            if technique == 'vertex_coloring' :
                self.hams = self.make_hams_by_vertex_coloring()
                self.hams_up_to_date = True
            elif technique == 'faces' :
                self.hams = self.make_hams_by_faces()
                self.hams_up_to_date = True
            elif technique == 'face_coloring' :
                self.hams = self.hunt_hams_by_face_color()
            elif technique == 'face_coloring_greedy' :
                self.hams = [self.hunt_hams_by_face_color_oneoff()]
            else :
                self.hams = self.make_hams_by_faces()
            if timed:
                t2 = time.clock()
                print(t2-t1)
        return self.hams

################################################################################
###############                  EXPERIMENTAL                    ###############
################################################################################

    """
    # make a disjoint cycle covering
    def make_layer_cycles(self, f=0):
        cycles = [self.faces[f].verts]
        q = Queue()
        q.put(cycles[-1])
        visited = set(cycle[-1])  # in a cycle
        done = set() #= set(cycles[-1])
        while not q.empty():
            cycle = q.get()
            # the next layer inside the cycle
            start = cycle[0]
            a, b = cycle[:2]
            b_neighbors = self.get_neighbor_verts(b)
            c = b_neighbors[(b_neighbors.index(a)-1)%3]
            if c in cycle:
                c = 
            c =  = ix = 0
            start = cycle[0]
            a = start
            B = [v for v in self.get_neighbor_verts(a) if v not in visited]
            b = [b for b in B if b not in cycle][0]
            new_b = 




            q.put()
    """

################################################################################
################################################################################
################################################################################
###############      HAMILTONIAN CYCLE FINDERS ARE BELOW         ###############
################################################################################
################################################################################
################################################################################

    # find all hamiltonian cycles in g
    def make_hams_by_vertex_coloring(g):
        # numbers and structures
        hams = []
        black_set, _ = g.make_vertex_coloring()
        blacks = list(black_set)
        nverts = len(g.verts)
        nedges = len(g.edges)
        nblack = len(blacks)
        ham_cyc_pairings = []
        unused_reds = set(range(nverts))
        unused_reds.difference_update(blacks)
        black_edg_ix = [0 for i in range(nblack)]
        edg_by_black = [nedges for i in range(nblack)]
        red_by_black = [nverts for i in range(nblack)]

        # back-tracking through edge-choices for each black vertex
        done = False
        i = 0
        while not done:
            while i < nblack:
                while black_edg_ix[i] == 3:
                    if i == 0:
                        done = True
                        break
                    # ascend to previous black vertex
                    black_edg_ix[i] = 0
                    i -= 1
                    unused_reds.add(red_by_black[i])
                    black_edg_ix[i] += 1
                if done:
                    break
                e = g.verts[blacks[i]].edges[black_edg_ix[i]]
                v = g.edges[e].get_other_vert(blacks[i])
                if v in unused_reds:
                    # descend to next black vertex
                    unused_reds.remove(v)
                    edg_by_black[i] = e
                    red_by_black[i] = v
                    i += 1
                else :
                    # stay on the level
                    black_edg_ix[i] += 1
            if done :
                break
            # check that cycle visits all vertices
            bad_edges = set(edg_by_black)
            visited = set()
            good_cycle = True
            v = 0
            if black_edg_ix[v] == 0:
                prev = g.verts[v].edges[1]
            else :
                prev = g.verts[v].edges[0]
            for _ in range(nverts-1):
                visited.add(v)
                e = set(g.verts[v].edges)
                e.remove(prev)
                e.difference_update(bad_edges)
                e = e.pop()
                v = g.edges[e].get_other_vert(v)
                if v in visited:
                    good_cycle = False
                    break
                prev = e
            if good_cycle:
                cycle = [e for e in range(nedges) 
                                    if e not in bad_edges]
                hams.append(cycle)
            # ascend to previous black vertex
            i -= 1
            unused_reds.add(red_by_black[i])
            black_edg_ix[i] = 3
        return hams

    # Uses the fact that the faces inside a HC form an induced tree
    # to search for HCs
    def make_hams_by_faces(g, start_face=0):
        hams = []
        # get_3_col_stats = (start_face == -2)
        if start_face < 0:  # This means select a face with 4 edges
            start_face = [i for i in range(len(g.faces)) if len(g.faces[i].edges) == 4][0]
            
        selected = [start_face]
        g.boundary = [g.edges[e].get_other_face(start_face) for e in g.faces[start_face].edges] + [-1]*len(g.faces)
        g.bcount = [(1 if f in g.boundary else 0) for f in range(len(g.faces))] 
        count = len(g.faces[start_face].edges)
        g.bcount[start_face] = 2
        g.make_hams_by_faces_visit(selected, 0, count, count, hams)
        # if get_3_col_stats: # This means gather face-3-color stats
        #     fcols = g.get_dual_3coloring()
        #     fcoltuples = set()
        #     for h in hams:
        #         colcount = tuple(map(lambda x : 1 if len(x.intersection(set(h)))>0 else 0, fcols))
        #         fcoltuples.add(colcount)
        #     print(fcoltuples)
       
        # hams here are lists of faces. Convert to lists of edges
        return [g.permeating_tree_to_ham(h) for h in hams]

    # sel is the induced tree of faces so far
    # bdry is the set of unselected faces adjacent to a selected face
    # start is the index into the boundary list from which to start the loop
    # end is the index after which the loop should end
    # c is the number of edges in the boundary cycle so far
    # hams is the list of found induced face trees
    def make_hams_by_faces_visit(g, sel, start, end, c, hams):
        # If we found a HC, celebrate, append it and return
        if c == len(g.verts):
            hams.append(sel)
            return

        # Otherwise, loop over boundary starting at given index
        for i in range(start, end):
            f = g.faces[g.boundary[i]]

            if g.bcount[g.boundary[i]] == 1:
                my_end = end
                for e in f.edges:
                    otherf_index = g.edges[e].get_other_face(g.boundary[i])
                    if g.bcount[otherf_index] == 0: # New to the boundary. Welcome!
                        g.boundary[my_end] = otherf_index
                        my_end += 1
                    g.bcount[otherf_index] += 1 
                g.make_hams_by_faces_visit(sel+[g.boundary[i]], i+1, my_end, c+len(f.edges)-2, hams)
                for e in f.edges:
                    g.bcount[g.edges[e].get_other_face(g.boundary[i])] -= 1

    # get ONE hamiltonian cycle by dfc backtracking 
    # (maybe update to find more than one?)
    # (formerly "dfc_backtracking(g)" by Dr. Hochberg)
    def hunt_hams_by_face_color(g):
        face_colors = g.get_face_coloring()
        cycle = set()

        # iterate over all colors
        for face_color in face_colors:
            usable = set(range(len(g.faces))).difference(face_color)
            # iterate over all starting faces
            for face in face_color:
                # set the starting face
                cycle = set(g.faces[face].edges)
                result = g.hunt_hams_by_face_color_visit(cycle, face_color, usable)
                if len(result) == len(g.verts):
                    return [result]
        return []

    # (formerly "dfc_backtracking_visit" by Dr. Hochberg)
    def hunt_hams_by_face_color_visit(g, cycle, dfc, usable):
        # get the eligible faces
        if len(cycle) == len(g.verts): # We found a Hamilton cycle!
            return cycle
        
        eligible = [f for f in usable if 
                      (len(cycle.intersection(g.faces[f].edges)) == 1)]
        eligible.sort(key=lambda x : len(g.faces[x].verts))
        
        if len(eligible) == 0:
            return [] # No way to extend this branch
        
        for f in eligible:
            # get the face that touches the most eligible face
            contact_edge = cycle.intersection(g.faces[f].edges).pop()
            contact_face = g.edges[contact_edge].get_other_face(f)
            # expand the cycle through the most_eligible face
            to_flip = set([g.edges[e].get_other_face(f) for e in 
                           g.faces[f].edges])
            to_flip = to_flip.intersection(dfc)
            to_flip.remove(contact_face)
            to_flip.add(f)
            #print("faces being flipped:", to_flip)
            new_cycle = cycle.copy() # make a copy
            for face in to_flip:
                g.flip_face_edges(face, new_cycle)

            # See if this new thing creates a HC.
            result = g.hunt_hams_by_face_color_visit(new_cycle, dfc, usable)
            if len(result) == len(g.verts): # We found a Hamilton cycle!
                return result

        # If we make it here, no cycle was found
        return []


    # tries to make a hamiltonian cycle by joining faces of a color
    # (formerly the "dfc bigfirst non dfc start" method)
    def hunt_hams_by_face_color_oneoff(g, dfc = None):
        if dfc == None :
            dfc, _, _ = g.get_face_coloring()
        #print("dfc:", dfc)
        # mark the non_dfc faces
        non_dfc = set(range(len(g.faces))).difference(dfc)
        # start the cycle with the edges on that face
        cycle = set()
        # get the best eligible face
        biggest_non_dfc = -1
        max_degree = 0
        for face in non_dfc:
            degree = len(g.faces[face].verts)
            if degree > max_degree:
                biggest_non_dfc = face
                max_degree = degree
        #print("chosen face:", most_eligible)
        # start the cycle through the biggest_non_dfc face
        to_flip = set([g.edges[e].get_other_face(biggest_non_dfc) for e in 
                            g.faces[biggest_non_dfc].edges])
        to_flip = to_flip.intersection(dfc)
        to_flip.add(biggest_non_dfc)
        #print("faces being flipped:", to_flip)
        for face in to_flip:
            g.flip_face_edges(face, cycle)
        # initialize the eligible set
        eligible = [f for f in non_dfc if 
                      (len(cycle.intersection(g.faces[f].edges)) == 1)]
        #print("starting face options:", eligible)
        while eligible:
            # get the best eligible face
            most_eligible = -1
            max_degree = 0
            for face in eligible:
                degree = len(g.faces[face].verts)
                if degree > max_degree:
                    most_eligible = face
                    max_degree = degree
            #print("chosen face:", most_eligible)
            # get the face that touches the most eligible face
            contact_edge = cycle.intersection(g.faces[most_eligible].edges).pop()
            contact_face = g.edges[contact_edge].get_other_face(most_eligible)
            # expand the cycle through the most_eligible face
            to_flip = set([g.edges[e].get_other_face(most_eligible) for e in 
                                g.faces[most_eligible].edges])
            to_flip = to_flip.intersection(dfc)
            to_flip.remove(contact_face)
            to_flip.add(most_eligible)
            #print("faces being flipped:", to_flip)
            for face in to_flip:
                g.flip_face_edges(face, cycle)
            # check if there is a hamiltonian cycle
            if len(cycle) == len(g.verts):# g.is_ham_cycle(cycle):
                # since this method always gives a cycle, g.is_ham_cycle is optional
                return cycle
            # get the eligible faces
            eligible = [f for f in non_dfc if 
                          (len(cycle.intersection(g.faces[f].edges)) == 1)]
            #print("cycle:", cycle)
            #print("eligible faces:", eligible)
        return cycle
