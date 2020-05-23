"""
shektup.py
by Ted Morin

shektup <= shell kektup

the shell class for managing a vektup/viektup
""" 


import sys, cmd, os, random
import copy
import pickle
import numpy as np

from sys import maxsize, platform
from iektup import Iektup
from viektup import Viektup

# these should all be handled indirectly in iektup/vektup, right?
# import csv 
# from xml.etree.ElementTree import Element, SubElement, Comment, tostring 
# from xml.etree import ElementTree
# from xml.dom import minidom
# import matplotlib.pyplot as plt

class Shektup(cmd.Cmd):
    def __init__(self, v = None):
        cmd.Cmd.__init__(self)
        if v == None:
            self.v = Vektup()
        else :
            self.v = v

    def do_tu(self, arg): 
        'Show tutte embedding of the graph'
        try :
            f = int(arg)
            self.v.show_tutte_embedding(f=f)
        except :
            self.v.show_tutte_embedding()

    def do_sp(self,arg): # full command should be "spring"
        'Show spring embedding of the graph'
        try :
            f = int(arg)
            self.v.show_spring_embedding(f=f)
        except :
            self.v.show_spring_embedding()

    def do_r0exp(self,arg):
        'Apply the R0 expansion on the edges given'
        try :
            pair = tuple(map(int, arg.strip().split()))
        except :
            pair = ()
        if len(pair) != 2:
            print("must indicate two edges by index.")
            return
        nedges = len(self.v.g.edges)
        if (not 0 <= pair[0]  < nedges) or (not 0 <= pair[1] < nedges):
            print("Invalid edge choice: {}".format(arg))
            return
        if self.v.r0exp(*pair):
            print("Edge pair not eligible: {}".format(arg))

    def do_r4exp(self,arg):
        'Apply the R4 expansion on the edges given'
        try :
            vert = int(arg.strip())
        except :
            vert = -1
        if not 0 <= vert < len(v.g.verts):
            print("Invalid vertex choice: {}".format(arg))
            return
        self.v.r4exp(vert)

    def do_facecolor(self, arg):
        '''
        Set the face coloring mode.
        
        Options:
        "proper" - proper 3-coloring of the faces
        "cycle"  - color faces based on which edges are active
        "ham"    - color inside and outside of any active ham
        "none"   - do not update face coloring
        '''
        if arg not in ['proper', 'cycle', 'ham', 'none']:
            print("No valid coloring specified. No action.")
        elif arg == 'proper':
            self.v.face_color_mode = 'proper'
        elif arg == 'cycle':
            self.v.face_color_mode = 'cycle'
        elif arg == 'ham':
            self.v.face_color_mode = 'ham'
        elif arg == 'none':
            self.v.face_color_mode = None
        else :
            pass
        self.v.update_visual()

    def do_ham(self, arg):
        '''
        Get hamiltonian cycles. 

        Cycle finding options:
        faces, vertex_coloring, face_coloring, face_coloring_oneoff

        Other options:
        "doit" - force re-computation
        "notime" - suppress timing info
        '''
        args = arg.strip().split()
        timed = (0 if 'notime' in args else 1)
        doit = (1 if 'doit' in args else 0)
        technique = (args[0] if args else '')
        print(technique)
        if technique not in ['vertex_coloring', 'faces', 'face_coloring',
                                                 'face_coloring_oneoff']:
            technique = None
            print("No valid technique specified, using 'faces'")
        if timed:
            print("Timing hamiltonian cycle acquisition. Search time:")
        print(technique)
        self.v.g.get_hams(technique = technique, timed = timed, doit = doit)
        print("{} hamiltonian cycles found.".format(len(self.v.g.hams)))

    def do_sh(self,arg): # full command should be "showham"
        'Display the hamiltonian cycle specified (default to 0)'
        if arg == '':
            num = 0
        else :
            try :
                num = int(arg.strip())
            except :
                print("Invalid cycle number. Defaulting to 0.".format(arg))
                num = 0
        if 0 <= num < len(self.v.g.hams):
            self.v.g.active_edges = copy.copy(self.v.g.hams[num])
            self.v.update_visual()
        else :
            print("Not enough cycles are known. Find them via 'ham' command.")

    def do_nameshuffle(self, arg):
        'Shuffle the names of the things?'
        # TODO make these lists random
        rand_verts = [(x+1)%len(g.g.verts) for x in range(len(g.g.verts))]
        rand_edges = [(x+1)%len(g.g.edges) for x in range(len(g.g.edges))]
        rand_faces = [(x+1)%len(g.g.faces) for x in range(len(g.g.faces))]
        self.v.g.permute(rand_verts,rand_edges,rand_faces)
        temp = np.zeros((len(self.v.point_pos),2)) # TODO this could be done better
        for ix, vert in enumerate(rand_verts):
            temp[vert] = self.v.point_pos[ix]
        self.v.point_pos = temp
        self.v.outside = rand_faces[self.v.outside]
        self.v.update_visual()

    # information methods

    def do_faces(self,arg):
        'Display info about faces'
        count = self.v.g.face_info()
        for fi, num in sorted(count.items()):
            print("F" + str(fi) + ": " + str(num))

    def do_info(self,arg):
        'Display information about the graph'
        print("Edges: " + str(self.v.g.edge_count()))
        print("Vertices: " + str(self.v.g.vert_count()))
        print("Total Faces: " + str(self.v.g.face_count()))
        count = self.v.g.face_info()
        for fi, num in sorted(count.items()):
            print("   F" + str(fi) + ": " + str(num))
        # print("Hamilton Circuits: " + str(len(self.v.g.get_ham_cycles())))


    def do_flipface(self,arg):
        'Flip activation status of the edges on the face'
        try :
            targets = map(int, arg.strip().split())
        except :
            return
        for t in targets:
            self.v.g.flip_face_edges(t, self.v.g.active_edges)
        self.v.update_visual()

    def do_label(self,arg):
        self.v.toggle_labels()

    # starting/reseting/saving
    def do_save(self,arg):
        'Save the graph'
        with open('graph.g', 'wb') as gobj:
            pickle.dump(self.v.g, gobj, protocol = 2)
            gobj.close()

    def do_xml(self,arg):
        'Output XML'
        self.v.g.generateXML()

    def do_clear(self,arg):
        'Clear the prompt'
        os.system('clear')
        # if platform == 'linux' or platform == 'linux2' or platform =='darwin':
        #     os.system('clear')
        # elif platform == 'win32':
        #     os.system('cls')

    # TODO
    def do_reset(self,arg):
        'Reset the graph to default kektup'
        self.v.redraw_visual(Iektup())
        self.do_clear('')

    def do_newrand(self,arg):
        'Generates a new random graph with N vertices'
        try :
            sizeseed = tuple(map(int, arg.strip().split()))
            if len(sizeseed) == 2:
                size, seed = sizeseed
            else :
                size = sizeseed[0]
                seed = None
        except :
            print("Invalid size and seed input: {}".format(arg))
            return
        # if seed != None: seed = random.randrange(maxsize)
        if seed is None: seed = random.randrange(sys.maxsize)
        self.v.redraw_visual(Iektup.random_graph(size, seed))
        self.do_clear('')
        print("seed = {}".format(seed))

    def do_load(self,arg):
        with open('graph.g', 'rb') as gobj:
            g = pickle.load(gobj)
        self.v.redraw_visual(g.g)

    def do_exit(self,arg):
        'Exit the program'
        return True

    # shortcuts

    def do_quit(self,arg):
        'Alternate for exit'
        return self.do_exit(arg)

    def do_0(self,arg):
        'Shortcut to r0exp'
        return self.do_r0exp(arg)

    def do_4(self,arg):
        'Shortcut to r4exp'
        return self.do_r4exp(arg)

    def do_ff(self,arg):
        'Shortcut to faceflip'
        return self.do_flipface(arg)

    def postloop(self):
        'print out the history of the graph'
        for step in self.v.g.hist:
            print(step)
        print("{} steps.".format(len(self.v.g.hist)))

if __name__ == '__main__':
    v = Viektup(g = Iektup.random_graph(30, 0))
    v.show_tutte_embedding(f=9)
    Shektup(v=v).cmdloop()
