"""
viektup.py
by Ted Morin

viektup <= visual interactive kektup

a class for visualizing Barnette graphs via interactive visualization
"""
import numpy as np
from vektup import Vektup
from iektup import Iektup

class Viektup(Vektup):
    def __init__(self, g=None,
            showing_points=True, 
            showing_lines=True,
            showing_polys=True,
            showing_labels=True):
        Vektup.__init__(self, g=g,
            showing_points=showing_points, 
            showing_lines=showing_lines,
            showing_polys=showing_polys,
            showing_labels=showing_labels)
        self.selected = ()
        self.face_color_mode = 'proper'
        self.fig.canvas.mpl_connect('pick_event', self.pick_event)

        # use a more sophisticated update_visual method after the first call
        self.update_visual = self.eventual_update_visual

    def init_visual(self):
        Vektup.init_visual(self)
        self.indicator = self.ax.annotate("Pending", 
                                np.array([.9,.9]))
        

    def eventual_update_visual(self, face_color_mode = None):
        if face_color_mode is not None:
            self.face_color_mode = face_color_mode
        if self.face_color_mode == 'cycle':
            self.cycle_color_faces()
        elif self.face_color_mode == 'proper':
            self.properly_color_faces()
        elif self.face_color_mode == 'ham':
            pass # case handled below
        else :
            pass

        is_ham = self.g.is_ham_cycle(self.g.active_edges)
        if is_ham:
            self.indicator.set_text('Hamiltonian')
            self.indicator.set_color('g')
            if self.face_color_mode == 'ham':
                self.inside_outside_color_faces(self.g.active_edges)
        else :
            self.indicator.set_text('Not Hamiltonian')
            self.indicator.set_color('r')
        Vektup.update_visual(self)

    # event to call when an edge is clicked on (or "picked")
    def pick_event(self, event):
        # make sure it is not just a vertex moving action
        if event.mouseevent.button == 3:
            return
        # figure out who originated the event
        if hasattr(event.artist, 'edge_num'):
            self.edge_pick_event(event)
        elif hasattr(event.artist, 'face_num'):
            self.face_pick_event(event)
        elif hasattr(event.artist, 'vert_num'):
            return
            #self.vert_pick_event(event)
        self.update_visual()

    def edge_pick_event(self, event):
        # assumes that event originated with a line
        edge_num = event.artist.edge_num
        if event.mouseevent.key == 'control':
            # select/deselect the edge
            self.select(event.artist.edge_num)
        else :
            # add to/remove from active edges
            if edge_num in self.g.active_edges:
                self.g.active_edges.remove(edge_num)
            else :
                self.g.active_edges.add(edge_num)

    def face_pick_event(self, event):
        face_num = event.artist.face_num
        for edge_num in self.g.faces[face_num].edges:
            if edge_num in self.g.active_edges:
                self.g.active_edges.remove(edge_num)
            else :
                self.g.active_edges.add(edge_num)
        self.update_visual()
            # print("Face Event by", face_num)
            # self.faces[face_num].set_visible(False)

    # handle details of selecting an edge
    def select(self, edge_num):
        if edge_num in self.selected: # remove the edge
            if len(self.selected) == 1:
                self.selected = ()
            else :
                ix = self.selected.index(edge_num)
                self.selected = self.selected[:ix] + self.selected[ix+1:]
        else :
            if self.selected:
                self.selected = (self.selected[-1], edge_num)
            else :
                self.selected = (edge_num,)
        self.update_visual()

    def cycle_color_faces(self, update = False):
        for f in range(len(self.g.faces)):
            if self.g.face_has_alternating_edges(f, self.g.active_edges):
                self.face_colors[f] = 'g'
            elif self.g.face_has_all_edges(f, self.g.active_edges):
                self.face_colors[f] = 'r'
            else :
                self.face_colors[f] = 'y'
        if update :
            self.update_visual()

    def inside_outside_color_faces(self, ham, update = False, 
                                    inside_color='r', outside_color='g'):
        A, B = self.g.ham_to_permeating_tree(ham)
        for f in range(len(self.g.faces)):
            if f in A :
                self.face_colors[f] = inside_color
            else :
                self.face_colors[f] = outside_color
        if update :
            self.update_visual()

    # for VR? I do not know what this is doing
    def color_edges(self):
        for e, edge in enumerate(self.edges):
            if self.edge_counts[e] == 1:
                edge.set_color('b')
                edge.set_linewidth(self.active_line_width)
            elif self.edge_counts[e] == 0:
                edge.set_color('g')
                edge.set_linewidth(self.default_line_width)
            elif self.edge_counts[e] == 2:
                edge.set_color('r')
                edge.set_linewidth(self.default_line_width)

    # for VR? I do not know what this is doing
    def flip_colors(self):
        for e, edge in enumerate(self.edges):
            if self.edge_counts[e] == 0:
                if self.flip_flag == True:
                    edge.set_color('r')
                else:
                    edge.set_color('g')
                edge.set_linewidth(self.default_line_width)
            elif self.edge_counts[e] == 2:
                if self.flip_flag == True:
                    edge.set_color('g')
                else:
                    edge.set_color('r')
        self.flip_flag = not(self.flip_flag)

if __name__ == '__main__':

    v = Viektup(g = Iektup.random_graph(30, 0))
    v.show_tutte_embedding(f=9)

    input("exit on enter")
