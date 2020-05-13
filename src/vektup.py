"""
vektup.py
by Ted Morin

vektup <= visual kektup

a class for visualizing Barnette graphs via visualization
"""
import numpy as np
import sys
import random as rand
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import time
from iektup import Iektup
from math import sqrt

# Takes two ordered pairs and finds the distance between them
def euc_dist(v, w):
    return sqrt((v[0]-w[0])**2 + (v[1]-w[1])**2)

# a class to visualize a barnette graph
class Vektup:
    def __init__(self, g=None,
            showing_points=True, 
            showing_lines=True,
            showing_polys=True,
            showing_labels=True):
        # make the graph
        if g == None:
            self.g = Iektup()
        else :
            self.g = g
        # drawing properties
        self.showing_labels = showing_labels
        self.showing_points       = showing_points
        # self.showing_point_labels = showing_labels
        self.default_vert_color = 'k'
        self.vert_colors = {}

        self.showing_polys       = showing_polys
        # self.showing_poly_labels = showing_labels
        self.default_face_color = 'g'
        self.face_colors = {}

        self.showing_lines       = showing_lines
        # self.showing_line_labels = showing_labels
        self.edge_colors = {}
        self.default_line_width = 2
        self.default_line_color = 'w'
        self.active_line_width = 4
        self.active_line_color = 'k'
        # set up visual
        self.fig = plt.figure(figsize=(10,10))
        self.ax = self.fig.add_subplot(111)
        # set up positions
        self.outside = 5
        self.init_visual()
        
    def init_visual(self):
        self.ax.cla()
        # self.ax.margins(x=.01,y=.01) #this just controls margins within space
        self.ax.axis('off')
        plt.ion()
        self.fig.show()
        # locations where notes should go for all points, lines, polygons
        self.point_pos = self.get_tutte_embedding(self.outside)
        # self.line_midpoints = self.make_line_midpoints() # TODO
        # self.poly_averages = self.make_poly_averages() # TODO
        # the point, line, and polygons from matplotlib.pyplot
        self.points = []
        self.lines  = []
        self.polys  = []
        # annotations from matplotlib.pyplot
        self.vert_notes = []
        self.edge_notes = []
        self.face_notes = []
        # mpl_connections
        self.dragging_ind = None
        self.fig.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.fig.canvas.mpl_connect('button_release_event', self.button_release_callback)
        self.fig.canvas.mpl_connect('motion_notify_event', self.motion_notify_callback)
        # various not understood
        self.test_points = []       # for use with VR?
        self.holder = []            # for use with VR? 
        self.flip_flag = True       # for use with VR?
        self.edge_locations = []    # for use with VR?
        self.test_points = []       # for use with VR?
        self.add_to_visual(len(self.g.verts), len(self.g.edges), len(self.g.faces))

    # add edges and annotations to the plot - give number of verts, edges, faces
    def add_to_visual(self, nv, ne, nf):
        #determine whether the edge annotations should be invisible or not
        label_alpha = int(self.showing_labels)
        # annotate and draw the new vertices
        nverts = len(self.g.verts)
        for ix in range(nverts - nv, nverts):
            self.vert_notes.append(self.ax.annotate(ix, self.point_pos[ix], 
                                            visible = label_alpha))
            posx = self.point_pos[ix][0]
            posy = self.point_pos[ix][1]
            self.points.append(self.ax.plot([posx],[posy], 'o', 
                                    color=self.default_vert_color,
                                    picker=3, zorder=2)[0])
            self.points[-1].vert_num = ix
        # annotate and draw the new edges
        nedges = len(self.g.edges)
        for ix in range(nedges - ne, nedges):
            edge = self.g.edges[ix]
            posa = self.point_pos[edge.verts[0]]
            posb = self.point_pos[edge.verts[1]]
            self.edge_notes.append(
                         self.ax.annotate(ix,.5*(posa+posb), color = 'k',
                            alpha= label_alpha))
            posa = tuple(posa)
            posb = tuple(posb)
            self.lines.append(self.ax.plot(
                    [posa[0],posb[0]],[posa[1],posb[1]], 'k-',
                    picker=3, zorder=1)[0])
            self.lines[-1].set_linewidth(self.default_line_width)
            self.lines[-1].edge_num = ix
        # annotate and draw the new faces
        nfaces = len(self.g.faces)
        for ix in range(nfaces-nf, nfaces):
            face = self.g.faces[ix]
            avg = np.array([0.,0.])
            for v in face.verts:
                avg += self.point_pos[v]
            avg /= len(face.verts)
            self.face_notes.append(self.ax.annotate(ix, avg ,color = 'k', 
                                                        alpha= label_alpha))
            if ix == self.outside :
                positions = [[-1, -1], [-1,-.95],[-.95,-.95], [-.95,-1]]
            else :
                positions = [self.point_pos[jj] for jj in self.g.faces[ix].verts]
            self.polys.append(Polygon(positions, 'w', picker=0))
            self.ax.add_patch(self.polys[-1])
            self.polys[-1].face_num = ix
        self.update_visual()

    # update the graph
    def update_visual(self):
        label_alpha = int(self.showing_labels)

        # update the vertex positions and annotations
        for ix, p in enumerate(self.point_pos):
            self.vert_notes[ix].set_position(p)
            self.vert_notes[ix].set_alpha(label_alpha)
            pos = self.point_pos[ix]
            self.points[ix].set_data([pos[0]],[pos[1]])
            if ix in self.vert_colors :
                self.points[ix].set_color(self.vert_colors[ix])
            else :
                self.points[ix].set_color('k')

        # update the edge positions and annotations
        for ix, e in enumerate(self.g.edges):
            # TODO incorporate dependence on line_midpoints
            posa = self.point_pos[e.verts[0]]
            posb = self.point_pos[e.verts[1]]
            self.edge_notes[ix].set_position(.5*(posa+posb))
            self.edge_notes[ix].set_alpha(label_alpha)
            posa = tuple(posa)
            posb = tuple(posb)
            self.lines[ix].set_data([posa[0],posb[0]],[posa[1],posb[1]])
            if ix in self.g.active_edges:
                self.lines[ix].set_linewidth(self.active_line_width)
                self.lines[ix].set_color(self.active_line_color)
            else :
                self.lines[ix].set_linewidth(self.default_line_width)
                self.lines[ix].set_color(self.default_line_color)
            if ix in self.edge_colors:
                self.lines[ix].set_color(self.edge_colors[ix])

        # update the face positions and annotations
        for ix, face in enumerate(self.g.faces):
            # TODO incorportate dependence on poly_averages
            # handle the outside face
            if ix == self.outside :
                positions = [[-1, -1], [-1,-.95],[-.95,-.95], [-.95,-1]]
                avg = np.array([-.99,-.99])
            else :
                positions = [self.point_pos[jj] for jj in self.g.faces[ix].verts]
                avg = sum(positions) / len(face.verts)
            # annotation
            self.face_notes[ix].set_position(avg)
            self.face_notes[ix].set_alpha(label_alpha)
            # filling
            self.polys[ix].set_xy(positions)
            if ix in self.face_colors:
                self.polys[ix].set_facecolor(self.face_colors[ix])
            else :
                self.polys[ix].set_facecolor(self.default_face_color)
        self.fig.canvas.draw()

    def redraw_visual(self, newGraph):
        self.g = newGraph
        self.outside = 0
        self.point_pos = self.get_tutte_embedding(self.outside)
        self.init_visual()

    def toggle_labels(self):
        self.showing_labels = not self.showing_labels
        self.update_visual()

    # call the r0exp method on g and update the visual
    def r0exp(self,e1,e2):
        # check for elligibility
        if not self.g.r0exp_eli(e1,e2):
            return 1
        # get positions
        # get common face f and abort if e1 and e2 are not elligible
        f = self.g.get_common_face(e1,e2)
        # get indices of e1 and e2 in f, check that they are an evenly spaced
        e1_f_ix = self.g.faces[f].edges.index(e1)
        e2_f_ix = self.g.faces[f].edges.index(e2)
        # make positions for the new vertices
        newpos = np.zeros((4,2))
        deg = len(self.g.faces[f].verts) # mod by the degree of the face
        e1posa = self.point_pos[self.g.faces[f].verts[e1_f_ix]]
        e1posb = self.point_pos[self.g.faces[f].verts[(e1_f_ix+1)%deg]]
        e2posa = self.point_pos[self.g.faces[f].verts[e2_f_ix]]
        e2posb = self.point_pos[self.g.faces[f].verts[(e2_f_ix+1)%deg]]
        newpos[0] += .66667*e1posa + .33333*e1posb
        newpos[1] += .33333*e1posa + .66667*e1posb
        newpos[2] += .66667*e2posa + .33333*e2posb
        newpos[3] += .33333*e2posa + .66667*e2posb
        self.point_pos = np.vstack((self.point_pos,newpos))
        # apply expansion to the graph
        self.g.r0exp(e1,e2)
        # update the visual
        self.add_to_visual(4,6,2)
        return 0

    # call the R4exp method on g and update the visual
    def r4exp(self,v):
        # ensure that v is in the range
        if v < 0 or v > len(self.g.verts):
            # v is out of range
            return 1
        # make positions for the new vertices
        newpos = np.zeros((6,2))
        neighbors = [self.g.edges[e].get_other_vert(v) 
                            for e in self.g.verts[v].edges]
        for ix in range(3):
            newpos[2*ix] +=.5*(self.point_pos[v]+self.point_pos[neighbors[ix]])
        for pair in ((0,2),(2,4),(4,0)):
            newpos[pair[0]+1] = .5*(newpos[pair[0]]+newpos[pair[1]])
        self.point_pos = np.vstack((self.point_pos,newpos))
        # apply expansion to the graph
        self.g.r4exp(v)
        # update the visual
        self.add_to_visual(6,9,3)
        return 0

    # credit and many thanks to:
    # https://github.com/jadenstock/Tutte-embedding
    def get_tutte_embedding(self,f):
        # produce positions of a tutte embedding
        g = self.g
        face = g.faces[f]
        pos = np.zeros((len(g.verts),2))
        # put vectices of f around a circle
        ang = 2*np.pi/len(face.verts)
        cosang = np.cos(ang)
        sinang = np.sin(ang)
        mat = np.array([[cosang,sinang],[-sinang,cosang]])
        vec = np.array([0.,-1.])
        for v in face.verts:
            pos[v] = vec
            vec = np.matmul(mat,vec)
        # get inner verts
        inner_verts = set(range(len(g.verts)))
        inner_verts.difference_update(face.verts)
        inner_verts = list(inner_verts)
        size = len(inner_verts)
        # create the system of equations that will give inner vert positions.
        # values are indexed by inner_vert indices
        Xmat = np.identity(size)
        Ymat = np.identity(size)
        XYvec = np.zeros((size,2))
        for i, u in enumerate(inner_verts):
            neighbors = [g.edges[e].get_other_vert(u) for e in g.verts[u].edges]
            inverse_n = .333333333333333
            for v in neighbors:
                if v in face.verts:
                    XYvec[i] += np.array(pos[v])*inverse_n
                else:
                    j = inner_verts.index(v)
                    Xmat[i][j] = -inverse_n
                    Ymat[i][j] = -inverse_n
        Xvec, Yvec = np.split(XYvec, 2, axis=1)
        # solve
        x = np.linalg.solve(Xmat, Xvec)
        y = np.linalg.solve(Ymat, Yvec)
        # update values and return
        for i, u in enumerate(inner_verts):
            pos[u] = np.array([x[i][0],y[i][0]])
        # print pos
        return pos

    def show_tutte_embedding(self, f=-1):
        # set graph to the tutte embedding
        if 0 <= f < len(self.g.faces):
            self.outside = f
        self.point_pos = self.get_tutte_embedding(self.outside)
        # update graph
        self.update_visual()

    # Take a planar (Tutte) embedding and make it prettier by using a
    # spring embedding
    def spring_embedding(self, tpos):
        g = self.g
        for i, v in enumerate(g.verts):
            v.label = i

        # Find average edge length
        avg = sum([euc_dist(tpos[e.verts[0]], tpos[e.verts[1]]) \
                   for e in g.edges]) / len(g.edges)
        # Compute force on each vertex, and treat as movement
        move = []
        scale = 0.1
        for i, v in enumerate(g.verts):
            # Find average edge length
            avg = sum([euc_dist(tpos[g.edges[e].verts[0]], tpos[g.edges[e].verts[1]]) \
                       for e in v.edges]) / len(v.edges)
            dx = 0
            dy = 0
            for ei in v.edges:
                e = g.edges[ei]
                x1, y1 = tpos[i]
                x2, y2 = tpos[e.get_other_vert(v.label)]
                dist = euc_dist(tpos[i], tpos[e.get_other_vert(v.label)])
                force = dist - avg
                dx += scale * force * (x2 - x1) / dist
                dy += scale * force * (y2 - y1) / dist
            move.append([x1 + dx, y1 + dy])

        # Re-scale so that it fits on the screen
        maxx, maxy, minx, miny = -10, -10, 10, 10
        for x, y in move:
            if x > maxx: maxx = x
            if y > maxy: maxy = y
            if x < minx: minx = x
            if y < miny: miny = y
        move = [[-1+2*(x-minx)/(maxx-minx), -1+2*(y-miny)/(maxy-miny)] \
                for x, y in move]
        return np.array(move)
        
    def show_spring_embedding(self, f=-1):
        # set graph to the spring embedding
        # First we get a tutte embedding, as a starting point
        if 0 <= f < len(self.g.faces):
            self.outside = f
        pos = self.get_tutte_embedding(self.outside)
        for _ in range(25):
            pos = self.spring_embedding(pos)
        self.point_pos = pos
        # update graph
        self.update_visual()

    # credit to:
    # https://matplotlib.org/3.1.1/gallery/event_handling/poly_editor.html
    def get_ind_under_point(self, event): # modified from poly_editor
        'get the index of the vertex under point if within epsilon tolerance'

        # credit to:
        # https://stackoverflow.com/questions/13662525/
        #     how-to-get-pixel-coordinates-for-matplotlib-generated-scatterplot
        # Get the x and y data and transform it into pixel coordinates
        x = np.array([point.get_data()[0] for point in self.points])
        y = np.array([point.get_data()[1] for point in self.points])
        xy_pixels = self.ax.transData.transform(np.hstack([x,y]))
        xpix, ypix = xy_pixels.T

        d = np.hypot(xpix - event.x, ypix - event.y)
        indseq, = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= 10:
            ind = None

        return ind

    def button_press_callback(self, event): # modified from poly_editor
        'whenever a mouse button is pressed'
        if not self.showing_points:
            return
        if event.inaxes is None:
            return
        if event.button != 3:
            return
        self.dragging_ind = self.get_ind_under_point(event)

    def button_release_callback(self, event): # modified from poly_editor
        'whenever a mouse button is released'
        if not self.showing_points:
            return
        if event.button != 3:
            return
        self.dragging_ind = None
        self.update_visual()

    def motion_notify_callback(self, event):
        'on mouse movement'
        if not self.showing_points:
            return
        if self.dragging_ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 3:
            return
        x, y = event.xdata, event.ydata

        self.point_pos[self.dragging_ind] = x, y
        self.points[self.dragging_ind].set_data([x],[y])

        # self.fig.canvas.restore_region(self.background)
        # self.ax.draw_artist(self.poly)
        # self.fig.canvas.restore_region(self.canvas.copy_from_bbox(self.ax.bbox))
        self.ax.draw_artist(self.points[self.dragging_ind])
        self.fig.canvas.blit(self.ax.bbox)

    def properly_color_faces(self, update=False, c1='r', c2='b', c3='g'):
        R, B, G = self.g.get_face_coloring()
        for f in range(len(self.g.faces)):
            if f in R:
                self.face_colors[f] = c1
            elif f in B:
                self.face_colors[f] = c2
            else : # elif f in G:
                self.face_colors[f] = c3
        if update:
            self.update_visual()

    def show_vertex_coloring(self, update=False, c1='k', c2='w'):
        B, W = self.g.get_vertex_coloring()
        for v in range(len(self.g.verts)):
            if v in B:
                self.vert_colors[v] = c1
            else : # elif v in W:
                self.vert_colors[v] = c2
        if update:
            self.update_visual()

if __name__ == '__main__':

    v = Vektup()

    print("drag vertices via right-click and drag")

    input("test r0exp on enter")
    v.r0exp(0,2)

    input("test r4exp on enter")
    v.r4exp(4)

    input("test new tutte_embedding on enter")
    v.show_tutte_embedding()

    input("test label toggle on enter")
    v.toggle_labels()
    input()
    v.toggle_labels()

    input("test tutte on other face on enter")
    v.show_tutte_embedding(f=4)

    input("test spring embedding on enter")
    v.show_spring_embedding()

    input("test newrand 30 on enter")
    v.redraw_visual(Iektup.random_graph(30, 0))
    v.show_tutte_embedding(f=9)

    input("test properly_color_faces on enter")
    v.properly_color_faces(update = True)

    input("test show_vertex_coloring on enter")
    v.show_vertex_coloring(update = True)

    input("exit on enter")
