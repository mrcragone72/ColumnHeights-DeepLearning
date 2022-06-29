import numpy as np

import random

from ase.build import bulk
from ase import Atoms
from ase.io import read



class ATK_Random_HEA():

    def __init__(self,path,spatial_domain,random_transl,t_xy,random_rot,a_y):

        self.path = path

        self.spatial_domain  = spatial_domain

        self.random_transl = random_transl
        
        self.t_xy = t_xy

        self.random_rot = random_rot
        
        self.a_y = a_y

    def get_model(self):

        self.model = read(self.path)
        
        a = random.choice(self.a_y)

        if self.random_rot:

            self.model.rotate(v = 'y', a = a, center = 'COP')

        self.model.rotate(v = 'z', a = random.random() * 360, center='COP')

        self.model.center(vacuum=0)

        cell = self.model.get_cell()

        size = np.diag(cell)

        self.model.set_cell((self.spatial_domain[0],) * 3)

        self.model.center()

        if self.random_transl:

            tx = np.random.uniform(-self.spatial_domain[0] * self.t_xy, self.spatial_domain[0] * self.t_xy)
            ty = random.uniform(-self.spatial_domain[1] * self.t_xy, self.spatial_domain[1] * self.t_xy)

            self.model.translate((tx, ty, 0))

        return self.model
    
    def random_mix(self,elements):

        cs = np.array(self.model.get_chemical_symbols())

        random_cs_idx = []

        for e in elements:

            cs_idx = np.where(cs == e)[0]
            random_cs_idx.append(cs_idx)
            
        random_cs_idx = np.concatenate(random_cs_idx)
        random_cs = cs[random_cs_idx]
        random.shuffle(random_cs)
        cs[random_cs_idx] = random_cs

        self.model.set_chemical_symbols(cs)


class Random_NP(object):

    directions = np.array(

        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0],
         [0, -1, 0], [0, 0, -1], [1, 1, 1], [-1, 1, 1],
         [1, -1, 1], [1, 1, -1], [-1, -1, 1], [-1, 1, -1],
         [1, -1, -1], [-1, -1, -1]]
    )

    directions = (directions.T / np.linalg.norm(directions, axis=1)).T

    def __init__(self,crystal_structure,
                 material,
                 lc, 
                 random_size,
                 spatial_domain,
                 random_transl,
                 t_xy,
                 random_rot,a_y):


        self.material = material

        self.lc = lc

        self.crystal_structure = crystal_structure

        self.random_size = random_size

        self.spatial_domain = spatial_domain

        self.sites = self.get_sites()

        self.bonds = self.get_bonds()

        self.random_transl = random_transl

        self.t_xy = t_xy

        self.random_rot = random_rot

        self.a_y = a_y

    def get_sites(self):

        grid_size = 22

        cluster = bulk(self.material, self.crystal_structure, a = self.lc, cubic = True)

        cluster *= (grid_size,) * 3

        cluster.center()

        self.center = np.diag(cluster.get_cell()) / 2

        positions = cluster.get_positions()

        return positions

    def get_bonds(self):

        bond_length = 3.91 / np.sqrt(2)

        bonds = []

        for i, s in enumerate(self.sites):

            distances = np.linalg.norm(self.sites - s, axis=1)

            indices = np.where(distances <= bond_length * 1.05)[0]

            bonds.append(indices)

        return bonds

    def create_seed(self, lengths100, lengths111):

        self.active = np.ones(len(self.sites), dtype=bool)

        lengths = np.hstack((lengths100, lengths111))

        for length, direction in zip(lengths, self.directions):

            r0 = self.center + length * direction

            for i, site in enumerate(self.sites):

                if self.active[i]:

                    self.active[i] = np.sign(np.dot(direction, site - r0)) == -1

        self.active_bonds = np.array([self.active[b] for b in self.bonds],dtype = object)

        self.available_sites = np.where([any(ab) & (not a) for ab, a in zip(self.active_bonds, self.active)])[0]

    def build(self, grid_size, T0, T1=None):

        if T1 is None:

            T1 = T0

        for i in range(grid_size):

            T = T0 + (T1 - T0) * i / grid_size

            coordination = self.get_coordination(self.available_sites)

            p = np.zeros_like(coordination, dtype=np.float)

            p[coordination > 2] = np.exp(coordination[coordination > 2] / T)

            p = p / float(np.sum(p))

            p[p < 0] = 0

            n = np.random.choice(len(p), p=p)

            k = self.available_sites[n]

            self.available_sites = np.delete(self.available_sites, n)

            self.expand(k)

    def expand(self, k):

        self.active[k] = True

        new_avail = self.bonds[k][self.active[self.bonds[k]] == 0]

        self.available_sites = np.array(list(set(np.append(self.available_sites, new_avail))))

        if len(new_avail) > 0:

            to_update = np.array([np.where(self.bonds[x] == k)[0] for x in new_avail]).T[0]

            for i, j in enumerate(to_update):

                self.active_bonds[new_avail][i][j] = True

    def get_coordination(self, sites):

        return np.array([sum(self.active_bonds[site]) for site in sites])

    def get_cluster(self):

        return Atoms([self.material] * len(self.sites[self.active]), self.sites[self.active])

    def get_model(self):

        radius = self.random_size/2

        lengths100 = np.random.uniform(radius, radius + .2 * radius, 6)

        lengths111 = np.random.uniform(radius, radius + .2 * radius, 8)

        self.create_seed(lengths100, lengths111)

        self.build(int(np.sum(self.active) / 4.), 10, 2)

        self.model = self.get_cluster()

        a = random.choice(self.a_y)

        if self.random_rot:
            self.model.rotate(v='y', a=a, center='COP')

        self.model.rotate(v='z', a=random.random() * 360, center='COP')

        self.model.center(vacuum=0)

        cell = self.model.get_cell()

        size = np.diag(cell)

        self.model.set_cell((self.spatial_domain[0],) * 3)

        self.model.center()

        if self.random_transl:
            tx = np.random.uniform(-self.spatial_domain[0] * self.t_xy, self.spatial_domain[0] * self.t_xy)
            ty = random.uniform(-self.spatial_domain[1] * self.t_xy, self.spatial_domain[1] * self.t_xy)

            self.model.translate((tx, ty, 0))

        return self.model



