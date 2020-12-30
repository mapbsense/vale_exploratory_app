# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:15:34 2014

@author: Ian
"""

import unittest
import os
import math

import numpy as np
from PIL import Image

# import geosoft.gxpy.system as gsys
# import geosoft.gxpy.gx as gxp
# import geosoft.gxpy.gdb as gxgdb
import mvar

# import pylab as plt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def progress(s, pct, som):
    print('{}>'.format(pct), s)
    if som is not None:
        print(mvar.separations(som.som))
        print(som.density())

def show_results(images, maps, xyzlabels):
    """
    Show the result as images

    images - list of images to display
    maps - list of maps to display
    xyzlabels - list of labels for X, Y and Z axis

    returns a mapplotlib figure that can be displayed
    """

    fig = plt.figure(1)
    cols = max(len(images), len(maps))
    rows = 0
    if len(images) > 0:
        rows += 1
    if len(maps) > 0:
        rows += 1

    # images
    for i in range(len(images)):
        plt.subplot(rows, cols, i + 1)
        plt.title(images[i][0])
        plt.imshow(images[i][1], interpolation=images[i][2])

    # SOM map
    mapp = maps[0]
    nn = len(mapp[1])
    n2 = nn // 2
    ax = fig.add_subplot(rows, cols, cols + 1, projection='3d')
    ax.scatter(mapp[1][:n2, 0],
               mapp[1][:n2, 1],
               mapp[1][:n2, 2],
               s=40, c=mapp[2][:n2, :], marker='+')
    ax.scatter(mapp[1][n2:, 0],
               mapp[1][n2:, 1],
               mapp[1][n2:, 2],
               s=40, c=mapp[2][n2:, :], marker='o')
    ax.set_xlabel(xyzlabels[0])
    ax.set_ylabel(xyzlabels[1])
    ax.set_zlabel(xyzlabels[2])
    plt.title(mapp[0])

    # data
    mapp = maps[1]
    ax = fig.add_subplot(rows, cols, cols + 2, projection='3d')
    ax.scatter(mapp[1][:, 0],
               mapp[1][:, 1],
               mapp[1][:, 2],
               s=20, c=mapp[2], marker='o')
    ax.set_xlabel(xyzlabels[0])
    ax.set_ylabel(xyzlabels[1])
    ax.set_zlabel(xyzlabels[2])
    plt.title(mapp[0])

    return (plt)


class TestSOM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gxPy = gxp.GXpy()

    @classmethod
    def start(cls, test):
        print("\n*** {} ***".format(test))
        os.chdir(os.path.dirname(__file__))

    def test_init(self):

        # test som creation
        data = np.random.random_sample((2000, 4))
        t = mvar.SOM(data, 25)
        self.assertEqual(t.som.shape[0], 25)
        self.assertEqual(t.som.shape[1], 4)
        self.assertEqual(t.dim, 5)
        del (t)

        try:
            t = mvar.SOM(data, 0)
            del (t)
        except:
            self.assertRaises(ValueError)

        try:
            t = mvar.SOM(data, 1000)
            del (t)
        except:
            self.assertRaises(ValueError)

        try:
            t = mvar.SOM(data, 7.0)
            del (t)
        except:
            self.assertRaises(ValueError)

    @classmethod
    def start(cls, test):
        print("\n*** {} ***".format(test))

    def test_som_levels(self):
        self.start(gsys.func_name())

        data = np.random.random_sample((500, 4))
        t = mvar.SOM(data, 9, rate=0.9999)
        self.assertEqual(t.som.shape, (9, 4))

        t = mvar.SOM(data, 9, rate=0.9999, levels=0)
        self.assertEqual(t.som.shape, (9, 4))

        data = np.random.random_sample((500, 4))
        t = mvar.SOM(data, 9, rate=0.9999, levels=1)
        self.assertEqual(t.som.shape, (18, 4))

        data = np.random.random_sample((5000, 4))
        t = mvar.SOM(data, 9, rate=0.9999, levels=3, percent=50)
        self.assertEqual(t.som.shape, (36, 4))

    def test_dim(self):
        self.start(gsys.func_name())

        dimList = mvar.SOM.list_dim()
        self.assertEqual(len(dimList), 15)
        self.assertEqual(dimList[0], 4)
        self.assertEqual(dimList[14], 256)

    def test_map(self):
        self.start(gsys.func_name())

        data = np.random.random_sample((1000, 6)) * 100
        t = mvar.SOM(data, 16)
        index = mvar.classify_data(t.som, data)
        self.assertTrue((index.max() < (t.dim * t.dim)))
        self.assertTrue((index.min() >= 0))
        eud = mvar.euclidean_distance(t.som[index], data)
        self.assertNotEqual(np.std(eud), 0.0)

    def test_pic_euclidean(self):
        self.start(gsys.func_name())

        def progress(s, pct=None, som=None):
            print('{}>'.format(pct), s, som)

        # read an image and put it in a data array
        im = Image.open('testdata\\mini_jeff.jpg')
        im.thumbnail((400, 400), Image.ANTIALIAS)
        im.save('testdata\\original.png')
        image_in = np.asarray(im, dtype=np.float32)

        height, width, depth = image_in.shape

        # scale up to byte range 0-255
        if image_in.max() <= 1:
            image_in *= 255
        data = image_in.reshape((width * height, depth))

        # som process
        somdim = 25
        levels = 1
        som = mvar.SOM(data, somdim, rate=0.9999, focus=1000, weight=0.01, levels=levels, percent=0, progress=progress, \
                       similarity=mvar.bmu_euclidean)  # create the SOM
        classMap = mvar.classify_data(som.som, data)
        classMap = som.som[classMap]
        Image.fromarray(np.uint8(classMap)).save("testdata\\som.png")

        # simple report - scale index map to colour range 0-255
        ss = som.som / 255.0

        show_results([("original", data.reshape((height, width, depth)).astype('B'), 'bilinear'), \
                      ("anomalous", classMap.reshape((height, width, depth)).astype('B'), 'bilinear')], \
                     [('som', ss, ss), ('som', ss, ss)], \
                     xyzlabels=('r', 'g', 'b')).show()

    def test_pic_cosine(self):
        self.start(gsys.func_name())

        # read an image and put it in a data array
        im = Image.open('testdata\\mini_jeff.jpg')
        im.thumbnail((400, 400), Image.ANTIALIAS)
        im.save('testdata\\original.png')
        image_in = np.asarray(im, dtype=np.float32)

        height, width, depth = image_in.shape

        # scale up to byte range 0-255
        if image_in.max() <= 1:
            image_in *= 255
        data = image_in.reshape((width * height, depth))

        # som process
        somdim = 25
        levels = 1
        som = mvar.SOM(data, somdim, rate=0.9999, focus=100, weight=0.01, levels=levels, percent=0, progress=progress, \
                       similarity=mvar.bmu_cosine)  # create the SOM
        classMap = mvar.classify_data(som.som, data)
        classMap = som.som[classMap]
        Image.fromarray(np.uint8(classMap)).save("testdata\\som.png")

        # simple report - scale index map to colour range 0-255
        ss = som.som / 255.0

        show_results([("original", data.reshape((height, width, depth)).astype('B'), 'bilinear'), \
                      ("anomalous", classMap.reshape((height, width, depth)).astype('B'), 'bilinear')], \
                     [('som', ss, ss), ('som', ss, ss)], \
                     xyzlabels=('r', 'g', 'b')).show()

    def test_gdb(self):
        self.start(gsys.func_name())

        def progress(s, pct=None, som=None):
            print('{}>'.format(pct), s, som)

        with gxgdb.Geosoft_gdb.new('testdata\\gdbSOMtest.gdb') as gdb:

            # read an image and put it in a new database
            im = Image.open('testdata\\mini_jeff.jpg')
            im.thumbnail((200, 200), Image.ANTIALIAS)
            image_in = np.asarray(im, dtype=np.float32)
            gdb.new_channel('R', dtype=np.int)
            gdb.new_channel('G', dtype=np.int)
            gdb.new_channel('B', dtype=np.int)
            for l in range(image_in.shape[0]):
                gdb.write_line('L{}'.format(l), image_in[l, :, :], channels=['R', 'G', 'B'])

            # select first line only
            gdb.select_lines('', True)
            # gdb.select_lines('L0',True)

            # som process
            som = mvar.SOMgdb(gdb,
                              ['R', 'G', 'B'],
                              dim=9,
                              progress=progress,
                              class_err=['cls', 'eud'])

    '''
    def test_big_gdb(self):
        self.start(gsys.func_name())

        def progress(s, pct=None, som=None):
            print('{}>'.format(pct), s, som)

        gdb = gxgdb.Geosoft_db.open(self.gxPy, 'E:\\Projects\\Simandou\\SimSouth200_Model.gdb')

        # som process
        som = mvar.SOMgdb(gdb, \
                          ['b_x', 'b_y', 'b_z'], \
                          dim=4, \
                          progress=progress, \
                          class_err=['c_no', 'c_no_eud'])
    '''

    def test_gdb2(self):
        self.start(gsys.func_name())

        def progress(s, pct=None, som=None):
            print('{}>'.format(pct), s, som)

        with gxgdb.Geosoft_gdb.open('testdata\\gdbSOMtest2.gdb') as gdb:
            gdb.select_lines('', True)
            # gdb.select_lines('L0',True)

            # som process
            som = mvar.SOMgdb(gdb, \
                              ['R', 'G', 'B'], \
                              dim=9, \
                              progress=progress, \
                              class_err=['cls', 'eud'])

    def test_euclidean(self):
        self.start(gsys.func_name())

        cls = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0.25, .5, -.3], [-0.1, -99.0, 4000]])
        self.assertEqual(mvar.bmu_euclidean(cls, [1, 0, 0]), 0)
        self.assertEqual(mvar.bmu_euclidean(cls, [0, 1, 0]), 1)
        self.assertEqual(mvar.bmu_euclidean(cls, [0, 0, 1]), 2)
        self.assertEqual(mvar.bmu_euclidean(cls, [0, 0, 0]), 3)
        self.assertEqual(mvar.bmu_euclidean(cls, [0.5, 0.5, -.1]), 4)
        self.assertEqual(mvar.bmu_euclidean(cls, [0, -80, 3000]), 5)

        data = np.array([[1, 2, 3], [0, 0, 3990]])
        c = mvar.classify_data(cls, data, similarity=mvar.bmu_euclidean)
        self.assertEqual(c.shape, (2,))
        self.assertEqual(c[0], 2)
        self.assertEqual(c[1], 5)

        d = mvar.euclidean_distance(cls[c], data)
        self.assertEqual(d.shape, (2,))
        self.assertEqual(d[0], 3.0)
        self.assertEqual(d[1], math.sqrt(0.1 * 0.1 + 99 * 99 + 100))

    def test_cosine(self):
        self.start(gsys.func_name())

        cls = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0.25, .5, -.3], [-0.1, -99.0, 4000]])
        amp = mvar.amp_squared(cls)
        self.assertEqual(mvar.bmu_cosine(cls, [1, 0, 0]), 0)
        self.assertEqual(mvar.bmu_cosine(cls, [0, 1, 0]), 1)
        self.assertEqual(mvar.bmu_cosine(cls, [0, 0, 0]), 3)
        self.assertEqual(mvar.bmu_cosine(cls, [0.5, 0.5, -.1], amp=amp), 4)
        self.assertEqual(mvar.bmu_cosine(cls, [0, -80, 3000]), 5)

    def test_separations(self):
        self.start(gsys.func_name())

        cls = np.array(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0.25, .5, -.3], [-0.1, -99.0, 4000], [1, 2, 3], [1, 2, 4],
             [4, 3, 2]]).reshape(3, 3, 3)
        sep = mvar.separations(cls)
        self.assertEqual(sep.shape, (2, 2))
        self.assertAlmostEqual(sep[0, 0], 9.20767829e-01)
        self.assertAlmostEqual(sep[1, 0], 2.49788735)
        self.assertAlmostEqual(sep[0, 1], 2000.9527874407552)
        self.assertAlmostEqual(sep[1, 1], 2002.29935587)

    def test_similarity_functions(self):
        self.start(gsys.func_name())

        sims = mvar.similarity_functions()
        self.assertEqual(sims.index('Euclidean distance'), 0)
        self.assertTrue(sims.index('Cosine (direction)') >= 0)

        cls = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0.25, .5, -.3], [-0.1, -99.0, 4000]])
        data = np.array([[1, 2, 3], [0, 0, 3990]])
        c = mvar.classify_data(cls, data, similarity='Euclidean distance')
        self.assertEqual(c.shape, (2,))
        self.assertEqual(c[0], 2)
        self.assertEqual(c[1], 5)

        cls = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0.25, .5, -.3], [-0.1, -99.0, 4000]])
        c = mvar.classify_data(cls, data, similarity='Cosine (direction)')
        self.assertEqual(c.shape, (2,))
        self.assertEqual(c[0], 3)
        self.assertEqual(c[1], 3)

        try:
            c = mvar.classify_data(cls, data, similarity="this does not exist")
            self.assertTrue(False)
        except mvar.MvarException:
            self.assertTrue(True)


    def test_normalize(self):
        self.start(gsys.func_name())

        data = np.array([-1.0,0.0,0.1,0.2,0.5,1.0,1.5,10,50,1000,10000])
        d1 = data.copy()
        spec = mvar.normalize(d1, mvar.NormType.none)
        self.assertEqual(d1[0], data[0])
        self.assertEqual(d1[-1], data[-1])
        self.assertEqual(spec[0], mvar.NormType.none)
        self.assertTrue(spec[1] is None)
        self.assertTrue(spec[2] is None)
        self.assertTrue(spec[3] is None)
        mvar.denormalize(d1, spec)
        self.assertEqual(d1[0], data[0])
        self.assertEqual(d1[-1], data[-1])

        spec = mvar.normalize(d1, mvar.NormType.normal)
        self.assertAlmostEqual(d1.mean(),0.0)
        self.assertAlmostEqual(d1.std(),1.0)
        self.assertEqual(spec[0], mvar.NormType.normal)
        self.assertAlmostEqual(spec[1], 1005.66363636)
        self.assertAlmostEqual(spec[2], 2858.4428645)
        self.assertAlmostEqual(spec[3], None)
        mvar.denormalize(d1,spec)
        self.assertAlmostEqual(d1[0], data[0])
        self.assertAlmostEqual(d1[-1], data[-1])

        d1 = data.copy()
        spec = mvar.normalize(d1, mvar.NormType.lognormal)
        self.assertAlmostEqual(d1.mean(),0.0)
        self.assertAlmostEqual(d1.std(),1.0)
        self.assertEqual(spec[0], mvar.NormType.lognormal)
        self.assertAlmostEqual(spec[1], 0.583458378247)
        self.assertAlmostEqual(spec[2], 4.53837937528)
        self.assertAlmostEqual(spec[3], 0.0028584428645)
        mvar.denormalize(d1,spec)
        self.assertAlmostEqual(d1[0], 0.0028584428645)
        self.assertAlmostEqual(d1[-1], data[-1])

        d1 = data.copy()
        spec2 = mvar.normalize(d1, spec)
        self.assertAlmostEqual(d1.mean(), 0.0)
        self.assertAlmostEqual(d1.std(), 1.0)
        self.assertEqual(spec[0], mvar.NormType.lognormal)
        self.assertAlmostEqual(spec2[1], 0.583458378247)
        self.assertAlmostEqual(spec2[2], 4.53837937528)
        self.assertAlmostEqual(spec2[3], 0.0028584428645)

        d1 = data.copy()
        spec = mvar.normalize(d1, (mvar.NormType.normal, 500, 0.1, None))
        self.assertAlmostEqual(d1.mean(), 5056.63636364)
        self.assertAlmostEqual(d1.std(), 28584.428645)
        self.assertEqual(spec[0], mvar.NormType.normal)
        self.assertAlmostEqual(spec[1], 500)
        self.assertAlmostEqual(spec[2], 0.1)
        self.assertAlmostEqual(spec[3], None)
        mvar.denormalize(d1,spec)
        self.assertAlmostEqual(d1[0], data[0])
        self.assertAlmostEqual(d1[-1], data[-1])

##############################################################################################
if __name__ == '__main__':
    unittest.main()
