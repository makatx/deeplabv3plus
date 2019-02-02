import openslide
import numpy as np
from PIL import Image
import xml.etree.cElementTree as ET
import cv2
import matplotlib.pyplot as plt
import os

class Annotation:

    scaleFactor = 1
    coords_orig = []
    coords_order = []
    coords_list = []
    bounds = []
    bounds_orig = []

    def __init__(self, filename, scaleFactor=1):
        self.scaleFactor = scaleFactor
        with open(filename, 'rb') as f:
            self.root = ET.parse(f)
        self.coords_orig = []
        self.coords_order = []
        self.group = []
        self.type = []

        for annot in self.root.iter('Annotation'):
            coords_tag = annot.find('Coordinates')
            lst = []
            for coord in coords_tag.findall('Coordinate'):
                lst.append([float(coord.attrib['Order']), float(coord.attrib['X']), float(coord.attrib['Y'])])
            n = np.array(lst)
            n = n[n[:,0].argsort()]
            self.coords_orig.append(n[:,1:])
            self.coords_order.append(n)
            self.group.append(annot.attrib['PartOfGroup'])
            self.type.append(annot.attrib['Type'])

        self.coords_list = self.scale(factor=scaleFactor)
        self.calcBounds()

    def scale(self, coords=None, factor=1):
        if coords == None: coords = self.coords_orig
        coords_scaled = []
        for n in range(len(coords)):
            coords_scaled.append((coords[n] / factor).astype(np.int));
        return coords_scaled

    def shift(self, coords=None, origin=(0,0)):
        if coords == None: coords = self.coords_orig
        shifted = []
        origin = np.array(origin)
        for n in coords:
            shifted.append(n - origin)
        return shifted

    def calcBounds(self):
        bounds = []
        for n in self.coords_list:
            xmin = n[:,0].min()
            ymin = n[:,1].min()
            xmax = n[:,0].max()
            ymax = n[:,1].max()
            bounds.append(np.array([xmin,ymin,xmax,ymax]))
        self.bounds = np.array(bounds)
        bounds = []
        for n in self.coords_orig:
            xmin = n[:,0].min()
            ymin = n[:,1].min()
            xmax = n[:,0].max()
            ymax = n[:,1].max()
            bounds.append(np.array([xmin,ymin,xmax,ymax]))
        self.bounds_orig = np.array(bounds)


def getWSI(filename):
    '''
        Returns image for desired level from given OpenSlide WSI format image filename

    '''
    slide = openslide.OpenSlide(filename)

    return slide

def getRegionFromSlide(slide, level=8, start_coord=(0,0), dims='full', from_level=8):
    if dims == 'full':
        img = np.array(slide.read_region((0,0), level, slide.level_dimensions[level]))
        img = img[:,:,:3]
    else:
        img = np.array(slide.read_region(start_coord, level, dims ))
        img = img[:,:,:3]

    return img

def getGTmask(img_filename, annotn_filename, level, coords, dims):
    slide = getWSI(img_filename)
    ann = Annotation(annotn_filename)
    c_shifted = ann.shift(origin=coords)
    c_scaled = ann.scale(c_shifted, slide.level_downsamples[level])

    mask = cv2.fillPoly(np.zeros((dims[0],dims[1],1)), c_scaled, (1))

    return mask

def getLabel(filename, level, coords, dims):
    '''
    Check if the annotation file with same name (extension .xml) exists: if not, return all zero mask of shape (dims,1)
    else, get the annotation file, shift its coordinates by coords and scale using level in slide downsample,
    followed by polyFill operation on a all zero mask of dimension (dims,1) with 1 and return it
    '''
    annotn_filename, _ = os.path.splitext(filename)
    annotn_filename = annotn_filename + '.xml'

    if os.path.exists(annotn_filename):
        return getGTmask(filename, annotn_filename, level, coords, dims)
    else:
        #print('{} does not exist'.format(annotn_filename))
        return np.zeros((dims[0],dims[1],1))


'''
Test code:
mask = getLabel( 'patient_015/patient_015_node_2.tif', 2,  [ 67700, 101508], (512,512))
print('Mask sum: {} ; shape: {}'.format(np.sum(mask), mask.shape))
plt.imshow(np.reshape(mask, (512,512)))
'''
