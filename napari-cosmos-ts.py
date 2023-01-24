""" napari plugin for analysis of colocalization single-molecule spectroscopy (CoSMoS) time series (TS)
"""

import os
import math
import numpy as np
import pandas as pd
import tifffile
import scipy.io as sio
from scipy.spatial import distance
from skimage import filters, morphology, measure
import napari
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QTabWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QGroupBox, \
    QFileDialog, QTextEdit, QComboBox, QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox, QMessageBox, QDoubleSpinBox, \
    QInputDialog
try:
    from pystackreg import StackReg
except ImportError:
    StackReg = None

__author__ = "Marcel Goldschen-Ohm <goldschen-ohm@utexas.edu, marcel.goldschen@gmail.com>"
__version__ = '1.0.0'


class CoSMoS_TS_napari_UI(QTabWidget):
    """ napari dock widget for analysis of colocalization single-molecule spectroscopy (CoSMoS) time series (TS)

        Image layer.metadata
            ['image_file_abspath'] = "abs/path/to/image/"
            ['point_zprojections']
                [points-layer-name] = [NxT] zprojections for points-layer-name
        
        Points layer.features
            ['tags'] = series of string tags for each point
        
        layerData
            ['point_zprojection_plot'] = pyqtgraph plot object
            ['point_zprojection_plot_data'] = pyqtgraph data series plot object
            ['point_zprojection_plot_vline'] = pyqtgraph vertical line plot object
    
        TODO
        - find colocalized points based on criteria (nn-dist and #nn ?)
        - register point layers
        - support per frame point positions (point tracking)
        - z-projection tag filter custom and, or grouping
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Array of dicts for metadata belonging to each layer.
        # Stored separately to keep non-serealizable objects (e.g., QObjects) out of the layer metadata dicts.
        # This unfortunately means we will have to manage this list upon any change to the layer order.
        # Other layer data that can be serialized and should be saved can be stored in the layer metadata dicts.
        self.layerMetadata = []

        # For copying and pasting layer transforms.
        self.layerTransformCopy = np.eye(3)

        # Keep track of selected point index
        # so we know if we are incrementing or decrementing the index when filtering tags.
        self.pointIndex = None

        # Default point mask to use for non-point locations.
        self.defaultPointMask = self.getPointMask([6, 6])

        # setup UI
        self.initUI()

        # event handling
        self.viewer.layers.events.inserted.connect(self.onLayerInserted)
        self.viewer.layers.events.removed.connect(self.onLayerRemoved)
        self.viewer.layers.events.moved.connect(self.onLayerMoved)
        self.viewer.dims.events.current_step.connect(self.onDimStepChanged)
        self.viewer.mouse_drag_callbacks.append(self.onMouseClickedOrDragged)
        self.viewer.mouse_double_click_callbacks.append(self.onMouseDoubleClicked)

        # # testing
        # layer = self.openTIFF('test.tif')
        # print(layer.source)
        # print(layer.source.path)
        # print(layer.metadata['image_file_abspath'])

        # theta = 0.4
        # rot = np.array([
        #     [np.cos(theta), -np.sin(theta), 0],
        #     [np.sin(theta), np.cos(theta), 0],
        #     [0, 0, 1]
        # ])

        # layer = self.openTIFF('tmp/test.tif')
        # layer.name = 'eGFP'
        # layer.colormap = 'green'

        # layer = self.openTIFF('tmp/test.tif')
        # layer.name = 'fcGMP'
        # layer.colormap = 'magenta'
        # layer.affine = rot

        # layer = self.openTIFF('spots.tif', memorymap=False)
        # layer.name = 'eGFP spots'
        # layer.colormap = 'green'

        # layer = self.openTIFF('spots.tif', memorymap=False)
        # layer.name = 'fcGMP spots'
        # layer.colormap = 'magenta'
        # layer.affine = rot

        # spots = np.random.uniform(0, 255, (15, 2))
        # self.viewer.add_points(spots, name='spots', size=9, edge_width=1, edge_width_is_relative=False, 
        #     edge_color='yellow', face_color=[0]*4, blending='translucent_no_depth', opacity=0.5)
    
    def printLayerMetadataStructure(self):
        for layer in self.viewer.layers:
            print(layer.name)
            printDictStructure(layer.metadata, start="\t")
    
    def initUI(self):
        self.addMetadataTab()
        self.addFileIOTab()
        self.addImageProcessingTab()
        self.addLayerRegistrationTab()
        self.addPointsTab()
        self.addPointZProjectionsTab()
    
    def addMetadataTab(self, title="Meta"):
        self.dateEdit = QLineEdit()
        self.idEdit = QLineEdit()
        self.nameEdit = QLineEdit()
        self.notesEdit = QTextEdit()

        tab = QWidget()
        form = QFormLayout(tab)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setSpacing(5)
        form.addRow("Experiment Date", self.dateEdit)
        form.addRow("Experiment ID", self.idEdit)
        form.addRow("Experimenter", self.nameEdit)
        form.addRow("Notes", self.notesEdit)
        self.addTab(tab, title)
    
    def addFileIOTab(self, title="I/O"):
        self.openMemoryMappedTiffButton = QPushButton("Open memory-mapped TIFF image file")
        self.openMemoryMappedTiffButton.clicked.connect(lambda x: self.openTIFF())

        self.openSessionButton = QPushButton("Open .mat session file")
        self.openSessionButton.clicked.connect(lambda x: self.importSession())

        self.saveSessionButton = QPushButton("Save session as .mat file")
        self.saveSessionButton.clicked.connect(lambda x: self.exportSession())

        tab = QWidget()
        tab.setMaximumWidth(500)
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(5)
        vbox.addWidget(self.openMemoryMappedTiffButton)
        vbox.addWidget(self.openSessionButton)
        vbox.addWidget(self.saveSessionButton)
        vbox.addStretch()
        self.addTab(tab, title)
    
    def addImageProcessingTab(self, title="Process"):
        self.zprojectImageButton = QPushButton("Z-Project Image")
        self.zprojectImageButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.zprojectImageLayer))

        self.zprojectOperation = QComboBox()
        self.zprojectOperation.addItem("max")
        self.zprojectOperation.addItem("min")
        self.zprojectOperation.addItem("std")
        self.zprojectOperation.addItem("sum")
        self.zprojectOperation.addItem("mean")
        self.zprojectOperation.addItem("median")
        self.zprojectOperation.setCurrentText("mean")

        self.zprojectImageFrames = QLineEdit()

        self.gaussianFilterButton = QPushButton("Gaussian Filter")
        self.gaussianFilterButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.gaussianFilterImageLayer))

        self.gaussianSigmaSpinBox = QDoubleSpinBox()
        self.gaussianSigmaSpinBox.setValue(1)

        self.tophatFilterButton = QPushButton("Tophat Filter")
        self.tophatFilterButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.tophatFilterImageLayer))

        self.tophatDiskRadiusSpinBox = QDoubleSpinBox()
        self.tophatDiskRadiusSpinBox.setValue(3)

        tab = QWidget()
        tab.setMaximumWidth(500)
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(5)

        text = QLabel("Applied to all selected image layers.\nResults are returned in new layers.")
        text.setWordWrap(True)
        vbox.addWidget(text)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.zprojectImageButton)
        form.addRow("Project", self.zprojectOperation)
        form.addRow("Z-start:stop:step", self.zprojectImageFrames)
        grid.addWidget(group, 0, 0)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.gaussianFilterButton)
        form.addRow("Sigma", self.gaussianSigmaSpinBox)
        grid.addWidget(group, 1, 0)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.tophatFilterButton)
        form.addRow("Disk Radius", self.tophatDiskRadiusSpinBox)
        grid.addWidget(group, 2, 0)

        vbox.addLayout(grid)
        vbox.addStretch()
        self.addTab(tab, title)
    
    def addLayerRegistrationTab(self, title="Align"):
        self.fixedLayerSelector = QComboBox()
        self.movingLayerSelector = QComboBox()

        self.layerTransformSelector = QComboBox()
        self.layerTransformSelector.addItem("Translation")
        self.layerTransformSelector.addItem("Rigid Body")
        self.layerTransformSelector.addItem("Affine")
        self.layerTransformSelector.setCurrentText("Affine")

        self.registerLayersButton = QPushButton("Register Layers")
        self.registerLayersButton.clicked.connect(lambda x: self.registerLayers())

        self.copyLayerTransformButton = QPushButton("Copy selected layer transform")
        self.copyLayerTransformButton.clicked.connect(lambda x: self.copyLayerTransform())

        self.pasteLayerTransformButton = QPushButton("Paste copied transform to all selected layers")
        self.pasteLayerTransformButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.pasteCopiedLayerTransform))

        self.clearLayerTransformButton = QPushButton("Clear transform from all selected layers")
        self.clearLayerTransformButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.clearLayerTransform))

        tab = QWidget()
        tab.setMaximumWidth(500)
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(5)
        text = QLabel("Registration sets the layer affine transform without altering the layer data.")
        text.setWordWrap(True)
        vbox.addWidget(text)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setSpacing(5)
        form.addRow(self.registerLayersButton)
        form.addRow("Fixed Layer", self.fixedLayerSelector)
        form.addRow("Moving Layer", self.movingLayerSelector)
        form.addRow("Transform", self.layerTransformSelector)
        vbox.addWidget(group)

        vbox.addWidget(self.copyLayerTransformButton)
        vbox.addWidget(self.pasteLayerTransformButton)
        vbox.addWidget(self.clearLayerTransformButton)
        vbox.addStretch()
        self.addTab(tab, title)
    
    def addPointsTab(self, title="Points"):
        self.findPointsButton = QPushButton("Find peaks in all selected image layers")
        self.findPointsButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.findPeaksInImageLayer))

        self.minPeakHeightSpinBox = QDoubleSpinBox()
        self.minPeakHeightSpinBox.setMaximum(65000)
        self.minPeakHeightSpinBox.setValue(10)

        self.minPeakSeparationSpinBox = QDoubleSpinBox()
        self.minPeakSeparationSpinBox.setValue(6)

        self.pointsSizeButton = QPushButton("Set point size for all selected points layers")
        self.pointsSizeButton.clicked.connect(lambda x: self.setSelectedPointsLayersPointSize())

        self.pointsEdgeWidthButton = QPushButton("Set edge width for all selected points layers")
        self.pointsEdgeWidthButton.clicked.connect(lambda x: self.setSelectedPointsLayersEdgeWidth())

        self.zprojectAllPointsButton = QPushButton("Compute point z-projections for all selected points layers")
        self.zprojectAllPointsButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.zprojectPointsLayer))

        self.findColocalizedPointsButton = QPushButton("Find colocalized points")
        self.findColocalizedPointsButton.clicked.connect(self.findColocalizedPoints)

        self.pointsColocalizationLayerSelectionBox = QComboBox()
        self.pointsColocalizationLayerSelectionBox.currentTextChanged.connect(self.updatePointsColocalizationPlot)
        self.neighborsColocalizationLayerSelectionBox = QComboBox()
        self.neighborsColocalizationLayerSelectionBox.currentTextChanged.connect(self.updatePointsColocalizationPlot)

        self.nearestNeighborDistanceCutoffSpinBox = QDoubleSpinBox()
        self.nearestNeighborDistanceCutoffSpinBox.setValue(3)

        self.colocalizationPlot = self.newPlot()
        self.colocalizationPlot.setLabels(left="Counts", bottom="Nearest Neighbor Distance")
        legend = pg.LegendItem()
        legend.setParentItem(self.colocalizationPlot.getPlotItem())
        legend.anchor((1,0), (1,0))
        # self.colocalizationPlot.addLegend()
        self.withinLayersNearestNeighborsHistogram = pg.PlotCurveItem([0, 0], [0], stepMode='center', pen=pg.mkPen([98, 143, 176, 80], width=1), fillLevel=0, brush=(98, 143, 176, 80), name="within layers")
        self.betweenLayersNearestNeighborsHistogram = pg.PlotCurveItem([0, 0], [0], stepMode='center', pen=pg.mkPen([255, 0, 0, 80], width=1), fillLevel=0, brush=(255, 0, 0, 80), name="between layers")
        self.colocalizationPlot.addItem(self.withinLayersNearestNeighborsHistogram)
        self.colocalizationPlot.addItem(self.betweenLayersNearestNeighborsHistogram)
        legend.addItem(self.withinLayersNearestNeighborsHistogram, "within layers")
        legend.addItem(self.betweenLayersNearestNeighborsHistogram, "between layers")

        tab = QWidget()
        tab.setMaximumWidth(500)
        vbox = QVBoxLayout(tab)
        vbox.setSpacing(5)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setSpacing(5)
        form.addRow(self.findPointsButton)
        form.addRow("Min Peak Height", self.minPeakHeightSpinBox)
        form.addRow("Min Separation", self.minPeakSeparationSpinBox)
        vbox.addWidget(group)

        vbox.addWidget(self.pointsSizeButton)
        vbox.addWidget(self.pointsEdgeWidthButton)
        vbox.addWidget(self.zprojectAllPointsButton)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setSpacing(5)
        form.addRow(self.findColocalizedPointsButton)
        form.addRow("Points Layer", self.pointsColocalizationLayerSelectionBox)
        form.addRow("Neighbors Layer", self.neighborsColocalizationLayerSelectionBox)
        form.addRow("Nearest Neighbor Distance Cutoff", self.nearestNeighborDistanceCutoffSpinBox)
        form.addRow(self.colocalizationPlot)
        vbox.addWidget(group)

        vbox.addStretch()
        self.addTab(tab, title)
    
    def addPointZProjectionsTab(self, title="Point Z-Projections"):
        self.pointZProjPlotLayout = QVBoxLayout()
        self.pointZProjPlotLayout.setSpacing(0)

        self.pointsZProjLayerSelectionBox = QComboBox()
        self.pointsZProjLayerSelectionBox.currentTextChanged.connect(lambda x: self.setActivePointsLayer())

        self.pointIndexSpinBox = QSpinBox()
        self.pointIndexSpinBox.setMaximum(65000)
        self.pointIndexSpinBox.setKeyboardTracking(False)
        self.pointIndexSpinBox.valueChanged.connect(lambda x: self.setSelectedPointIndex())

        self.pointZProjectionsWorldPositionText = QLabel()
        self.numPointsText = QLabel()

        self.pointTagsEdit = QLineEdit()
        self.pointTagsEdit.editingFinished.connect(self.updateSelectedPointTags)

        self.pointTagFilterEdit = QLineEdit()
        self.pointTagFilterCBox = QCheckBox("Filter")

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(QLabel("Points Layer"), 0, 0, Qt.AlignRight)
        grid.addWidget(self.pointsZProjLayerSelectionBox, 0, 1)
        grid.addWidget(self.numPointsText, 0, 2)
        grid.addWidget(self.pointTagFilterCBox, 0, 3)
        grid.addWidget(self.pointTagFilterEdit, 0, 4)
        grid.addWidget(QLabel("Point Index"), 1, 0, Qt.AlignRight)
        grid.addWidget(self.pointIndexSpinBox, 1, 1)
        grid.addWidget(self.pointZProjectionsWorldPositionText, 1, 2)
        grid.addWidget(QLabel("Tags"), 1, 3, Qt.AlignRight)
        grid.addWidget(self.pointTagsEdit, 1, 4)

        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.addLayout(grid)
        vbox.addLayout(self.pointZProjPlotLayout)
        self.addTab(tab, title)
    
    def exportSession(self, filename=None, writeImageStackData=False):
        """ Export data to MATLAB .mat file.

            Optionally write image stack data to file. Otherwise only save their filenames (default).
        """
        if filename is None:
            filename, _filter = QFileDialog.getSaveFileName(self, "Save data in MATLAB file format.", "", "MATLAB files (*.mat)")
            if filename == "":
                return
        sesseionAbsPath = os.path.abspath(filename)
        sessionAbsDir, sessionFile = os.path.split(sesseionAbsPath)
        mdict = {}
        mdict['date'] = self.dateEdit.text() + " "
        mdict['ID'] = self.idEdit.text() + " "
        mdict['experimenter'] = self.nameEdit.text() + " "
        mdict['notes'] = self.notesEdit.toPlainText() + " "
        mdict['layers'] = {}
        for layer in self.viewer.layers:
            layerName = layer.name.replace(' ', '_')
            layerDict = {}
            layerDict['metadata'] = {}
            if self.isImageLayer(layer):
                imageAbsPath = None
                if layer.source.path is not None:
                    imageAbsPath = os.path.abspath(layer.source.path)
                elif 'image_file_abspath' in layer.metadata:
                    imageAbsPath = os.path.abspath(layer.metadata['image_file_abspath'])
                if imageAbsPath is not None:
                    imageRelPath = os.path.relpath(imageAbsPath, start=sessionAbsDir)
                    layerDict['metadata']['image_file_abspath'] = imageAbsPath
                    layerDict['metadata']['image_file_relpath'] = imageRelPath
                if writeImageStackData or (imageAbsPath is None) or not self.isImageStackLayer(layer):
                    if type(layer.data) is np.ndarray:
                        layerDict['image'] = layer.data
                    else:
                        try:
                            layerDict['image'] = np.array(layer.data)
                        except:
                            pass
                if 'image' not in layerDict:
                    layerDict['metadata']['image_shape'] = layer.data.shape
                    layerDict['metadata']['image_dtype'] = str(layer.data.dtype)
            elif self.isPointsLayer(layer):
                layerDict['points'] = layer.data
                if not layer.features.empty:
                    layerDict['features'] = {}
                    for key in layer.features:
                        if key == 'tags':
                            layer.features['tags'].fillna("", inplace=True)
                        if type(layer.features[key][0]) == str:
                            layerDict['features'][key] = [astr + " " for astr in layer.features[key]]
                        else:
                            layerDict['features'][key] = layer.features[key]
            layerDict['affine'] = layer.affine.affine_matrix[-3:,-3:]
            layerDict['opacity'] = layer.opacity
            layerDict['blending'] = layer.blending
            if self.isImageLayer(layer):
                layerDict['contrast_limits'] = layer.contrast_limits
                layerDict['gamma'] = layer.gamma
                layerDict['colormap'] = layer.colormap.name
                layerDict['interpolation2d'] = layer.interpolation2d
            if self.isPointsLayer(layer):
                layerDict['size'] = layer.size
                layerDict['symbol'] = layer.symbol
                layerDict['face_color'] = layer.face_color
                layerDict['edge_color'] = layer.edge_color
                layerDict['edge_width'] = layer.edge_width
                layerDict['edge_width_is_relative'] = layer.edge_width_is_relative
            for key in layer.metadata:
                if key in ["image_file_abspath", "image_file_relpath"]:
                    continue
                layerDict['metadata'][key] = layer.metadata[key]
            if len(layerDict['metadata']) == 0:
                del layerDict['metadata']
            mdict['layers'][layerName] = layerDict
        sio.savemat(filename, mdict)

    def importSession(self, filename=None):
        """ Import data from MATLAB .mat file.
        """
        if filename is None:
            filename, _filter = QFileDialog.getOpenFileName(self, "Open MATLAB format data file.", "", "MATLAB files (*.mat)")
            if filename == "":
                return
        sesseionAbsPath = os.path.abspath(filename)
        sessionAbsDir, sessionFile = os.path.split(sesseionAbsPath)
        mdict = sio.loadmat(filename, simplify_cells=True)
        for key, value in mdict.items():
            if key == "date":
                self.dateEdit.setText(str(mdict['date']).strip())
            elif key == "id":
                self.idEdit.setText(str(mdict['ID']).strip())
            elif key == "experimenter":
                self.nameEdit.setText(str(mdict['experimenter']).strip())
            elif key == "notes":
                self.notesEdit.setPlainText(str(mdict['notes']).strip())
            elif key == "layers":
                for layerName, layerDict in value.items():
                    layerName = layerName.replace('_', ' ')
                    hasMetadata = 'metadata' in layerDict
                    affine = layerDict['affine']
                    opacity = layerDict['opacity']
                    blending = str(layerDict['blending'])
                    isImageLayer = ('image' in layerDict) or (hasMetadata and ( \
                        ('image_file_relpath' in layerDict['metadata']) \
                        or ('image_file_abspath' in layerDict['metadata']) \
                        or ('image_shape' in layerDict['metadata'])))
                    if 'points' in layerDict:
                        # points layer
                        points = layerDict['points']
                        features = pd.DataFrame()
                        if 'features' in layerDict:
                            for key, value in layerDict['features'].items():
                                if type(value[0]) == str:
                                    features[key] = [astr.strip() for astr in value]
                                else:
                                    features[key] = value
                        size = layerDict['size']
                        symbol = str(layerDict['symbol'])
                        face_color = layerDict['face_color']
                        edge_color = layerDict['edge_color']
                        edge_width = layerDict['edge_width']
                        edge_width_is_relative = layerDict['edge_width_is_relative']
                        self.viewer.add_points(points, name=layerName, features=features, affine=affine, opacity=opacity, blending=blending, 
                            size=size, symbol=symbol, face_color=face_color, edge_color=edge_color, edge_width=edge_width, edge_width_is_relative=edge_width_is_relative)
                    elif isImageLayer:
                        # image layer
                        imageAbsPath = None
                        if hasMetadata and ('image_file_relpath' in layerDict['metadata']):
                            imageRelPath = layerDict['metadata']['image_file_relpath']
                            imageAbsPath = os.path.join(sessionAbsDir, imageRelPath)
                        elif hasMetadata and ('image_file_abspath' in layerDict['metadata']):
                            imageAbsPath = os.path.abspath(layerDict['metadata']['image_file_abspath'])
                        contrast_limits = layerDict['contrast_limits']
                        gamma = layerDict['gamma']
                        colormap = str(layerDict['colormap'])
                        interpolation2d = str(layerDict['interpolation2d'])
                        if 'image' in layerDict:
                            image = layerDict['image']
                            layer = self.viewer.add_image(image, name=layerName, affine=affine, opacity=opacity, blending=blending, 
                                contrast_limits=contrast_limits, gamma=gamma, colormap=colormap, interpolation2d=interpolation2d)
                            if imageAbsPath is not None:
                                layer.metadata['image_file_abspath'] = imageAbsPath
                        elif imageAbsPath is not None:
                            try:
                                image = tifffile.memmap(imageAbsPath)
                                layer = self.viewer.add_image(image, name=layerName, affine=affine, opacity=opacity, blending=blending, 
                                    contrast_limits=contrast_limits, gamma=gamma, colormap=colormap, interpolation2d=interpolation2d)
                                layer.metadata['image_file_abspath'] = imageAbsPath
                            except:
                                try:
                                    layer = self.viewer.open(path=imageAbsPath, layer_type="image")[0]
                                    layer.metadata['image_file_abspath'] = os.path.abspath(layer.source.path)
                                except:
                                    msg = QMessageBox(self)
                                    msg.setIcon(QMessageBox.Warning)
                                    msg.setText(f"Failed to load image {imageAbsPath}")
                                    msg.setStandardButtons(QMessageBox.Close)
                                    msg.exec_()
                    if 'metadata' in layerDict:
                        layer = self.viewer.layers[-1]
                        for key in layerDict['metadata']:
                            if key in ["image_file_abspath", "image_file_relpath"]:
                                continue
                            layer.metadata[key] = layerDict['metadata'][key]

    def openTIFF(self, filename=None, memorymap=True):
        """ Open TIFF image file.

            Uses tifffile.
            Image data can either be memory mapped (default) or loaded into RAM.
        """
        if filename is None:
            filename, _filter = QFileDialog.getOpenFileName(self, "Open TIFF image file.", "", "TIFF (*.tif *.tiff)")
        if filename == "" or not os.path.exists(filename):
            return
        filenameNoExt = os.path.splitext(filename)[0]
        path, fileNoExt = os.path.split(filenameNoExt)
        try:
            if memorymap:
                data = tifffile.memmap(filename)
            else:
                data = tifffile.imread(filename)
            layer = self.viewer.add_image(data, name=fileNoExt, blending='additive')
            layer.metadata['image_file_abspath'] = os.path.abspath(filename)
        except:
            layer = self.viewer.open(path=filename, layer_type="image")[0]
            layer.name = fileNoExt
            layer.blending = 'additive'
            layer.metadata['image_file_abspath'] = os.path.abspath(layer.source.path)
        return layer
    
    def imageLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isImageLayer(layer)]
    
    def imageStackLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isImageStackLayer(layer)]
    
    def pointsLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isPointsLayer(layer)]
    
    def isImageLayer(self, layer) -> bool:
        return type(layer) is napari.layers.image.image.Image
    
    def isImageStackLayer(self, layer) -> bool:
        return type(layer) is napari.layers.image.image.Image and layer.ndim == 3
    
    def isPointsLayer(self, layer) -> bool:
        return type(layer) is napari.layers.points.points.Points
    
    def pointZProjectionPlots(self):
        return [data['point_zprojection_plot'] for data in reversed(self.layerMetadata) if 'point_zprojection_plot' in data]
    
    def linkPointZProjectionPlots(self):
        plots = self.pointZProjectionPlots()
        for i in range(1, len(plots)):
            plots[i].setXLink(plots[0])
    
    def clearPointZProjectionPlots(self):
        for layerMetadata in self.layerMetadata:
            if 'point_zprojection_plot_data' in layerMetadata:
                layerMetadata['point_zprojection_plot_data'].setData([])
    
    def newPlot(self) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.getAxis('left').setWidth(82)
        plot.getAxis('right').setWidth(10)
        plot.showGrid(x=True, y=True, alpha=0.3)
        # plot.setBackground([38, 41, 48])
        # hack to stop grid from clipping axis tick labels
        for key in ['left', 'bottom']:
            plot.getAxis(key).setGrid(False)
        for key in ['right', 'top']:
            plot.getAxis(key).setStyle(showValues=False)
            plot.showAxis(key)
        return plot
    
    def currentFrameIndex(self) -> int:
        try:
            return viewer.dims.current_step[-3]
        except IndexError:
            return 0
    
    def onLayerInserted(self, event):
        layer = event.value
        index = event.index
        # insert separately managed dict of layer metadata to reflect new layer
        self.layerMetadata.insert(index, {})
        layerMetadata = self.layerMetadata[index]
        if self.isImageStackLayer(layer):
            # create plot to show point z-projections for the new layer
            plot = self.newPlot()
            plot.setLabels(left=layer.name)
            plot_data = plot.plot([], pen=pg.mkPen([98, 143, 176], width=1))
            prevPlots = self.pointZProjectionPlots()
            if len(prevPlots):
                plot.setXLink(prevPlots[0])
            t = self.currentFrameIndex()
            plot_vline = plot.addLine(x=t, pen=pg.mkPen('y', width=1))
            # store non-serializable Qt plot objects in separate layer metadata
            layerMetadata['point_zprojection_plot'] = plot
            layerMetadata['point_zprojection_plot_data'] = plot_data
            layerMetadata['point_zprojection_plot_vline'] = plot_vline
            # insert plot into layout
            plotIndex = self.imageStackLayers().index(layer)
            self.pointZProjPlotLayout.insertWidget(plotIndex, plot)
            self.pointZProjPlotLayout.setStretch(plotIndex, 1)
        elif self.isPointsLayer(layer):
            n_points = len(layer.data)
            # tags string feature for each point
            if not 'tags' in layer.features:
                layer.features['tags'] = [""] * n_points
        self.updateLayerSelectionBoxes()
        # handle general events for new layer
        layer.events.name.connect(self.onLayerNameChanged)
        layer.events.visible.connect(self.onLayerVisibilityChanged)
    
    def onLayerRemoved(self, event):
        layer = event.value
        index = event.index
        layerMetadata = self.layerMetadata.pop(index)
        if 'point_zprojection_plot' in layerMetadata:
            # delete plot
            plot = layerMetadata['point_zprojection_plot']
            self.pointZProjPlotLayout.removeWidget(plot)
            plot.deleteLater()
            # reset plot linkage
            self.linkPointZProjectionPlots()
        del layerMetadata
        if self.isPointsLayer(layer):
            # remove point z-projections for this layer from all image stack layers
            for imlayer in self.imageStackLayers():
                if 'point_zprojections' in imlayer.metadata:
                    if layer.name in imlayer.metadata['point_zprojections']:
                        del imlayer.metadata['point_zprojections'][layer.name]
        self.updateLayerSelectionBoxes()
    
    def onLayerMoved(self, event):
        index = event.index
        new_index = event.new_index
        layerMetadata = self.layerMetadata.pop(index)
        self.layerMetadata.insert(new_index, layerMetadata)
        if 'point_zprojection_plot' in layerMetadata:
            # reposition plot to match layer order
            plot = layerMetadata['point_zprojection_plot']
            self.pointZProjPlotLayout.removeWidget(plot)
            plotIndex = self.pointZProjectionPlots().index(plot)
            self.pointZProjPlotLayout.insertWidget(plotIndex, plot)
            self.pointZProjPlotLayout.setStretch(plotIndex, 1)
        self.updateLayerSelectionBoxes()
    
    def onLayerNameChanged(self, event):
        index = event.index
        layer = self.viewer.layers[index]
        layerMetadata = self.layerMetadata[index]
        if 'point_zprojection_plot' in layerMetadata:
            # plot ylabel = layer name
            plot = layerMetadata['point_zprojection_plot']
            plot.setLabels(left=layer.name)
        if self.isPointsLayer(layer):
            # update point z-projection dicts for all image stack layers to reflect new layer name
            for imlayer in self.imageStackLayers():
                if 'point_zprojections' in imlayer.metadata:
                    # event does not tell us what the previous layer name was, so we have to figure it out
                    zproj_layer_names = imlayer.metadata['point_zprojections'].keys()
                    points_layer_names = [layer.name for layer in self.pointsLayers()]
                    for name in zproj_layer_names:
                        if name not in points_layer_names:
                            imlayer.metadata['point_zprojections'][layer.name] = imlayer.metadata['point_zprojections'].pop(name)
                            break
        self.updateLayerSelectionBoxes()
    
    def onLayerVisibilityChanged(self, event):
        index = event.index
        layer = self.viewer.layers[index]
        layerMetadata = self.layerMetadata[index]
        if 'point_zprojection_plot' in layerMetadata:
            # show/hide plot along with layer
            plot = layerMetadata['point_zprojection_plot']
            plot.setVisible(layer.visible)
    
    def onDimStepChanged(self, event):
        try:
            t = event.value[-3]
        except IndexError:
            t = 0
        for layerMetadata in self.layerMetadata:
            # update frame vline in point z-projection plots
            if 'point_zprojection_plot_vline' in layerMetadata:
                layerMetadata['point_zprojection_plot_vline'].setValue(t)
    
    def onMouseClickedOrDragged(self, viewer, event):
        if event.type == 'mouse_press':
            worldPoint = event.position[-2:]  # (row, col)
            # check if clicked on a visible point
            visiblePointsLayers = [layer for layer in self.pointsLayers() if layer.visible]
            if len(visiblePointsLayers):
                activePointsLayer = self.activePointsLayer()
                if activePointsLayer in visiblePointsLayers:
                    # check active points layer first
                    visiblePointsLayers.remove(activePointsLayer)
                    visiblePointsLayers.insert(0, activePointsLayer)
                for layer in visiblePointsLayers:
                    if layer.data.size == 0:
                        continue
                    worldPoints = self.transformPointsFromLayerToWorld(layer.data, layer)
                    squareDists = np.sum((worldPoints - worldPoint)**2, axis=1)
                    index = np.argmin(squareDists)
                    radius = layer.size[index].mean() / 2
                    if squareDists[index] <= radius**2:
                        self.setSelectedPointIndex(index, layer)
                        return
            # no point selected
            self.pointIndexSpinBox.clear()
            self.pointTagsEdit.setText("")
            for layer in self.pointsLayers():
                layer._selected_data = set()
                layer._highlight_index = []
                layer.events.highlight()
            # z-project clicked location
            self.zprojectWorldPoint(worldPoint, pointMask=self.defaultPointMask)
    
    def onMouseDoubleClicked(self, viewer, event):
        self.viewer.reset_view()
    
    def updateLayerSelectionBoxes(self):
        # layer registration
        fixedLayerName = self.fixedLayerSelector.currentText()
        movingLayerName = self.movingLayerSelector.currentText()
        self.fixedLayerSelector.clear()
        self.movingLayerSelector.clear()
        for layer in reversed(self.viewer.layers):
            self.fixedLayerSelector.addItem(layer.name)
            self.movingLayerSelector.addItem(layer.name)
        self.fixedLayerSelector.setCurrentText(fixedLayerName)
        self.movingLayerSelector.setCurrentText(movingLayerName)
        # layer point colocalization
        self.pointsColocalizationLayerSelectionBox.currentTextChanged.disconnect(self.updatePointsColocalizationPlot)
        self.neighborsColocalizationLayerSelectionBox.currentTextChanged.disconnect(self.updatePointsColocalizationPlot)
        pointsLayerName = self.pointsColocalizationLayerSelectionBox.currentText()
        neighborsLayerName = self.neighborsColocalizationLayerSelectionBox.currentText()
        self.pointsColocalizationLayerSelectionBox.clear()
        self.neighborsColocalizationLayerSelectionBox.clear()
        for layer in self.pointsLayers():
            self.pointsColocalizationLayerSelectionBox.addItem(layer.name)
            self.neighborsColocalizationLayerSelectionBox.addItem(layer.name)
        self.pointsColocalizationLayerSelectionBox.setCurrentText(pointsLayerName)
        self.neighborsColocalizationLayerSelectionBox.setCurrentText(neighborsLayerName)
        self.pointsColocalizationLayerSelectionBox.currentTextChanged.connect(self.updatePointsColocalizationPlot)
        self.neighborsColocalizationLayerSelectionBox.currentTextChanged.connect(self.updatePointsColocalizationPlot)
        self.updatePointsColocalizationPlot()
        # layer point z-projections
        self.pointsZProjLayerSelectionBox.currentTextChanged.disconnect()
        pointsLayerName = self.pointsZProjLayerSelectionBox.currentText()
        self.pointsZProjLayerSelectionBox.clear()
        for layer in self.pointsLayers():
            self.pointsZProjLayerSelectionBox.addItem(layer.name)
        self.pointsZProjLayerSelectionBox.setCurrentText(pointsLayerName)
        self.pointsZProjLayerSelectionBox.currentTextChanged.connect(lambda x: self.setActivePointsLayer())
        self.setActivePointsLayer()
    
    def registerLayers(self, fixedLayer=None, movingLayer=None, transformType=None):
        if fixedLayer is None:
            fixedLayerName = self.fixedLayerSelector.currentText()
            fixedLayer = self.viewer.layers[fixedLayerName]
        if movingLayer is None:
            movingLayerName = self.movingLayerSelector.currentText()
            movingLayer = self.viewer.layers[movingLayerName]
        if transformType is None:
            transformType = self.layerTransformSelector.currentText()
        if fixedLayer is movingLayer:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Registering a layer with itself is meaningless.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        if self.isImageLayer(fixedLayer) and self.isImageLayer(movingLayer):
            self.registerImageLayers(fixedLayer, movingLayer, transformType)
        elif self.isPointsLayer(fixedLayer) and self.isPointsLayer(movingLayer):
            self.registerPointLayers(fixedLayer, movingLayer, transformType)
        else:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Registering image with points layers not implemented.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
    
    def registerImageLayers(self, fixedLayer, movingLayer, transformType):
        if StackReg is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Image registration requires pystackreg.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        if not self.isImageLayer(fixedLayer) or not self.isImageLayer(movingLayer):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Layers must be image layers.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        fixedImage = fixedLayer.data
        movingImage = movingLayer.data
        # get current frame if layer is an image stack
        if fixedImage.ndim == 3:
            t = self.currentFrameIndex()
            fixedImage = fixedImage[t]
        if movingImage.ndim == 3:
            t = self.currentFrameIndex()
            movingImage = movingImage[t]
        # adjust image to match layer contrast limits
        fixedImage = normalizeImage(fixedImage, fixedLayer.contrast_limits)
        movingImage = normalizeImage(movingImage, movingLayer.contrast_limits)
        # register images
        tform = registerImages(fixedImage, movingImage, transformType=transformType)
        # apply net world transform to moving image
        movingLayer.affine = tform @ fixedLayer.affine.affine_matrix[-3:,-3:]
    
    def registerPointLayers(self, fixedLayer, movingLayer, transformType):
        if True:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Points layer registration not yet implemented.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        if not self.isPointsLayer(fixedLayer) or not self.isPointsLayer(movingLayer):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Layers must be points layers.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        # TODO
    
    def copyLayerTransform(self, layer=None):
        if layer is None:
            # first selected layer
            layers = list(self.viewer.layers.selection)
            if len(layers) == 0:
                return
            layer = layers[0]
        self.layerTransformCopy = self.layerToWorldTransform3x3(layer)
    
    def pasteCopiedLayerTransform(self, layer):
        layer.affine = self.layerTransformCopy
    
    def clearLayerTransform(self, layer):
        layer.affine = np.eye(3)
    
    def layerToWorldTransform3x3(self, layer):
        return layer.affine.inverse.affine_matrix[-3:,-3:]
    
    def worldToLayerTransform3x3(self, layer):
        return layer.affine.affine_matrix[-3:,-3:]
    
    def transformPointsFromLayerToWorld(self, points, layer):
        points = np.array(points)
        if points.ndim == 1:
            points = np.reshape(points, [1, -1])
        n_points = len(points)
        worldPoints = np.zeros((n_points, 2))
        for i, point in enumerate(points):
            if layer.ndim == 2:
                worldPoints[i] = layer.data_to_world(point)
            elif layer.ndim == 3:
                worldPoints[i] = layer.data_to_world((0, *point))[-2:]
        return worldPoints

    def transformPointsFromWorldToLayer(self, points, layer):
        points = np.array(points)
        if points.ndim == 1:
            points = np.reshape(points, [1, -1])
        n_points = len(points)
        layerPoints = np.zeros((n_points, 2))
        for i, point in enumerate(points):
            if layer.ndim == 2:
                layerPoints[i] = layer.world_to_data(point)
            elif layer.ndim == 3:
                layerPoints[i] = layer.world_to_data((0, *point))[-2:]
        return layerPoints
    
    def getLayerWorldBoundingBox(self, layer):
        if self.isImageLayer(layer):
            w, h = layer.data.shape[-2:]
            layerPoints = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        elif self.isPointsLayer(layer):
            layerPoints = layer.data
        else:
            return None
        worldPoints = self.transformPointsFromLayerToWorld(layerPoints, layer)
        worldRowLim = worldPoints[:,0].min(), worldPoints[:,0].max()
        worldColLim = worldPoints[:,1].min(), worldPoints[:,1].max()
        return worldRowLim, worldColLim
    
    def getLayerWorldGridBoxes(self):
        n_layers = len(self.viewer.layers)
        layerOrigins = np.zeros([n_layers, 2])
        layerRowLims = np.zeros([n_layers, 2])
        layerColLims = np.zeros([n_layers, 2])
        for i, layer in enumerate(self.viewer.layers):
            layerOrigins[i] = self.transformPointsFromLayerToWorld((0, 0), layer)
            rowLim, colLim = self.getLayerWorldBoundingBox(layer)
            layerRowLims[i] = rowLim
            layerColLims[i] = colLim
        rowOrigins = np.unique(layerOrigins[:,0])
        colOrigins = np.unique(layerOrigins[:,1])
        for rowOrigin in rowOrigins:
            rowMask = layerOrigins[:,0] == rowOrigin
            layerRowLims[rowMask,0] = layerRowLims[rowMask,0].min()
            layerRowLims[rowMask,1] = layerRowLims[rowMask,1].max()
        for colOrigin in colOrigins:
            colMask = layerOrigins[:,1] == colOrigin
            layerColLims[colMask,0] = layerColLims[colMask,0].min()
            layerColLims[colMask,1] = layerColLims[colMask,1].max()
        return layerRowLims, layerColLims
    
    def activePointsLayer(self):
        if len(self.pointsLayers()) == 0:
            return None
        if self.pointsZProjLayerSelectionBox.count() == 0:
            return None
        layerName = self.pointsZProjLayerSelectionBox.currentText()
        try:
            layer = self.viewer.layers[layerName]
        except KeyError:
            return None
        return layer
    
    def setActivePointsLayer(self, layer=None):
        # no highlighted points in any other layer
        for ptslayer in self.pointsLayers():
            ptslayer._selected_data = set()
            ptslayer._highlight_index = []
            ptslayer.events.highlight()
        if layer is None:
            layer = self.activePointsLayer()
        elif self.isPointsLayer(layer):
            self.pointsZProjLayerSelectionBox.setCurrentText(layer.name)
        else:
            layer = None
        if layer is not None and layer.data.size > 0:
            n_points = layer.data.shape[0]
        else:
            n_points = 0
        self.numPointsText.setText(f"{n_points}/{n_points} Points")
        if n_points > 0 and self.selectedPointIndex() is not None:
            self.setSelectedPointIndex()
    
    def selectedPointIndex(self):
        if self.activePointsLayer() is None:
            return None
        if self.pointIndexSpinBox.cleanText() == "":
            return None
        return self.pointIndexSpinBox.value()
    
    def setSelectedPointIndex(self, index=None, layer=None):
        if layer is None:
            layer = self.activePointsLayer()
        else:
            self.setActivePointsLayer(layer)
            layer = self.activePointsLayer()
        n_points = layer.data.shape[0]
        
        if index is None:
            index = self.selectedPointIndex()
        
        if index is None or index < 0 or layer is None or n_points == 0:
            # deselect all points
            self.pointIndexSpinBox.clear()
            self.pointZProjectionsWorldPositionText.setText("")
            self.pointTagsEdit.setText("")

            # no highlighted points
            if layer is not None:
                layer._selected_data = set()
                layer._highlight_index = []
                layer.events.highlight()

            # clear all z-projection plots
            self.clearPointZProjectionPlots()

            # done
            self.pointIndex = None
            return
        
        # tags filter?
        layer.features['tags'].fillna("", inplace=True)
        if self.pointTagFilterCBox.isChecked():
            index = min(max(0, index), n_points - 1)
            filter = self.pointTagFilterEdit.text()
            if (self.pointIndex is None) or (self.pointIndex <= index):
                # incrementing point index
                while index < n_points:
                    tags = layer.features['tags'][index]
                    if self.matchTagsToFilter(tags, filter):
                        break
                    index += 1
                if index == n_points:
                    # no matches found
                    if self.pointIndex is None:
                        self.pointIndexSpinBox.clear()
                    else:
                        self.pointIndexSpinBox.setValue(self.pointIndex)
                    return
            else:
                # decrementing point index
                while index >= 0:
                    tags = layer.features['tags'][index]
                    if self.matchTagsToFilter(tags, filter):
                        break
                    index -= 1
                if index < 0:
                    # no matches found
                    if self.pointIndex is None:
                        self.pointIndexSpinBox.clear()
                    else:
                        self.pointIndexSpinBox.setValue(self.pointIndex)
                    return

        # select point in layer
        index = min(max(0, index), n_points - 1)
        self.pointIndexSpinBox.setValue(index)
        self.pointIndex = index
        try:
            tags = layer.features['tags'][index]
            try:
                tags = tags.strip()
                self.pointTagsEdit.setText(tags)
            except AttributeError:
                layer.features['tags'][index] = ""
                self.pointTagsEdit.setText("")
        except (KeyError, IndexError):
            self.pointTagsEdit.setText("")
        
        # highlight selected point
        layer._selected_data = set([index])
        layer._highlight_index = [index]
        layer.events.highlight()
        
        # z-project image stacks at selected world point row, col)
        point = layer.data[index]
        worldPoint = layer.data_to_world(point)
        pointMask = self.getPointMask(layer.size[index])
        self.zprojectWorldPoint(worldPoint, pointMask=pointMask)
    
    def updateSelectedPointTags(self):
        layer = self.activePointsLayer()
        if layer is not None:
            index = self.selectedPointIndex()
            if index is not None:
                tags = self.pointTagsEdit.text().strip()
                if not 'tags' in layer.features:
                    n_points = len(layer.data)
                    layer.features['tags'] = [""] * n_points
                layer.features['tags'][index] = tags
    
    def matchTagsToFilter(self, tags, filter):
        tags = [tag.strip() for tag in tags.split(",")]
        or_tags = [tag.strip() for tag in filter.split(",")]
        for or_tag in or_tags:
            and_tags = [tag.strip() for tag in or_tag.split("&")]
            and_matches = [tag in tags for tag in and_tags]
            if np.all(and_matches):
                return True
        return False
    
    def getPointMask(self, pointSize):
        n_rows, n_cols = pointSize
        rows = np.reshape(np.arange(n_rows, dtype=float), [-1, 1])
        rows -= rows.mean()
        cols = np.reshape(np.arange(n_cols, dtype=float), [1, -1])
        cols -= cols.mean()
        pointMask = rows**2 / (n_rows / 2)**2  + cols**2 / (n_cols / 2)**2 < 1
        return pointMask
    
    def zprojectWorldPointInImageStackLayer(self, layer, worldPoint, pointMask=None) -> np.ndarray:
        if not self.isImageStackLayer(layer):
            return np.array([])
        # transform world point into image pixels (row, col)
        # extra point coord is 0th frame
        imagePoint = layer.world_to_data((0, *worldPoint))[-2:]
        # z-project point in image stack
        zproj = zprojectPointInImageStack(layer.data, imagePoint, pointMask=pointMask)
        return zproj
    
    def zprojectWorldPoint(self, worldPoint, pointMask=None):
        if self.viewer.grid.enabled:
            layerRowLims, layerColLims = self.getLayerWorldGridBoxes()
            row, col = worldPoint
            inLayerIndex = np.where((layerRowLims[:,0] <= row) & (layerRowLims[:,1] > row) \
                & (layerColLims[:,0] <= col) & (layerColLims[:,1] > col))[0][0]
            gridRowOffset = row - layerRowLims[inLayerIndex,0]
            gridColOffset = col - layerColLims[inLayerIndex,0]
        for layer in self.imageStackLayers():
            if self.viewer.grid.enabled:
                layerIndex = list(self.viewer.layers).index(layer)
                row = layerRowLims[layerIndex,0] + gridRowOffset
                col = layerColLims[layerIndex,0] + gridColOffset
                zproj = self.zprojectWorldPointInImageStackLayer(layer, (row, col), pointMask=pointMask)
            else:
                zproj = self.zprojectWorldPointInImageStackLayer(layer, worldPoint, pointMask=pointMask)
            # update z-projection plot
            layerIndex = list(self.viewer.layers).index(layer)
            self.layerMetadata[layerIndex]['point_zprojection_plot_data'].setData(zproj)
        row, col = worldPoint
        self.pointZProjectionsWorldPositionText.setText(f"[{int(row)}, {int(col)}]")
    
    def zprojectPointsLayer(self, layer, pointMask=None):
        if not self.isPointsLayer(layer):
            return
        n_points = layer.data.shape[0]
        # transform points to world coords
        worldPoints = self.transformPointsFromLayerToWorld(layer.data, layer)
        if self.viewer.grid.enabled:
            # remove grid shift
            layerRowLims, layerColLims = self.getLayerWorldGridBoxes()
            layerIndex = list(self.viewer.layers).index(layer)
            layerGridShift = np.array([layerRowLims[layerIndex,0], layerColLims[layerIndex,0]])
            gridPoints = worldPoints - layerGridShift
        # z-project points in all image stack layers
        for imlayer in self.imageStackLayers():
            n_frames = imlayer.data.shape[0]
            zprojs = np.zeros((n_points, n_frames))
            if self.viewer.grid.enabled:
                # add grid shift
                layerIndex = list(self.viewer.layers).index(layer)
                layerGridShift = np.array([layerRowLims[layerIndex,0], layerColLims[layerIndex,0]])
                shiftedPoints = gridPoints + layerGridShift
            for i in range(n_points):
                if pointMask is None:
                    # set point mask based on point size
                    thisPointMask = self.getPointMask(layer.size[i])
                else:
                    thisPointMask = pointMask
                try:
                    if self.viewer.grid.enabled:
                        zprojs[i] = self.zprojectWorldPointInImageStackLayer(imlayer, shiftedPoints[i], pointMask=thisPointMask)
                    else:
                        zprojs[i] = self.zprojectWorldPointInImageStackLayer(imlayer, worldPoints[i], pointMask=thisPointMask)
                except:
                    # in case point not on image
                    pass
            # store z-projections in image stack layer
            if 'point_zprojections' not in imlayer.metadata:
                imlayer.metadata['point_zprojections'] = {}
            imlayer.metadata['point_zprojections'][layer.name] = zprojs
    
    def zprojectImageLayer(self, layer, method=None, frames=None):
        if not self.isImageStackLayer(layer):
            return
        if method is None:
            method = self.zprojectOperation.currentText()
        methods = {"max": np.max, "min": np.min, "std": np.std, "sum": np.sum, "mean": np.mean, "median": np.median}
        func = methods[method]
        n_frames = layer.data.shape[0]
        if frames is None:
            framesText = self.zprojectImageFrames.text().strip()
            if framesText == "":
                frames = np.arange(n_frames)
            else:
                slice = framesText.split(':')
                try:
                    start = int(slice[0])
                except (IndexError, ValueError):
                    start = 0
                try:
                    stop = int(slice[1])
                except (IndexError, ValueError):
                    stop = n_frames
                try:
                    step = int(slice[2])
                except (IndexError, ValueError):
                    step = 1
                frames = np.arange(start, stop, step)
        projected = func(layer.data[frames], axis=0)
        name = layer.name + f" {method}-proj"
        tform = self.worldToLayerTransform3x3(layer)
        self.viewer.add_image(projected, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
    
    def gaussianFilterImageLayer(self, layer, sigma=None):
        if not self.isImageLayer(layer):
            return
        if sigma is None:
            sigma = self.gaussianSigmaSpinBox.value()
        if self.isImageStackLayer(layer):
            # default is to not blur together images in stack
            if type(sigma) is float:
                sigma = (0, sigma, sigma)
            elif len(sigma) == 1:
                sigma = (0, sigma[0], sigma[0])
            elif len(sigma) == 2:
                sigma = (0, sigma[0], sigma[1])
        filtered = filters.gaussian(layer.data, sigma=sigma, preserve_range=True)
        name = layer.name + " gauss-filt"
        tform = self.worldToLayerTransform3x3(layer)
        self.viewer.add_image(filtered, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)

    def tophatFilterImageLayer(self, layer, diskRadius=None):
        if not self.isImageLayer(layer):
            return
        if diskRadius is None:
            diskRadius = self.tophatDiskRadiusSpinBox.value()
        disk = morphology.disk(diskRadius)
        if self.isImageStackLayer(layer):
            filtered = np.empty(layer.data.shape)
            n_frames = layer.data.shape[0]
            for t in range(n_frames):
                filtered[t] = morphology.white_tophat(layer.data[t], disk)
        else:
            filtered = morphology.white_tophat(layer.data, disk)
        name = layer.name + " tophat-filt"
        tform = self.worldToLayerTransform3x3(layer)
        self.viewer.add_image(filtered, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)

    def findPeaksInImageLayer(self, layer, minPeakHeight=None, minPeakSeparation=None):
        if not self.isImageLayer(layer):
            return
        if minPeakHeight is None:
            minPeakHeight = self.minPeakHeightSpinBox.value()
        if minPeakSeparation is None:
            minPeakSeparation = self.minPeakSeparationSpinBox.value()
        if self.isImageStackLayer(layer):
            t = self.currentFrameIndex()
            image = layer.data[t]
        else:
            image = layer.data
        points = findPeaksInImage(image, minPeakHeight=minPeakHeight, minPeakSeparation=minPeakSeparation)
        n_points = len(points)
        name = layer.name + " peaks"
        tform = self.worldToLayerTransform3x3(layer)
        features = pd.DataFrame({"tags": [""] * n_points})
        self.viewer.add_points(points, name=name, affine=tform, features=features, 
            size=minPeakSeparation, edge_width=1, edge_color='yellow', edge_width_is_relative=False, 
            face_color=[0]*4, blending='translucent_no_depth', opacity=0.5)
    
    def visualizeColocalizationOfPointsLayers(self, pointsLayer, neighborsLayer, bins=30):
        if neighborsLayer is pointsLayer:
            neighborsLayer = None
        if not pointsLayer is None:
            points = self.transformPointsFromLayerToWorld(pointsLayer.data, pointsLayer)
            pointsNNs = distance.squareform(distance.pdist(points))
            np.fill_diagonal(pointsNNs, np.inf)
            pointsNNs = np.min(pointsNNs, axis=1)
        if not neighborsLayer is None:
            neighbors = self.transformPointsFromLayerToWorld(neighborsLayer.data, neighborsLayer)
            neighborsNNs = distance.squareform(distance.pdist(neighbors))
            np.fill_diagonal(neighborsNNs, np.inf)
            neighborsNNs = np.min(neighborsNNs, axis=1)
        if (not pointsLayer is None) and (not neighborsLayer is None):
            withinLayerNNs = np.concatenate([pointsNNs, neighborsNNs])
            counts, bin_edges = np.histogram(withinLayerNNs, bins=bins)
            self.withinLayersNearestNeighborsHistogram.setData(bin_edges, counts)
            betweenLayerNNs = np.min(np.linalg.norm(points[:, None, :] - neighbors[None, :, :], axis=-1), axis=1)
            counts, bin_edges = np.histogram(betweenLayerNNs, bins=bins)
            self.betweenLayersNearestNeighborsHistogram.setData(bin_edges, counts)
        elif not pointsLayer is None:
            counts, bin_edges = np.histogram(pointsNNs, bins=bins)
            self.withinLayersNearestNeighborsHistogram.setData(bin_edges, counts)
            self.betweenLayersNearestNeighborsHistogram.setData([0, 0], [0])
        elif not neighborsLayer is None:
            counts, bin_edges = np.histogram(neighborsNNs, bins=bins)
            self.withinLayersNearestNeighborsHistogram.setData(bin_edges, counts)
            self.betweenLayersNearestNeighborsHistogram.setData([0, 0], [0])
        else:
            self.withinLayersNearestNeighborsHistogram.setData([0, 0], [0])
            self.betweenLayersNearestNeighborsHistogram.setData([0, 0], [0])
    
    def updatePointsColocalizationPlot(self):
        pointsLayerName = self.pointsColocalizationLayerSelectionBox.currentText()
        neighborsLayerName = self.neighborsColocalizationLayerSelectionBox.currentText()
        try:
            pointsLayer = self.viewer.layers[pointsLayerName]
        except KeyError:
            pointsLayer = None
        try:
            neighborsLayer = self.viewer.layers[neighborsLayerName]
        except KeyError:
            neighborsLayer = None
        self.visualizeColocalizationOfPointsLayers(pointsLayer, neighborsLayer)
    
    def findColocalizedPoints(self):
        if self.viewer.grid.enabled:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Colocalization requires grid view to be disabled.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        pointsLayerName = self.pointsColocalizationLayerSelectionBox.currentText()
        neighborsLayerName = self.neighborsColocalizationLayerSelectionBox.currentText()
        try:
            pointsLayer = self.viewer.layers[pointsLayerName]
        except KeyError:
            pointsLayer = None
        try:
            neighborsLayer = self.viewer.layers[neighborsLayerName]
        except KeyError:
            neighborsLayer = None
        if neighborsLayer is pointsLayer:
            neighborsLayer = None
        if (pointsLayer is None) or (neighborsLayer is None):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Select two points layers for colocalization.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        points = self.transformPointsFromLayerToWorld(pointsLayer.data, pointsLayer)
        neighbors = self.transformPointsFromLayerToWorld(neighborsLayer.data, neighborsLayer)
        pdists = np.linalg.norm(points[:, None, :] - neighbors[None, :, :], axis=-1)
        pointsNearestNeighborsDists = pdists.min(axis=1)
        # neighborsNearestPointsDists = pdists.min(axis=0)
        pointsNearestNeighborsIndexes = pdists.argmin(axis=1)
        neighborsNearestPointsIndexes = pdists.argmin(axis=0)
        colocalizedPointNeighborIndexes = []
        cutoff = self.nearestNeighborDistanceCutoffSpinBox.value()
        for i in range(len(points)):
            j = pointsNearestNeighborsIndexes[i]
            if neighborsNearestPointsIndexes[j] == i:
                if pointsNearestNeighborsDists[i] <= cutoff:
                    colocalizedPointNeighborIndexes.append([i, j])
        colocalizedPointNeighborIndexes = np.array(colocalizedPointNeighborIndexes, dtype=int)
        colocalizedPointIndexes = colocalizedPointNeighborIndexes[:,0]
        colocalizedNeighborIndexes = colocalizedPointNeighborIndexes[:,1]
        colocalizedPoints = np.reshape(points[colocalizedPointIndexes], [-1, 2])
        colocalizedNeighbors = np.reshape(neighbors[colocalizedNeighborIndexes], [-1, 2])
        colocalizedPoints = (colocalizedPoints + colocalizedNeighbors) / 2
        colocalizedPoints = self.transformPointsFromWorldToLayer(colocalizedPoints, pointsLayer)
        n_points = len(colocalizedPoints)
        size = pointsLayer.size[colocalizedPointIndexes]
        name = "colocalized " + pointsLayer.name + "-" + neighborsLayer.name
        tform = self.worldToLayerTransform3x3(pointsLayer)
        features = pd.DataFrame({"tags": [""] * n_points})
        self.viewer.add_points(colocalizedPoints, name=name, affine=tform, features=features, 
            size=size, edge_width=1, edge_color='yellow', edge_width_is_relative=False, 
            face_color=[0]*4, blending='translucent_no_depth', opacity=0.5)
    
    def setSelectedPointsLayersPointSize(self, pointSize=None):
        layers = [layer for layer in list(self.viewer.layers.selection) if self.isPointsLayer(layer)]
        if len(layers) == 0:
            return
        if pointSize is None:
            pointSize, ok = QInputDialog().getDouble(self, "Selected Points Layers", "Point size:", 9, 0.1, 100, 1)
            if not ok:
                return
        for layer in layers:
            layer.size = pointSize
    
    def setSelectedPointsLayersEdgeWidth(self, edgeWidth=None):
        layers = [layer for layer in list(self.viewer.layers.selection) if self.isPointsLayer(layer)]
        if len(layers) == 0:
            return
        if edgeWidth is None:
            edgeWidth, ok = QInputDialog().getDouble(self, "Selected Points Layers", "Edge width (pixels):", 1.0, 0.1, 100, 1)
            if not ok:
                return
        for layer in layers:
            layer.edge_width_is_relative = False
            layer.edge_width = edgeWidth
    
    def applyToAllLayers(self, func, *args, **kwargs):
        layers = list(self.viewer.layers)
        for layer in layers:
            func(layer, *args, **kwargs)
    
    def applyToSelectedLayers(self, func, *args, **kwargs):
        layers = list(self.viewer.layers.selection)
        for layer in layers:
            func(layer, *args, **kwargs)
    

def normalizeImage(image, contrastLimits=None) -> np.ndarray:
    image = image.astype(float)
    if contrastLimits is None:
        cmin, cmax = image.min(), image.max()
    else:
        cmin, cmax = contrastLimits
    image -= cmin
    image /= cmax
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def registerImages(fixedImage, movingImage, transformType="Affine") -> np.ndarray:
    if transformType == "Translation":
        sreg = StackReg(StackReg.TRANSLATION)
    elif transformType == "Rigid Body":
        sreg = StackReg(StackReg.RIGID_BODY)
    elif transformType == "Affine":
        sreg = StackReg(StackReg.AFFINE)
    tform = sreg.register(fixedImage, movingImage)
    return tform


def findPeaksInImage(image, minPeakHeight=None, minPeakSeparation=2) -> np.ndarray:
    pixelRadius = max(1, math.ceil(minPeakSeparation / 2))
    if minPeakHeight is None or minPeakHeight == 0:
        peakMask = morphology.local_maxima(image, connectivity=pixelRadius, indices=False, allow_borders=False)
    else:
        disk = morphology.disk(pixelRadius)
        peakMask = morphology.h_maxima(image, h=minPeakHeight, footprint=disk) > 0
        peakMask[:,0] = False
        peakMask[0,:] = False
        peakMask[:,-1] = False
        peakMask[-1,:] = False
    labelImage = measure.label(peakMask)
    rois = measure.regionprops(labelImage)
    n_rois = len(rois)
    points = np.zeros((n_rois, 2))
    for i, roi in enumerate(rois):
        points[i] = roi.centroid
    return points


def zprojectPointInImageStack(imageStack, point, pointMask=None) -> np.ndarray:
    # pixel (row, col) from subpixel float point
    row, col = np.round(point).astype(int)
    # find overlap of point mask with image
    n_rows, n_cols = imageStack.shape[-2:]
    if pointMask is None or np.all(pointMask.shape == 1):
        # z-project single pixel
        if (0 <= row < n_rows) and (0 <= col < n_cols):
            zproj = np.squeeze(imageStack[:,row,col])
        else:
            zproj = np.array([])
    else:
        # z-project overlap between mask and image
        n_mask_rows, n_mask_cols = pointMask.shape
        if n_mask_rows % 2 == 0:
            # even rows
            drows = n_mask_rows / 2
            if point[0] >= row:
                rows = np.arange(row - drows + 1, row + drows + 1, dtype=int)
            else:
                rows = np.arange(row - drows, row + drows, dtype=int)
        else:
            # odd rows
            drows = (n_mask_rows - 1) / 2
            rows = np.arange(row - drows, row + drows + 1, dtype=int)
        if n_mask_cols % 2 == 0:
            # even cols
            dcols = n_mask_cols / 2
            if point[1] >= col:
                cols = np.arange(col - dcols + 1, col + dcols + 1, dtype=int)
            else:
                cols = np.arange(col - dcols, col + dcols, dtype=int)
        else:
            # odd cols
            dcols = (n_mask_cols - 1) / 2
            cols = np.arange(col - dcols, col + dcols + 1, dtype=int)
        rowsInImage = (rows >= 0) & (rows < n_rows)
        colsInImage = (cols >= 0) & (cols < n_cols)
        if np.any(rowsInImage) and np.any(colsInImage):
            rows = rows[rowsInImage]
            cols = cols[colsInImage]
            pointMaskInImage = np.reshape(rowsInImage, [-1, 1]) & np.reshape(colsInImage, [1, -1])
            pointMask = np.reshape(pointMask[pointMaskInImage], [1, len(rows), len(cols)])
            zproj = np.squeeze(np.sum(imageStack[:,rows[0]:rows[-1]+1,cols[0]:cols[-1]+1] * pointMask, axis=(1, 2)))
        else:
            zproj = np.array([])
    return zproj

    
def printDictStructure(dic, start="", indent="\t"):
    for key, value in dic.items():
        if type(value) is np.ndarray:
            szstr = "x".join([str(dimsz) for dimsz in value.shape])
            print(start, indent, key, szstr)
        elif type(value) is list:
            szstr = "x" + str(len(value))
            print(start, indent, key, szstr)
        elif type(value) is dict:
            print(start, indent, key)
            printDictStructure(value, start + indent, indent)
        else:
            try:
                print(start, indent, key, str(value))
            except:
                print(start, indent, key, '?')


if __name__ == "__main__":
    viewer = napari.Viewer()
    ui = CoSMoS_TS_napari_UI(viewer)
    viewer.window.add_dock_widget(ui, name='CoSMoS-TS', area='right')
    napari.run()