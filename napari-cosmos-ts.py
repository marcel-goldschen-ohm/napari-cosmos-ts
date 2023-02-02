""" napari plugin for analysis of colocalization single-molecule spectroscopy (CoSMoS) time series (TS)
"""

import os
import math
# from colorsys import rgb_to_hsv, hsv_to_rgb
# from pprint import pprint
import numpy as np
import pandas as pd
import tifffile
import scipy.io as sio
from scipy.spatial import distance
from skimage import filters, morphology, measure
import napari
import pyqtgraph as pg
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QTabWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QFormLayout, QGroupBox, \
    QFileDialog, QTextEdit, QComboBox, QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox, QMessageBox, QDoubleSpinBox, \
    QInputDialog
try:
    from pystackreg import StackReg
except ImportError:
    StackReg = None
try:
    from pycpd import AffineRegistration
except ImportError:
    AffineRegistration = None

__author__ = "Marcel Goldschen-Ohm <goldschen-ohm@utexas.edu, marcel.goldschen@gmail.com>"
__version__ = '1.0.0'


class CoSMoS_TS_napari_UI(QTabWidget):
    """ napari dock widget for analysis of colocalization single-molecule spectroscopy (CoSMoS) time series (TS)

        Image layer.metadata
            ['image_file_abspath'] = "abs/path/to/image/"
            ['subimage_slice'] = [ndim x 2] start:stop or [ndim x 3] start:stop:step slice
            ['roi_zprojections']
                [roi-layer-name] = [NxT] zprojections for ROIs in roi-layer-name
        
        Shapes/Points layer.features
            ['tags'] = series of string tags for each point
        
        layerData
            ['roi_zprojection_plot'] = pyqtgraph plot object
            ['roi_zprojection_plot_data'] = pyqtgraph data series plot object
            ['roi_zprojection_plot_vline'] = pyqtgraph vertical line plot object
    
        TODO
        - account for grid view?
        - support per frame point positions (point tracking)? i.e., napari.layers.Tracks
        - point size does not scale during zoom on Mac! (see https://github.com/vispy/vispy/issues/2078, maybe no fix?)
    """
    
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # Array of dicts for metadata belonging to each layer.
        # Stored separately to keep non-serealizable objects (e.g., QObjects) out of the layer metadata dicts.
        # This unfortunately means we will have to manage this list upon any change to the layer order.
        # Other layer data that can be serialized and should be saved can be stored in the layer metadata dicts.
        self._layerMetadata = []

        # For copying and pasting layer transforms.
        self._layerTransformCopy = np.eye(3)

        # Keep track of selected point index
        # so we know if we are incrementing or decrementing the index when filtering tags.
        self._roiIndex = None

        # Special layer for selected ROI highlighting
        self._selectedRoiLayer = None

        # setup UI
        self.initUI()

        # event handling
        self.viewer.layers.events.inserted.connect(self.onLayerInserted)
        self.viewer.layers.events.removed.connect(self.onLayerRemoved)
        self.viewer.layers.events.moved.connect(self.onLayerMoved)
        self.viewer.dims.events.current_step.connect(self.onDimStepChanged)
        self.viewer.mouse_drag_callbacks.append(self.onMouseClickedOrDragged)
        self.viewer.mouse_double_click_callbacks.append(self.onMouseDoubleClicked)

        # testing
        # self.unitTests()
        self.TO_BE_REMOVED_customInit()

    
    # TESTING
    
    def unitTests(self):
        data2d = np.random.randint(0, 65536 + 1, [1024, 1024]).astype(np.uint16)
        data3d = np.random.randint(0, 65536 + 1, [100, 1024, 1024]).astype(np.uint16)
        points = np.random.uniform(10, 1013, size=[10, 2])

        if not os.path.isdir("unit-tests/"):
            os.makedirs("unit-tests/")
        data2dPath = "unit-tests/data2d.tif"
        data3dPath = "unit-tests/data3d.tif"
        tifffile.imwrite(data2dPath, data2d)
        tifffile.imwrite(data3dPath, data3d)

        layer = self.openTIFF(data2dPath, memorymap=False)
        layer.colormap = 'green'

        layer = self.openTIFF(data3dPath, memorymap=False)
        layer.colormap = 'magenta'

        layer = self.openTIFF(data2dPath, memorymap=True)
        layer.name = "data2d memmap"
        layer.colormap = 'cyan'

        layer = self.openTIFF(data3dPath, memorymap=True)
        layer.name = "data3d memmap"
        layer.colormap = 'red'

        # test single pixel Z-projections
        layer = self.addRoisLayer(center=points, size=1, shape_type="point", name="points")
        pointFeatures = layer.features.copy()
        self.zprojectRoisInImageLayers(layer)
        for i, point in enumerate(points):
            row, col = np.round(point).astype(int)
            for imageLayer in self.imageStackLayers():
                zproj = imageLayer.data[:,row,col]
                assert(np.allclose(zproj, imageLayer.metadata['roi_zprojections'][layer.name][i]))

        # test ellipitcal ROI Z-projections
        ellipseSize = 5
        ellipseFeatures = pd.DataFrame({"tags": [str(i) for i in range(len(layer.data))]})
        ellipseFeatures.loc[3,"tags"] = ""
        layer = self.addRoisLayer(center=points, size=ellipseSize, shape_type="ellipse", name="ellipses", features=ellipseFeatures)
        self.zprojectRoisInImageLayers(layer)
        mask = getRoiMask(ellipseSize, "ellipse")
        for i, point in enumerate(points):
            for imageLayer in self.imageStackLayers():
                zproj = zprojectRoiInImage(imageLayer.data, point, mask)
                assert(np.allclose(zproj, imageLayer.metadata['roi_zprojections'][layer.name][i]))

        # test rectangular ROI Z-projections
        rectSize = 5
        layer = self.addRoisLayer(center=points, size=rectSize, shape_type="rectangle", name="rectangles")
        rectFeatures = layer.features.copy()
        self.zprojectRoisInImageLayers(layer)
        for i, point in enumerate(points):
            row, col = np.round(point).astype(int)
            for imageLayer in self.imageStackLayers():
                zproj = np.squeeze(imageLayer.data[:,row-2:row+3,col-2:col+3].mean(axis=(-2,-1)))
                assert(np.allclose(zproj, imageLayer.metadata['roi_zprojections'][layer.name][i]))

        # test session file i/o
        sessionPath = 'unit-tests/session.mat'
        self.exportSession(sessionPath)
        self.viewer.layers.clear()
        self.importSession(sessionPath)

        # verify session file paths
        sesseionAbsPath = os.path.abspath(sessionPath)
        sessionAbsDir, sessionFile = os.path.split(sesseionAbsPath)
        data2dAbsPath = os.path.abspath(data2dPath)
        data3dAbsPath = os.path.abspath(data3dPath)
        data2dRelPath = os.path.relpath(data2dAbsPath, start=sessionAbsDir)
        data3dRelPath = os.path.relpath(data3dAbsPath, start=sessionAbsDir)

        # verify session data
        layer = self.viewer.layers["data2d"]
        assert(np.all(layer.data == data2d))
        assert(layer.metadata['image_file_abspath'] == data2dAbsPath)
        layer = self.viewer.layers["data3d"]
        assert(np.all(layer.data == data3d))
        assert(layer.metadata['image_file_abspath'] == data3dAbsPath)
        layer = self.viewer.layers["data2d memmap"]
        assert(np.all(layer.data == data2d))
        assert(layer.metadata['image_file_abspath'] == data2dAbsPath)
        layer = self.viewer.layers["data3d memmap"]
        assert(np.all(layer.data == data3d))
        assert(layer.metadata['image_file_abspath'] == data3dAbsPath)
        layer = self.viewer.layers["points"]
        assert(np.allclose(layer.data, points))
        assert(np.allclose(layer.size, 1))
        for col in layer.features.columns:
            assert((layer.features[col] == pointFeatures[col]).all())
        for i, point in enumerate(points):
            row, col = np.round(point).astype(int)
            for imageLayer in self.imageStackLayers():
                zproj = imageLayer.data[:,row,col]
                assert(np.allclose(zproj, imageLayer.metadata['roi_zprojections'][layer.name][i]))
        layer = self.viewer.layers["ellipses"]
        centers = self.getRoiCenters2d(layer)
        sizes = self.getRoiSizes2d(layer)
        assert(np.allclose(centers, points))
        assert(np.allclose(sizes, ellipseSize))
        for col in layer.features.columns:
            assert((layer.features[col] == ellipseFeatures[col]).all())
        mask = getRoiMask(ellipseSize, "ellipse")
        for i, point in enumerate(points):
            for imageLayer in self.imageStackLayers():
                zproj = zprojectRoiInImage(imageLayer.data, point, mask)
                assert(np.allclose(zproj, imageLayer.metadata['roi_zprojections'][layer.name][i]))
        layer = self.viewer.layers["rectangles"]
        centers = self.getRoiCenters2d(layer)
        sizes = self.getRoiSizes2d(layer)
        assert(np.allclose(centers, points))
        assert(np.allclose(sizes, rectSize))
        for col in layer.features.columns:
            assert((layer.features[col] == rectFeatures[col]).all())
        mask = getRoiMask(rectSize, "rectangle")
        for i, point in enumerate(points):
            for imageLayer in self.imageStackLayers():
                zproj = zprojectRoiInImage(imageLayer.data, point, mask)
                assert(np.allclose(zproj, imageLayer.metadata['roi_zprojections'][layer.name][i]))
        print("Unit tests completed successfully.")
    
    def TO_BE_REMOVED_customInit(self):
        # This probably does NOT apply to you.
        # This is for custom files as a quick test
        # and will be removed in a future version.
        theta = 0.4
        tform = np.array([
            [np.cos(theta), -np.sin(theta), 100],
            [np.sin(theta), np.cos(theta), -25],
            [0, 0, 1]
        ])

        layer = self.openTIFF('test.tif')
        layer.name = 'eGFP'
        layer.colormap = 'green'
        layer = self.zprojectImageLayer(layer, method="mean")
        layer = self.gaussianFilterImageLayer(layer, sigma=1)
        layer = self.tophatFilterImageLayer(layer, diskRadius=3)
        layer = self.findPeakRoisInImageLayer(layer, minPeakHeight=10, minPeakSeparation=2.5)
        layer.edge_color = 'red'
        del self.viewer.layers[-4:-2]

        layer = self.openTIFF('test.tif')
        layer.name = 'fcGMP'
        layer.colormap = 'magenta'
        layer.affine = tform
        layer = self.zprojectImageLayer(layer, method="mean")
        layer = self.gaussianFilterImageLayer(layer, sigma=1)
        layer = self.tophatFilterImageLayer(layer, diskRadius=3)
        layer = self.findPeakRoisInImageLayer(layer, minPeakHeight=10, minPeakSeparation=2.5)
        layer.edge_color = 'cyan'
        del self.viewer.layers[-4:-2]
    
    # UI
    
    def initUI(self):
        self.addMetadataTab()
        self.addFileTab()
        self.addImageProcessingTab()
        self.addLayerRegistrationTab()
        self.addRoisTab()
        self.addColocalizationTab()
        self.addRoiZProjectionsTab()
    
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
    
    def addFileTab(self, title="File"):
        self.openMemoryMappedTiffButton = QPushButton("Open memory-mapped TIFF image file")
        self.openMemoryMappedTiffButton.clicked.connect(lambda x: self.openTIFF())

        self.openSessionButton = QPushButton("Open .mat session file")
        self.openSessionButton.clicked.connect(lambda x: self.importSession())

        self.saveSessionButton = QPushButton("Save session as .mat file")
        self.saveSessionButton.clicked.connect(lambda x: self.exportSession())

        tab = QWidget()
        hbox = QHBoxLayout(tab)
        vbox = QVBoxLayout()
        vbox.setSpacing(5)
        vbox.addWidget(self.openMemoryMappedTiffButton)
        vbox.addWidget(self.openSessionButton)
        vbox.addWidget(self.saveSessionButton)
        vbox.addStretch()
        hbox.addLayout(vbox)
        hbox.addStretch()
        self.addTab(tab, title)
    
    def addImageProcessingTab(self, title="Image"):
        self.splitImageButton = QPushButton("Split Image")
        self.splitImageButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.splitImageLayer))

        self.splitImageRegionsComboBox = QComboBox()
        self.splitImageRegionsComboBox.addItem("Top/Bottom")
        self.splitImageRegionsComboBox.addItem("Left/Right")
        self.splitImageRegionsComboBox.addItem("Quad")
        self.splitImageRegionsComboBox.setCurrentText("Top/Bottom")

        self.sliceImageButton = QPushButton("Slice Image")
        self.sliceImageButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.sliceImageLayer))

        self.sliceImageEdit = QLineEdit()

        self.zprojectImageButton = QPushButton("Z-Project Image")
        self.zprojectImageButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.zprojectImageLayer))

        self.zprojectImageOperationComboBox = QComboBox()
        self.zprojectImageOperationComboBox.addItem("max")
        self.zprojectImageOperationComboBox.addItem("min")
        self.zprojectImageOperationComboBox.addItem("std")
        self.zprojectImageOperationComboBox.addItem("sum")
        self.zprojectImageOperationComboBox.addItem("mean")
        self.zprojectImageOperationComboBox.addItem("median")
        self.zprojectImageOperationComboBox.setCurrentText("mean")

        self.zprojectImageFrameSliceEdit = QLineEdit()

        self.gaussianFilterButton = QPushButton("Gaussian Filter")
        self.gaussianFilterButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.gaussianFilterImageLayer))

        self.gaussianFilterSigmaSpinBox = QDoubleSpinBox()
        self.gaussianFilterSigmaSpinBox.setValue(1)

        self.tophatFilterButton = QPushButton("Tophat Filter")
        self.tophatFilterButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.tophatFilterImageLayer))

        self.tophatFilterDiskRadiusSpinBox = QDoubleSpinBox()
        self.tophatFilterDiskRadiusSpinBox.setValue(3)

        tab = QWidget()
        hbox = QHBoxLayout(tab)
        vbox = QVBoxLayout()
        vbox.setSpacing(5)

        text = QLabel("Applied to all selected image layers.\nResults are returned in new layers.")
        text.setWordWrap(True)
        vbox.addWidget(text)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.splitImageButton)
        form.addRow("Regions", self.splitImageRegionsComboBox)
        vbox.addWidget(group)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.sliceImageButton)
        form.addRow("start:stop[:step],...", self.sliceImageEdit)
        vbox.addWidget(group)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.zprojectImageButton)
        form.addRow("Projection", self.zprojectImageOperationComboBox)
        form.addRow("start:stop[:step]", self.zprojectImageFrameSliceEdit)
        vbox.addWidget(group)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.gaussianFilterButton)
        form.addRow("Sigma", self.gaussianFilterSigmaSpinBox)
        vbox.addWidget(group)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.tophatFilterButton)
        form.addRow("Disk Radius", self.tophatFilterDiskRadiusSpinBox)
        vbox.addWidget(group)

        vbox.addStretch()
        hbox.addLayout(vbox)
        hbox.addStretch()
        self.addTab(tab, title)
    
    def addLayerRegistrationTab(self, title="Align"):
        self.fixedLayerComboBox = QComboBox()
        self.movingLayerComboBox = QComboBox()

        self.layerRegistrationTransformTypeComboBox = QComboBox()
        self.layerRegistrationTransformTypeComboBox.addItem("Translation")
        self.layerRegistrationTransformTypeComboBox.addItem("Rigid Body")
        self.layerRegistrationTransformTypeComboBox.addItem("Affine")
        self.layerRegistrationTransformTypeComboBox.setCurrentText("Affine")

        self.registerLayersButton = QPushButton("Register Layers")
        self.registerLayersButton.clicked.connect(lambda x: self.registerLayers())

        self.copyLayerTransformButton = QPushButton("Copy selected layer transform")
        self.copyLayerTransformButton.clicked.connect(lambda x: self.copyLayerTransform())

        self.pasteLayerTransformButton = QPushButton("Paste copied transform to all selected layers")
        self.pasteLayerTransformButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.pasteCopiedLayerTransform))

        self.clearLayerTransformButton = QPushButton("Clear transform from all selected layers")
        self.clearLayerTransformButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.clearLayerTransform))

        tab = QWidget()
        hbox = QHBoxLayout(tab)
        vbox = QVBoxLayout()
        vbox.setSpacing(5)

        text = QLabel("Registration sets the layer affine transform without altering the layer data.")
        text.setWordWrap(True)
        vbox.addWidget(text)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.registerLayersButton)
        form.addRow("Fixed Layer", self.fixedLayerComboBox)
        form.addRow("Moving Layer", self.movingLayerComboBox)
        form.addRow("Transform", self.layerRegistrationTransformTypeComboBox)
        vbox.addWidget(group)

        vbox.addWidget(self.copyLayerTransformButton)
        vbox.addWidget(self.pasteLayerTransformButton)
        vbox.addWidget(self.clearLayerTransformButton)

        vbox.addStretch()
        hbox.addLayout(vbox)
        hbox.addStretch()
        self.addTab(tab, title)
    
    def addRoisTab(self, title="ROIs"):
        self.updateRoisButton = QPushButton("Update ROIs in all selected ROI layers")
        self.updateRoisButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.updateRoisLayer))

        self.roiShapeComboBox = QComboBox()
        self.roiShapeComboBox.addItem("point")
        self.roiShapeComboBox.addItem("ellipse")
        self.roiShapeComboBox.addItem("rectangle")
        self.roiShapeComboBox.setCurrentText("ellipse")
        self.roiShapeComboBox.currentTextChanged.connect(self.updateRoiMaskWidget)

        self.roiSizeSpinBox = QDoubleSpinBox()
        self.roiSizeSpinBox.setMinimum(1)
        self.roiSizeSpinBox.setMaximum(65000)
        self.roiSizeSpinBox.setValue(5)
        self.roiSizeSpinBox.valueChanged.connect(self.updateRoiMaskWidget)

        self.roiEdgeWidthSpinBox = QDoubleSpinBox()
        self.roiEdgeWidthSpinBox.setMinimum(0)
        self.roiEdgeWidthSpinBox.setMaximum(100)
        self.roiEdgeWidthSpinBox.setSingleStep(0.25)
        self.roiEdgeWidthSpinBox.setValue(0.5)

        self.roiEdgeColorEdit = QLineEdit("yellow")

        self.roiFaceColorEdit = QLineEdit("")

        self.selectedRoiEdgeWidthSpinBox = QDoubleSpinBox()
        self.selectedRoiEdgeWidthSpinBox.setMinimum(0)
        self.selectedRoiEdgeWidthSpinBox.setMaximum(100)
        self.selectedRoiEdgeWidthSpinBox.setSingleStep(0.25)
        self.selectedRoiEdgeWidthSpinBox.setValue(1)

        self.selectedRoiEdgeColorEdit = QLineEdit("cyan")

        self.findPeakPointsButton = QPushButton("Find peaks in all selected image layers")
        self.findPeakPointsButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.findPeakRoisInImageLayer))

        self.minPeakHeightSpinBox = QDoubleSpinBox()
        self.minPeakHeightSpinBox.setMinimum(0)
        self.minPeakHeightSpinBox.setMaximum(65000)
        self.minPeakHeightSpinBox.setValue(10)

        self.minPeakSeparationSpinBox = QDoubleSpinBox()
        self.minPeakSeparationSpinBox.setMinimum(1)
        self.minPeakSeparationSpinBox.setMaximum(65000)
        self.minPeakSeparationSpinBox.setValue(self.roiSizeSpinBox.value())

        self.zprojectRoisLayerButton = QPushButton("Compute ROI z-projections for all selected ROI layers")
        self.zprojectRoisLayerButton.clicked.connect(lambda x: self.applyToSelectedLayers(self.zprojectRoisInImageLayers))

        self.roiMaskWidget = QWidget()
        self.roiMaskGrid = QGridLayout(self.roiMaskWidget)
        self.roiMaskGrid.setContentsMargins(5, 5, 5, 5)
        self.roiMaskGrid.setSpacing(2)
        self.updateRoiMaskWidget()

        tab = QWidget()
        hbox = QHBoxLayout(tab)

        # column 1
        vbox = QVBoxLayout()
        vbox.setSpacing(5)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.updateRoisButton)
        form.addRow("ROI Shape", self.roiShapeComboBox)
        form.addRow("ROI Size", self.roiSizeSpinBox)
        form.addRow("ROI Edge Width", self.roiEdgeWidthSpinBox)
        form.addRow("ROI Edge Color", self.roiEdgeColorEdit)
        form.addRow("ROI Face Color", self.roiFaceColorEdit)
        form.addRow("Selected ROI Edge Width", self.selectedRoiEdgeWidthSpinBox)
        form.addRow("Selected ROI Edge Color", self.selectedRoiEdgeColorEdit)
        vbox.addWidget(group)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.findPeakPointsButton)
        form.addRow("Min Peak Height", self.minPeakHeightSpinBox)
        form.addRow("Min Separation", self.minPeakSeparationSpinBox)
        vbox.addWidget(group)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.zprojectRoisLayerButton)
        vbox.addWidget(group)

        vbox.addStretch()
        hbox.addLayout(vbox)

        # column 2
        vbox = QVBoxLayout()

        group = QGroupBox("ROI Mask")
        gvbox = QVBoxLayout(group)
        gvbox.addWidget(self.roiMaskWidget)
        vbox.addWidget(group)

        vbox.addStretch()
        hbox.addLayout(vbox)

        hbox.addStretch()
        self.addTab(tab, title)
    
    def updateRoiMaskWidget(self):
        roiShape = self.roiShapeComboBox.currentText()
        roiSize = self.roiSizeSpinBox.value()
        roiMask = getRoiMask(roiSize, roiShape).astype(float)
        clearQLayout(self.roiMaskGrid)
        n_rows, n_cols = roiMask.shape
        for i in range(n_rows):
            for j in range(n_cols):
                cell = QLineEdit(str(roiMask[i,j]))
                cell.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
                bg = min(max(0, int(roiMask[i,j] * 255)), 255)
                bg = f"rgb({bg}, {bg}, {bg})"
                fg = "white" if roiMask[i,j] < 0.5 else "black"
                cell.setStyleSheet(f"width: 30; height: 30; background-color: {bg}; color: {fg}; font-size: 8pt")
                cell.editingFinished.connect(self.updateRoiMaskWidget)
                self.roiMaskGrid.addWidget(cell, i, j)
    
    def addColocalizationTab(self, title="Colocalization"):
        self.findColocalizedRoisButton = QPushButton("Find colocalized ROIs")
        self.findColocalizedRoisButton.clicked.connect(self.findColocalizedRois)

        self.colocalizeRoisLayerComboBox = QComboBox()
        self.colocalizeRoisLayerComboBox.currentTextChanged.connect(self.updateRoisColocalizationPlot)
        self.colocalizeNeighborsLayerComboBox = QComboBox()
        self.colocalizeNeighborsLayerComboBox.currentTextChanged.connect(self.updateRoisColocalizationPlot)

        self.colocalizeNearestNeighborCutoffSpinBox = QDoubleSpinBox()
        self.colocalizeNearestNeighborCutoffSpinBox.setMinimum(0)
        self.colocalizeNearestNeighborCutoffSpinBox.setMaximum(1000)
        self.colocalizeNearestNeighborCutoffSpinBox.setSingleStep(0.5)
        self.colocalizeNearestNeighborCutoffSpinBox.setValue(self.roiSizeSpinBox.value() / 2)

        self.roiColocalizationPlot = self.newPlot()
        self.roiColocalizationPlot.setLabels(left="Counts", bottom="Nearest Neighbor Distance")
        legend = pg.LegendItem()
        legend.setParentItem(self.roiColocalizationPlot.getPlotItem())
        legend.anchor((1,0), (1,0))
        self.withinRoiLayersNearestNeighborsHistogram = pg.PlotCurveItem([0, 0], [0], name="within layers", 
            stepMode='center', pen=pg.mkPen([98, 143, 176, 80], width=1), fillLevel=0, brush=(98, 143, 176, 80))
        self.betweenRoiLayersNearestNeighborsHistogram = pg.PlotCurveItem([0, 0], [0], name="between layers", 
            stepMode='center', pen=pg.mkPen([255, 0, 0, 80], width=1), fillLevel=0, brush=(255, 0, 0, 80))
        self.roiColocalizationPlot.addItem(self.withinRoiLayersNearestNeighborsHistogram)
        self.roiColocalizationPlot.addItem(self.betweenRoiLayersNearestNeighborsHistogram)
        legend.addItem(self.withinRoiLayersNearestNeighborsHistogram, "within layers")
        legend.addItem(self.betweenRoiLayersNearestNeighborsHistogram, "between layers")

        tab = QWidget()
        hbox = QHBoxLayout(tab)
        vbox = QVBoxLayout()
        vbox.setSpacing(5)

        group = QGroupBox()
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self.findColocalizedRoisButton)
        form.addRow("ROIs Layer", self.colocalizeRoisLayerComboBox)
        form.addRow("Neighbors Layer", self.colocalizeNeighborsLayerComboBox)
        form.addRow("Nearest Neighbor Distance Cutoff", self.colocalizeNearestNeighborCutoffSpinBox)
        form.addRow(self.roiColocalizationPlot)
        vbox.addWidget(group)

        vbox.addStretch()
        hbox.addLayout(vbox)
        hbox.addStretch()
        self.addTab(tab, title)
    
    def addRoiZProjectionsTab(self, title="ROI Z-Projections"):
        self.roiZProjectionPlotsLayout = QVBoxLayout()
        self.roiZProjectionPlotsLayout.setSpacing(0)

        self.activeRoisLayerComboBox = QComboBox()
        self.activeRoisLayerComboBox.textActivated.connect(lambda x: self.setSelectedRoiIndex())

        self.roiIndexSpinBox = QSpinBox()
        self.roiIndexSpinBox.setMaximum(65000)
        self.roiIndexSpinBox.setKeyboardTracking(False)
        self.roiIndexSpinBox.valueChanged.connect(self.setSelectedRoiIndex)

        self.currentRoiWorldPositionLabel = QLabel()
        self.numSelectedRoisLabel = QLabel()

        self.roiTagsEdit = QLineEdit()
        self.roiTagsEdit.setMinimumWidth(100)
        self.roiTagsEdit.editingFinished.connect(self.updateSelectedRoiTags)

        self.roiTagFilterEdit = QLineEdit()
        self.roiTagFilterEdit.setMinimumWidth(100)
        self.roiTagFilterEdit.editingFinished.connect(self.setSelectedRoiIndex)

        self.roiTagFilterCheckBox = QCheckBox("Filter")
        self.roiTagFilterCheckBox.stateChanged.connect(lambda state: self.setSelectedRoiIndex())

        self.noVisibleImageStacksLabel = QLabel("No visible image stacks.")

        grid = QGridLayout()
        grid.setSpacing(5)
        grid.addWidget(QLabel("ROIs Layer"), 0, 0, Qt.AlignRight)
        grid.addWidget(self.activeRoisLayerComboBox, 0, 1)
        grid.addWidget(self.numSelectedRoisLabel, 0, 2)
        grid.addWidget(self.roiTagFilterCheckBox, 0, 3)
        grid.addWidget(self.roiTagFilterEdit, 0, 4)
        grid.addWidget(QLabel("ROI Index"), 1, 0, Qt.AlignRight)
        grid.addWidget(self.roiIndexSpinBox, 1, 1)
        grid.addWidget(self.currentRoiWorldPositionLabel, 1, 2)
        grid.addWidget(QLabel("ROI Tags"), 1, 3, Qt.AlignRight)
        grid.addWidget(self.roiTagsEdit, 1, 4)

        tab = QWidget()
        vbox = QVBoxLayout(tab)
        vbox.addLayout(grid)
        vbox.addLayout(self.roiZProjectionPlotsLayout)
        vbox.addWidget(self.noVisibleImageStacksLabel)
        self.addTab(tab, title)
    
    def roiZProjectionPlots(self):
        return [data['roi_zprojection_plot'] for data in reversed(self._layerMetadata) if 'roi_zprojection_plot' in data]
    
    def linkRoiZProjectionPlots(self):
        plots = self.roiZProjectionPlots()
        for i in range(1, len(plots)):
            plots[i].setXLink(plots[0])
    
    def clearRoiZProjectionPlots(self):
        for layerMetadata in self._layerMetadata:
            if 'roi_zprojection_plot_data' in layerMetadata:
                layerMetadata['roi_zprojection_plot_data'].setData([])
    
    def newPlot(self) -> pg.PlotWidget:
        plot = pg.PlotWidget()
        plot.getAxis('left').setWidth(82)
        plot.getAxis('right').setWidth(10)
        plot.showGrid(x=True, y=True, alpha=0.3)
        #plot.setBackground([38, 41, 48])
        # hack to stop grid from clipping axis tick labels
        for key in ['left', 'bottom']:
            plot.getAxis(key).setGrid(False)
        for key in ['right', 'top']:
            plot.getAxis(key).setStyle(showValues=False)
            plot.showAxis(key)
        return plot
    
    def updateLayerSelectionBoxes(self):
        # layer registration
        fixedLayerName = self.fixedLayerComboBox.currentText()
        movingLayerName = self.movingLayerComboBox.currentText()
        self.fixedLayerComboBox.clear()
        self.movingLayerComboBox.clear()
        for layer in reversed(self.viewer.layers):
            self.fixedLayerComboBox.addItem(layer.name)
            self.movingLayerComboBox.addItem(layer.name)
        try:
            self.fixedLayerComboBox.setCurrentText(fixedLayerName)
        except:
            self.fixedLayerComboBox.setCurrentIndex(0)
        try:
            self.movingLayerComboBox.setCurrentText(movingLayerName)
        except:
            self.movingLayerComboBox.setCurrentIndex(0)
        
        # layer ROI colocalization
        self.colocalizeRoisLayerComboBox.currentTextChanged.disconnect(self.updateRoisColocalizationPlot)
        self.colocalizeNeighborsLayerComboBox.currentTextChanged.disconnect(self.updateRoisColocalizationPlot)
        roisLayerName = self.colocalizeRoisLayerComboBox.currentText()
        neighborsLayerName = self.colocalizeNeighborsLayerComboBox.currentText()
        self.colocalizeRoisLayerComboBox.clear()
        self.colocalizeNeighborsLayerComboBox.clear()
        for layer in self.roisLayers():
            self.colocalizeRoisLayerComboBox.addItem(layer.name)
            self.colocalizeNeighborsLayerComboBox.addItem(layer.name)
        try:
            self.colocalizeRoisLayerComboBox.setCurrentText(roisLayerName)
        except:
            self.colocalizeRoisLayerComboBox.setCurrentIndex(0)
        try:
            self.colocalizeNeighborsLayerComboBox.setCurrentText(neighborsLayerName)
        except:
            self.colocalizeNeighborsLayerComboBox.setCurrentIndex(0)
        self.colocalizeRoisLayerComboBox.currentTextChanged.connect(self.updateRoisColocalizationPlot)
        self.colocalizeNeighborsLayerComboBox.currentTextChanged.connect(self.updateRoisColocalizationPlot)
        self.updateRoisColocalizationPlot()
        
        # layer ROI z-projections
        self.activeRoisLayerComboBox.textActivated.disconnect()
        roisLayerName = self.activeRoisLayerComboBox.currentText()
        self.activeRoisLayerComboBox.clear()
        for layer in self.roisLayers():
            self.activeRoisLayerComboBox.addItem(layer.name)
        try:
            self.activeRoisLayerComboBox.setCurrentText(roisLayerName)
        except:
            self.activeRoisLayerComboBox.setCurrentIndex(0)
        self.activeRoisLayerComboBox.textActivated.connect(lambda x: self.setSelectedRoiIndex())
        self.setSelectedRoisLayer()
    
    def currentFrameIndex(self) -> int:
        try:
            return viewer.dims.current_step[-3]
        except IndexError:
            return 0
    
    # EVENTS
    
    def onLayerInserted(self, event):
        layer = event.value
        index = event.index
        # insert separately managed dict of layer metadata to reflect new layer
        self._layerMetadata.insert(index, {})
        layerMetadata = self._layerMetadata[index]
        if self.isImageStackLayer(layer):
            # create plot to show point z-projections for the new layer
            plot = self.newPlot()
            plot.setLabels(left=layer.name)
            plot_data = plot.plot([], pen=pg.mkPen([98, 143, 176], width=1))
            prevPlots = self.roiZProjectionPlots()
            if len(prevPlots):
                plot.setXLink(prevPlots[0])
            t = self.currentFrameIndex()
            plot_vline = plot.addLine(x=t, pen=pg.mkPen('y', width=1))
            # store non-serializable Qt plot objects in separate layer metadata
            layerMetadata['roi_zprojection_plot'] = plot
            layerMetadata['roi_zprojection_plot_data'] = plot_data
            layerMetadata['roi_zprojection_plot_vline'] = plot_vline
            # insert plot into layout
            plotIndex = self.imageStackLayers().index(layer)
            self.roiZProjectionPlotsLayout.insertWidget(plotIndex, plot)
            self.roiZProjectionPlotsLayout.setStretch(plotIndex, 1)
            self.noVisibleImageStacksLabel.hide()
        elif self.isRoisLayer(layer):
            n_rois = len(layer.data)
            # tags string feature for each ROI
            if not 'tags' in layer.features:
                layer.features['tags'] = [""] * n_rois
        self.updateLayerSelectionBoxes()
        # handle general events for new layer
        layer.events.name.connect(self.onLayerNameChanged)
        layer.events.visible.connect(self.onLayerVisibilityChanged)
    
    def onLayerRemoved(self, event):
        layer = event.value
        index = event.index
        layerMetadata = self._layerMetadata.pop(index)
        if 'roi_zprojection_plot' in layerMetadata:
            # delete plot
            plot = layerMetadata['roi_zprojection_plot']
            self.roiZProjectionPlotsLayout.removeWidget(plot)
            plot.deleteLater()
            # reset plot linkage
            self.linkRoiZProjectionPlots()
        del layerMetadata
        if self.isRoisLayer(layer):
            # remove ROI z-projections for this layer from all image stack layers
            for imlayer in self.imageStackLayers():
                if 'roi_zprojections' in imlayer.metadata:
                    if layer.name in imlayer.metadata['roi_zprojections']:
                        del imlayer.metadata['roi_zprojections'][layer.name]
        elif self.isImageStackLayer(layer):
            visibleImageStackLayers = [layer for layer in self.imageStackLayers() if layer.visible]
            if len(visibleImageStackLayers) == 0:
                self.noVisibleImageStacksLabel.show()
        self.updateLayerSelectionBoxes()
    
    def onLayerMoved(self, event):
        index = event.index
        new_index = event.new_index
        layerMetadata = self._layerMetadata.pop(index)
        self._layerMetadata.insert(new_index, layerMetadata)
        if 'roi_zprojection_plot' in layerMetadata:
            # reposition plot to match layer order
            plot = layerMetadata['roi_zprojection_plot']
            self.roiZProjectionPlotsLayout.removeWidget(plot)
            plotIndex = self.roiZProjectionPlots().index(plot)
            self.roiZProjectionPlotsLayout.insertWidget(plotIndex, plot)
            self.roiZProjectionPlotsLayout.setStretch(plotIndex, 1)
        self.updateLayerSelectionBoxes()
    
    def onLayerNameChanged(self, event):
        index = event.index
        layer = self.viewer.layers[index]
        layerMetadata = self._layerMetadata[index]
        if 'roi_zprojection_plot' in layerMetadata:
            # plot ylabel = layer name
            plot = layerMetadata['roi_zprojection_plot']
            plot.setLabels(left=layer.name)
        if self.isRoisLayer(layer):
            # update ROI z-projection dicts for all image stack layers to reflect new layer name
            for imlayer in self.imageStackLayers():
                if 'roi_zprojections' in imlayer.metadata:
                    # event does not tell us what the previous layer name was, so we have to figure it out
                    zproj_layer_names = imlayer.metadata['roi_zprojections'].keys()
                    roi_layer_names = [layer.name for layer in self.roisLayers()]
                    for name in zproj_layer_names:
                        if name not in roi_layer_names:
                            imlayer.metadata['roi_zprojections'][layer.name] = imlayer.metadata['roi_zprojections'].pop(name)
                            break
        self.updateLayerSelectionBoxes()
    
    def onLayerVisibilityChanged(self, event):
        index = event.index
        layer = self.viewer.layers[index]
        layerMetadata = self._layerMetadata[index]
        if 'roi_zprojection_plot' in layerMetadata:
            # show/hide plot along with layer
            plot = layerMetadata['roi_zprojection_plot']
            plot.setVisible(layer.visible)
        visibleImageStackLayers = [layer for layer in self.imageStackLayers() if layer.visible]
        if len(visibleImageStackLayers) == 0:
            self.noVisibleImageStacksLabel.show()
        else:
            self.noVisibleImageStacksLabel.hide()
    
    def onDimStepChanged(self, event):
        try:
            t = event.value[-3]
        except IndexError:
            t = 0
        for layerMetadata in self._layerMetadata:
            # update frame vline in point z-projection plots
            if 'roi_zprojection_plot_vline' in layerMetadata:
                layerMetadata['roi_zprojection_plot_vline'].setValue(t)
    
    def onMouseClickedOrDragged(self, viewer, event):
        # ignore initial mouse press event
        if event.type == 'mouse_press':
            yield
        else:
            return
        
        # if mouse dragged (beyond a tiny bit), ignore subsequent mouse release event
        n_move_events = 0
        while event.type == 'mouse_move':
            n_move_events += 1
            if n_move_events <= 3:
                yield
            else:
                return
        
        # if we get here, then mouse was clicked without dragging (much)
        if event.type == 'mouse_release':
            mousePointInWorld = event.position[-2:]  # (row, col)
            # check if clicked on a visible point or shape
            visibleRoisLayers = [layer for layer in self.roisLayers() if layer.visible]
            if len(visibleRoisLayers):
                activeRoisLayer = self.selectedRoisLayer()
                if activeRoisLayer in visibleRoisLayers:
                    # check active points layer first
                    visibleRoisLayers.remove(activeRoisLayer)
                    visibleRoisLayers.insert(0, activeRoisLayer)
                if self._selectedRoiLayer is not None:
                    # ignore selected ROI layer
                    visibleRoisLayers.remove(self._selectedRoiLayer)
                for roiLayer in visibleRoisLayers:
                    # Find closest ROI to mouse click.
                    # If within ROI, then select the ROI.
                    if len(roiLayer.data) == 0:
                        continue
                    mousePointInRoiLayer = self.transformPoints2dFromWorldToLayer(mousePointInWorld, roiLayer)
                    if self.isPointsLayer(roiLayer):
                        roiCentersInRoiLayer = roiLayer.data[:,-2:]
                        roiSizes = roiLayer.size[:,-2:]
                    elif self.isShapesLayer(roiLayer):
                        shapesData = [data[:,-2:] for data in roiLayer.data]
                        shapeTypes = roiLayer.shape_type
                        roiCentersInRoiLayer = np.array([data.mean(axis=0) for data in shapesData])
                        roiSizes = np.array([data.max(axis=0) - data.min(axis=0) for data in shapesData])
                    squareDists = np.sum((roiCentersInRoiLayer - mousePointInRoiLayer)**2, axis=1)
                    indexes = np.argsort(squareDists)
                    for index in indexes:
                        rowSize, colSize = roiSizes[index]
                        drow, dcol = mousePointInRoiLayer[0] - roiCentersInRoiLayer[index]
                        if self.isPointsLayer(roiLayer) or (shapeTypes[index] == "ellipse"):
                            if drow**2 / (rowSize / 2)**2 + dcol**2 / (colSize / 2)**2 <= 1:
                                self.setSelectedRoiIndex(index, roiLayer)
                                return
                        elif shapeTypes[index] == "rectangle":
                            if (-rowSize / 2 <= drow <= rowSize / 2) and (-colSize / 2 <= dcol <= colSize / 2):
                                self.setSelectedRoiIndex(index, roiLayer)
                                return
            # no point selected
            self.roiIndexSpinBox.clear()
            self.roiTagsEdit.setText("")
            for roiLayer in self.roisLayers():
                roiLayer._selected_data = set()
                roiLayer._highlight_index = []
                roiLayer.events.highlight()
            # z-project clicked location
            # if self.viewer.grid.enabled:
            #     # find layer grid that was clicked in
            #     # and use the grid relative coords as the ROI world position
            #     # TODO
            #     mousePointInGrid = self.UNUSED_transformPoint2dFromWorldToGrid(mousePointInWorld)
            #     self.updateSelectedRoiLayer(roiWorldPoint=mousePointInGrid)
            # else:
            self.updateSelectedRoiLayer(roiWorldPoint=mousePointInWorld)
            self.zprojectRoisInImageLayers(self._selectedRoiLayer)
            row, col = np.round(mousePointInWorld).astype(int)
            self.currentRoiWorldPositionLabel.setText(f"[{row},{col}]")
    
    def onMouseDoubleClicked(self, viewer, event):
        self.viewer.reset_view()
    
    # I/O
    
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
        mdict['layers'] = []
        for layer in self.viewer.layers:
            layerDict = {}
            layerDict['name'] = layer.name
            layerDict['metadata'] = {}
            if self.isImageLayer(layer):
                # image layer
                imageAbsPath = self.getImageLayerAbsFilePath(layer)
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
            elif self.isRoisLayer(layer):
                # ROIs layer
                if self.isPointsLayer(layer):
                    layerDict['points'] = layer.data
                    layerDict['size'] = layer.size
                    layerDict['symbol'] = layer.symbol
                elif self.isShapesLayer(layer):
                    layerDict['shapes'] = layer.data
                    layerDict['shape_type'] = layer.shape_type
                if not layer.features.empty:
                    layerDict['features'] = {}
                    for key in layer.features:
                        if key == 'tags':
                            layer.features['tags'].fillna("", inplace=True)
                            layer.features['tags'].replace("", " ", inplace=True)
                        layerDict['features'][key] = layer.features[key].to_numpy()
            layerDict['affine'] = layer.affine.affine_matrix[-3:,-3:]
            layerDict['opacity'] = layer.opacity
            layerDict['blending'] = layer.blending
            if self.isImageLayer(layer):
                layerDict['contrast_limits'] = layer.contrast_limits
                layerDict['gamma'] = layer.gamma
                layerDict['colormap'] = layer.colormap.name
                layerDict['interpolation2d'] = layer.interpolation2d
            elif self.isRoisLayer(layer):
                layerDict['face_color'] = layer.face_color
                layerDict['edge_color'] = layer.edge_color
                layerDict['edge_width'] = layer.edge_width
                if self.isPointsLayer(layer):
                    layerDict['edge_width_is_relative'] = layer.edge_width_is_relative
            for key in layer.metadata:
                if key in ["image_file_abspath", "image_file_relpath"]:
                    continue
                layerDict['metadata'][key] = layer.metadata[key]
            if len(layerDict['metadata']) == 0:
                del layerDict['metadata']
            mdict['layers'].append(layerDict)
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
                for layerDict in value:
                    layerName = layerDict['name']
                    hasMetadata = 'metadata' in layerDict
                    affine = layerDict['affine']
                    opacity = layerDict['opacity']
                    blending = str(layerDict['blending'])
                    isImageLayer = ('image' in layerDict) or (hasMetadata and ( \
                        ('image_file_relpath' in layerDict['metadata']) \
                        or ('image_file_abspath' in layerDict['metadata']) \
                        or ('image_shape' in layerDict['metadata'])))
                    isPointsLayer = 'points' in layerDict
                    isShapesLayer = 'shapes' in layerDict
                    isRoisLayer = isPointsLayer or isShapesLayer
                    if isRoisLayer:
                        # ROIs layer
                        face_color = layerDict['face_color']
                        edge_color = layerDict['edge_color']
                        edge_width = layerDict['edge_width']
                        if isPointsLayer:
                            points = layerDict['points']
                            size = layerDict['size']
                            symbol = str(layerDict['symbol'])
                            edge_width_is_relative = layerDict['edge_width_is_relative']
                            layer = self.viewer.add_points(points, symbol=symbol, size=size, name=layerName, 
                                affine=affine, opacity=opacity, blending=blending, 
                                face_color=face_color, edge_color=edge_color, edge_width=edge_width, edge_width_is_relative=edge_width_is_relative)
                        elif isShapesLayer:
                            shapes = layerDict['shapes']
                            shape_type = [str(shape_type) for shape_type in layerDict['shape_type']]
                            layer = self.viewer.add_shapes(shapes, shape_type=shape_type, name=layerName, 
                                affine=affine, opacity=opacity, blending=blending, 
                                face_color=face_color, edge_color=edge_color, edge_width=edge_width)
                        n_rois = len(layer.data)
                        features = pd.DataFrame({"tags": [""] * n_rois})
                        if 'features' in layerDict:
                            for key in layerDict['features']:
                                features[key] = layerDict['features'][key]
                                if key == "tags":
                                    features['tags'].replace(" ", "", inplace=True)
                        layer.features = features
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
                            subimage_slice = None
                            if ('metadata' in layerDict) and ('subimage_slice' in layerDict['metadata']):
                                subimage_slice = layerDict['metadata']['subimage_slice']
                                slices = tuple([slice(start, stop, step) for (start, stop, step) in subimage_slice])
                                # if subimage_slice.shape[1] == 2:
                                #     subimage_slice = np.hstack([
                                #         subimage_slice, np.ones([subimage_slice.shape[0],1], dtype=int)
                                #     ])
                                # starts = subimage_slice[:,0]
                                # stops = subimage_slice[:,1]
                                # steps = subimage_slice[:,2]
                            try:
                                image = tifffile.memmap(imageAbsPath)
                                if subimage_slice is not None:
                                    image = image[slices]
                                    # if len(starts) == 1:
                                    #     # frame crop
                                    #     image = image[starts[0]:stops[0]:steps[0]]
                                    # elif len(starts) == 2:
                                    #     # row,col crop
                                    #     image = image[...,starts[-2]:stops[-2]:steps[-2],starts[-1]:stops[-1]:steps[-1]]
                                    # else:
                                    #     # frame,...,row,col crop
                                    #     indexer = tuple([slice(i,j,k) for (i,j,k) in zip(starts,stops,steps)])
                                    #     image = image[indexer]
                                layer = self.viewer.add_image(image, name=layerName, affine=affine, opacity=opacity, blending=blending, 
                                    contrast_limits=contrast_limits, gamma=gamma, colormap=colormap, interpolation2d=interpolation2d)
                                layer.metadata['image_file_abspath'] = imageAbsPath
                                if subimage_slice is not None:
                                    layer.metadata['subimage_slice'] = subimage_slice
                            except:
                                try:
                                    layer = self.viewer.open(path=imageAbsPath, layer_type="image")[0]
                                    layer.metadata['image_file_abspath'] = os.path.abspath(layer.source.path)
                                    if subimage_slice is not None:
                                        layer.data = layer.data[slices]
                                        # if len(starts) == 1:
                                        #     # frame crop
                                        #     layer.data = layer.data[starts[0]:stops[0]:steps[0]]
                                        # elif len(starts) == 2:
                                        #     # row,col crop
                                        #     layer.data = layer.data[...,starts[-2]:stops[-2]:steps[-2],starts[-1]:stops[-1]:steps[-1]]
                                        # else:
                                        #     # frame,...,row,col crop
                                        #     indexer = tuple([slice(i,j,k) for (i,j,k) in zip(starts,stops,steps)])
                                        #     layer.data = layer.data[indexer]
                                        layer.metadata['subimage_slice'] = subimage_slice
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

    def getImageLayerAbsFilePath(self, layer):
        if not self.isImageLayer(layer):
            return None
        imageAbsPath = None
        if layer.source.path is not None:
            imageAbsPath = os.path.abspath(layer.source.path)
        elif 'image_file_abspath' in layer.metadata:
            imageAbsPath = os.path.abspath(layer.metadata['image_file_abspath'])
        return imageAbsPath
    
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
    
    # LAYERS
    
    def imageLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isImageLayer(layer)]
    
    def imageStackLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isImageStackLayer(layer)]
    
    def pointsLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isPointsLayer(layer)]
    
    def shapesLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isShapesLayer(layer)]
    
    def roisLayers(self):
        return [layer for layer in reversed(self.viewer.layers) if self.isRoisLayer(layer)]
    
    def isImageLayer(self, layer) -> bool:
        return type(layer) is napari.layers.image.image.Image
    
    def isImageStackLayer(self, layer) -> bool:
        return type(layer) is napari.layers.image.image.Image and layer.ndim == 3
    
    def isPointsLayer(self, layer) -> bool:
        return type(layer) is napari.layers.points.points.Points
    
    def isShapesLayer(self, layer) -> bool:
        return type(layer) is napari.layers.shapes.shapes.Shapes
    
    def isRoisLayer(self, layer) -> bool:
        return self.isShapesLayer(layer) or self.isPointsLayer(layer)
    
    def deleteLayer(self, layer):
        layerIndex = list(self.viewer.layers).index(layer)
        del self.viewer.layers[layerIndex]
    
    def applyToAllLayers(self, func, *args, **kwargs):
        layers = list(self.viewer.layers)
        for layer in layers:
            func(layer, *args, **kwargs)
    
    def applyToSelectedLayers(self, func, *args, **kwargs):
        layers = list(self.viewer.layers.selection)
        for layer in layers:
            func(layer, *args, **kwargs)
    
    # TRANSFORMS
    
    def layerToWorldTransform3x3(self, layer):
        return layer.affine.inverse.affine_matrix[-3:,-3:]
    
    def worldToLayerTransform3x3(self, layer):
        return layer.affine.affine_matrix[-3:,-3:]
    
    def transformPoints2dFromLayerToWorld(self, points2d, layer):
        points2d = np.array(points2d).reshape([-1, 2])
        if layer.ndim == 2:
            pointsNd = points2d
        else:
            n_points = points2d.shape[0]
            pointsNd = np.zeros([n_points, layer.ndim])
            pointsNd[:,-2:] = points2d
        worldPoints2d = np.zeros(points2d.shape)
        for i, point in enumerate(points2d):
            worldPoints2d[i] = layer.data_to_world(point)[-2:]
        return worldPoints2d

    def transformPoints2dFromWorldToLayer(self, worldPoints2d, layer):
        worldPoints2d = np.array(worldPoints2d).reshape([-1, 2])
        if layer.ndim == 2:
            worldPointsNd = worldPoints2d
        else:
            n_points = worldPoints2d.shape[0]
            worldPointsNd = np.zeros([n_points, layer.ndim])
            worldPointsNd[:,-2:] = worldPoints2d
        points2d = np.zeros(worldPoints2d.shape)
        for i, worldPoint in enumerate(worldPointsNd):
            points2d[i] = layer.world_to_data(worldPoint)[-2:]
        return points2d
    
    def getLayerWorldBoundingBox(self, layer):
        if self.isImageLayer(layer):
            w, h = layer.data.shape[-2:]
            layerPoints = np.array([[0, 0], [w, 0], [w, h], [0, h]])
        elif self.isPointsLayer(layer):
            layerPoints = layer.data[:,-2:]
        elif self.isShapesLayer(layer):
            layerPoints = np.array([data[:,-2:] for data in layer.data])
        else:
            return None
        worldPoints = self.transformPoints2dFromLayerToWorld(layerPoints, layer)
        worldRowLim = worldPoints[:,0].min(), worldPoints[:,0].max()
        worldColLim = worldPoints[:,1].min(), worldPoints[:,1].max()
        return worldRowLim, worldColLim
    
    # TODO: grid bounds not quite right?
    def UNUSED_getLayerWorldGridBoxes(self):
        n_layers = len(self.viewer.layers)
        layerOrigins = np.zeros([n_layers, 2])
        layerRowLims = np.zeros([n_layers, 2])
        layerColLims = np.zeros([n_layers, 2])
        for i, layer in enumerate(self.viewer.layers):
            layerOrigins[i] = self.transformPoints2dFromLayerToWorld((0, 0), layer)
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
    
    def UNUSED_getIndexOfLayerWhoseGridContainsWorldPoint2d(self, worldPoint2d):
        worldPoint2d = np.array(worldPoint2d).reshape([-1])
        layerRowLims, layerColLims = self.UNUSED_getLayerWorldGridBoxes()
        row, col = worldPoint2d
        indexOfLayerContainingWorldPoint = \
            np.where((layerRowLims[:,0] <= row) & (layerRowLims[:,1] > row) \
            & (layerColLims[:,0] <= col) & (layerColLims[:,1] > col))[0][0]
        return indexOfLayerContainingWorldPoint
    
    def UNUSED_transformPoint2dFromWorldToGrid(self, worldPoint2d):
        worldPoint2d = np.array(worldPoint2d).reshape([-1])
        layerRowLims, layerColLims = self.UNUSED_getLayerWorldGridBoxes()
        row, col = worldPoint2d
        indexOfLayerContainingWorldPoint = \
            np.where((layerRowLims[:,0] <= row) & (layerRowLims[:,1] > row) \
            & (layerColLims[:,0] <= col) & (layerColLims[:,1] > col))[0][0]
        gridRowOrigin = layerRowLims[indexOfLayerContainingWorldPoint,0]
        gridColOrigin = layerColLims[indexOfLayerContainingWorldPoint,0]
        return np.array([[row - gridRowOrigin, col - gridColOrigin]])
    
    # ROIS
    
    def selectedRoisLayer(self):
        if len(self.roisLayers()) == 0:
            return None
        if self.activeRoisLayerComboBox.count() == 0:
            return None
        layerName = self.activeRoisLayerComboBox.currentText()
        try:
            layer = self.viewer.layers[layerName]
        except KeyError:
            return None
        return layer
    
    def setSelectedRoisLayer(self, roisLayer=None):
        # no highlighted points in any other layer
        for layer in self.roisLayers():
            layer._selected_data = set()
            layer._highlight_index = []
            layer.events.highlight()
        if self.isRoisLayer(roisLayer):
            self.activeRoisLayerComboBox.setCurrentText(roisLayer.name)
            n_rois = self.numRois(roisLayer)
            n_filteredRois = self.numFilteredRois(roisLayer)
            self.numSelectedRoisLabel.setText(f"{n_filteredRois}/{n_rois} ROIs")
        else:
            self.numSelectedRoisLabel.setText("")
    
    def selectedRoiIndex(self):
        if self.selectedRoisLayer() is None:
            return None
        if self.roiIndexSpinBox.cleanText() == "":
            return None
        return self.roiIndexSpinBox.value()
    
    def setSelectedRoiIndex(self, roiIndex=None, roisLayer=None):
        if roisLayer is None:
            roisLayer = self.selectedRoisLayer()
        if self.isRoisLayer(roisLayer):
            self.setSelectedRoisLayer(roisLayer)
        else:
            roisLayer = None
        if roisLayer is not None:
            n_rois = len(roisLayer.data)
            if roiIndex is None:
                roiIndex = self.selectedRoiIndex()
            if roiIndex is not None:
                roiIndex = min(max(0, roiIndex), n_rois - 1)
            if self.roiTagFilterCheckBox.isChecked():
                roiIndex = self.filterRoiIndex(roisLayer, roiIndex)
        
        if (roisLayer is None) or (roiIndex is None):
            # deselect all ROIs
            self.roiIndexSpinBox.clear()
            self.currentRoiWorldPositionLabel.setText("")
            self.roiTagsEdit.setText("")
            self._roiIndex = None

            # no highlighted ROIs
            if roisLayer is not None:
                roisLayer._selected_data = set()
                roisLayer._highlight_index = []
                roisLayer.events.highlight()

            # clear all ROI z-projection plots
            self.clearRoiZProjectionPlots()

            # done
            return

        # select ROI
        self.roiIndexSpinBox.setValue(roiIndex)
        try:
            tags = roisLayer.features['tags'][roiIndex]
            try:
                tags = tags.strip()
                self.roiTagsEdit.setText(tags)
            except AttributeError:
                roisLayer.features['tags'][roiIndex] = ""
                self.roiTagsEdit.setText("")
        except (KeyError, IndexError):
            self.roiTagsEdit.setText("")
        self._roiIndex = roiIndex
        
        # highlight selected ROI
        roisLayer._selected_data = set([roiIndex])
        roisLayer._highlight_index = [roiIndex]
        roisLayer.events.highlight()
        
        # update selected ROI layer
        self.updateSelectedRoiLayer(roisLayer, roiIndex)

        # z-project image stacks for selected ROI
        self.zprojectRoisInImageLayers(roisLayer, roiIndex)
    
    def updateSelectedRoiTags(self):
        layer = self.selectedRoisLayer()
        if layer is not None:
            index = self.selectedRoiIndex()
            if index is not None:
                tags = self.roiTagsEdit.text().strip()
                if not 'tags' in layer.features:
                    n_rois = len(layer.data)
                    layer.features['tags'] = [""] * n_rois
                layer.features['tags'][index] = tags
    
    def filterRoiIndex(self, roisLayer, roiIndex, tagFilter=None):
        if (roisLayer is None) or not self.isRoisLayer(roisLayer):
            return None
        if tagFilter is None:
            tagFilter = self.roiTagFilterEdit.text()
        roisLayer.features['tags'].fillna("", inplace=True)
        n_rois = len(roisLayer.data)
        roiIndex = min(max(0, roiIndex), n_rois - 1)
        if tagFilter.strip() == "":
            return roiIndex
        tags = roisLayer.features['tags'][roiIndex]
        if self.checkIfTagsMatchFilter(tags, tagFilter):
            return roiIndex
        if (roiIndex + 1 < n_rois) and ((self._roiIndex is None) or (self._roiIndex <= roiIndex)):
            # find next match
            for i in range(roiIndex + 1, n_rois):
                tags = roisLayer.features['tags'][i]
                if self.checkIfTagsMatchFilter(tags, tagFilter):
                    return i
        if (roiIndex > 0) and ((self._roiIndex is None) or (self._roiIndex >= roiIndex)):
            # find previous match
            for i in reversed(range(roiIndex)):
                tags = roisLayer.features['tags'][i]
                if self.checkIfTagsMatchFilter(tags, tagFilter):
                    return i
        # if didn't find anything, search ahead and behind irrespective of self._roiIndex
        if roiIndex + 1 < n_rois:
            # find next match
            for i in range(roiIndex + 1, n_rois):
                tags = roisLayer.features['tags'][i]
                if self.checkIfTagsMatchFilter(tags, tagFilter):
                    return i
        if roiIndex > 0:
            # find previous match
            for i in reversed(range(roiIndex)):
                tags = roisLayer.features['tags'][i]
                if self.checkIfTagsMatchFilter(tags, tagFilter):
                    return i
        return None
    
    def checkIfTagsMatchFilter(self, tags, filter):
        tags = [tag.strip() for tag in tags.split(",")]
        or_tags = [tag.strip() for tag in filter.split(",")]
        for or_tag in or_tags:
            and_tags = [tag.strip() for tag in or_tag.split("&")]
            and_matches = [tag in tags for tag in and_tags]
            if np.all(and_matches):
                return True
        return False
    
    def numRois(self, roisLayer=None):
        if roisLayer is None:
            roisLayer = self.selectedRoisLayer()
        if not self.isRoisLayer(roisLayer):
            return 0
        return len(roisLayer.data)
    
    def numFilteredRois(self, roisLayer=None, tagFilter=None):
        if roisLayer is None:
            roisLayer = self.selectedRoisLayer()
        if not self.isRoisLayer(roisLayer):
            return 0
        n_rois = self.numRois(roisLayer)
        if not self.roiTagFilterCheckBox.isChecked():
            return n_rois
        if tagFilter is None:
            tagFilter = self.roiTagFilterEdit.text()
        if tagFilter.strip() == "":
            return n_rois
        n_filteredRois = 0
        for i in range(n_rois):
            tags = roisLayer.features['tags'][i]
            if self.checkIfTagsMatchFilter(tags, tagFilter):
                n_filteredRois += 1
        return n_filteredRois
    
    def addRoisLayer(self, center, size=None, shape_type=None, affine=np.eye(3), features=None, name="ROIs", 
    edge_width=None, edge_color=None, face_color=None, blending="translucent_no_depth", opacity=1, isSelectedRoisLayer=False):
        if shape_type is None:
            shape_type = self.roiShapeComboBox.currentText()
        if size is None:
            size = self.roiSizeSpinBox.value()
        if edge_width is None:
            if isSelectedRoisLayer:
                edge_width = self.selectedRoiEdgeWidthSpinBox.value()
            else:
                edge_width = self.roiEdgeWidthSpinBox.value()
        if edge_color is None:
            if isSelectedRoisLayer:
                edge_color = str2rgba(self.selectedRoiEdgeColorEdit.text())
            else:
                edge_color = str2rgba(self.roiEdgeColorEdit.text())
        if face_color is None:
            if isSelectedRoisLayer:
                face_color = [0]*4
            else:
                face_color = str2rgba(self.roiFaceColorEdit.text())
        if features is None:
            n_rois = len(center)
            features = pd.DataFrame({"tags": [""] * n_rois})
        if isSelectedRoisLayer and (self._selectedRoiLayer is not None):
            self.deleteLayer(self._selectedRoiLayer)
            self._selectedRoiLayer = None
        if shape_type == "point":
            roisLayer = self.viewer.add_points(center, size=size, affine=affine, features=features, name=name, 
                edge_width=edge_width, edge_width_is_relative=False, 
                edge_color=edge_color, face_color=face_color, blending=blending, opacity=opacity)
        else:
            data = self.getRoisShapeData(center, size)
            roisLayer = self.viewer.add_shapes(data, shape_type=shape_type, affine=affine, features=features, name=name, 
                edge_width=edge_width, edge_color=edge_color, face_color=face_color, blending=blending, opacity=opacity)
        if isSelectedRoisLayer:
            self._selectedRoiLayer = roisLayer
        return roisLayer
    
    def updateRoisLayer(self, roisLayer, shape_type=None, size=None, edge_width=None, edge_color=None, face_color=None):
        if (roisLayer is None) or not self.isRoisLayer(roisLayer):
            return
        if shape_type is None:
            shape_type = self.roiShapeComboBox.currentText()
        if size is None:
            size = self.roiSizeSpinBox.value()
        if edge_width is None:
            if roisLayer is self._selectedRoiLayer:
                edge_width = self.selectedRoiEdgeWidthSpinBox.value()
            else:
                edge_width = self.roiEdgeWidthSpinBox.value()
        if edge_color is None:
            if roisLayer is self._selectedRoiLayer:
                edge_color = str2rgba(self.selectedRoiEdgeColorEdit.text())
            else:
                edge_color = str2rgba(self.roiEdgeColorEdit.text())
        if face_color is None:
            if roisLayer is self._selectedRoiLayer:
                face_color = [0]*4
            else:
                face_color = str2rgba(self.roiFaceColorEdit.text())
        if shape_type == "point" and self.isShapesLayer(roisLayer):
            roisLayer = self.convertShapesLayerToPointsLayer(roisLayer)
        elif shape_type != "point" and self.isPointsLayer(roisLayer):
            roisLayer = self.convertPointsLayerToShapesLayer(roisLayer)
        elif shape_type == "point":
            roisLayer.size = size
            roisLayer.edge_width_is_relative = False
        else:
            roiCenters = self.getRoiCenters2d(roisLayer)
            roiData = self.getRoisShapeData(roiCenters, size)
            roisLayer.data = roiData
            roisLayer.shape_type = [shape_type] * len(roiData)
        roisLayer.edge_width = edge_width
        roisLayer.edge_color = edge_color
        roisLayer.face_color = face_color
    
    def updateSelectedRoiLayer(self, roisLayer=None, roiIndex=None, roiWorldPoint=None):
        if roiWorldPoint is not None:
            # default ROI at world point
            roiShapeType = self.roiShapeComboBox.currentText()
            roiSize = self.roiSizeSpinBox.value()
            if roiShapeType == "point":
                roiData = roiWorldPoint
            else:
                roiData = roiWorldPoint + np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * (roiSize / 2)
            roiLayerTransform = np.eye(3)
        elif (roisLayer is not None) and (roiIndex is not None):
            # selected ROI
            if self.isPointsLayer(roisLayer):
                roiShapeType = "point"
                roiData = roisLayer.data[roiIndex,-2:]
                roiSize = roisLayer.size[roiIndex,-2:]
            elif self.isShapesLayer(roisLayer):
                roiShapeType = roisLayer.shape_type[roiIndex]
                roiData = roisLayer.data[roiIndex][:,-2:]
            else:
                return
            roiLayerTransform = roisLayer.affine
        else:
            # delete selected ROI layer
            self.deleteLayer(self._selectedRoiLayer)
            self._selectedRoiLayer = None
            return
        roiEdgeWidth = self.selectedRoiEdgeWidthSpinBox.value()
        roiEdgeColor = str2rgba(self.selectedRoiEdgeColorEdit.text())
        roiFaceColor = [0]*4
        
        # delete selected ROI layer if it is the wrong layer type
        if self._selectedRoiLayer is not None:
            if ((roiShapeType == "point") and not self.isPointsLayer(self._selectedRoiLayer)) \
                or ((roiShapeType != "point") and self.isPointsLayer(self._selectedRoiLayer)):
                self.deleteLayer(self._selectedRoiLayer)
                self._selectedRoiLayer = None
        
        if self._selectedRoiLayer is None:
            # add selected ROI layer
            if roiShapeType == "point":
                self._selectedRoiLayer = self.viewer.add_points(roiData, name="selected ROI", 
                    size=roiSize, edge_width=roiEdgeWidth, edge_width_is_relative=False,
                    edge_color=roiEdgeColor, face_color=roiFaceColor, 
                    affine=roiLayerTransform)
            else:
                self._selectedRoiLayer = self.viewer.add_shapes([roiData], name="selected ROI", 
                    shape_type=[roiShapeType], edge_width=roiEdgeWidth, 
                    edge_color=roiEdgeColor, face_color=roiFaceColor, 
                    affine=roiLayerTransform)
        else:
            # edit selected ROI layer
            if roiShapeType == "point":
                self._selectedRoiLayer.data = roiData
                self._selectedRoiLayer.size = roiSize
            else:
                self._selectedRoiLayer.data = [roiData]
                self._selectedRoiLayer.shape_type = [roiShapeType]
            self._selectedRoiLayer.edge_width = roiEdgeWidth
            self._selectedRoiLayer.edge_color = roiEdgeColor
            self._selectedRoiLayer.face_color = roiFaceColor
            self._selectedRoiLayer.affine = roiLayerTransform
    
    def getRoiCenters2d(self, roisLayer):
        if self.isPointsLayer(roisLayer):
            return roisLayer.data[:,-2:]
        elif self.isShapesLayer(roisLayer):
            return np.array([data[:,-2:].mean(axis=0) for data in roisLayer.data])
    
    def getRoiSizes2d(self, roisLayer):
        if self.isPointsLayer(roisLayer):
            return roisLayer.size[:,-2:]
        elif self.isShapesLayer(roisLayer):
            return np.array([data[:,-2:].max(axis=0) - data[:,-2:].min(axis=0) for data in roisLayer.data])
    
    def getRoiShapeTypes(self, roisLayer):
        if self.isPointsLayer(roisLayer):
            return ["point"] * len(roisLayer.data)
        elif self.isShapesLayer(roisLayer):
            return roisLayer.shape_type
    
    def getRoisShapeData(self, roiCenters, roiSizes):
        roiCenters = np.array(roiCenters).reshape([-1,2])
        if isinstance(roiSizes, np.ndarray) and (roiCenters.shape == roiSizes.shape):
            _roiSizes = roiSizes
        else:
            _roiSizes = np.zeros(roiCenters.shape)
            _roiSizes[:] = roiSizes
        normBox = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) / 2
        return [center + normBox * size for (center, size) in zip(roiCenters, _roiSizes)]
    
    def convertPointsLayerToShapesLayer(self, pointsLayer, roiShapeType=None):
        roiCenters = self.getRoiCenters2d(pointsLayer)
        roiSizes = self.getRoiSizes2d(pointsLayer)
        roiShapeData = self.getRoisShapeData(roiCenters, roiSizes)
        if roiShapeType is not None:
            roiShapeTypes = [roiShapeType] * len(roiShapeData)
        else:
            roiShapeTypes = ["ellipse"] * len(roiShapeData)
        roiLayerName = pointsLayer.name
        shapesLayer = self.viewer.add_shapes(roiShapeData, shape_type=roiShapeTypes, 
            name=pointsLayer.name, affine=pointsLayer.affine, features=pointsLayer.features, 
            edge_width=pointsLayer.edge_width, 
            edge_color=pointsLayer.edge_color, face_color=pointsLayer.face_color, 
            blending=pointsLayer.blending, opacity=pointsLayer.opacity)
        # move shapes layer to points layer
        shapesLayerIndex = list(self.viewer.layers).index(shapesLayer)
        pointsLayerIndex = list(self.viewer.layers).index(pointsLayer)
        self.viewer.layers.move(shapesLayerIndex, pointsLayerIndex)
        # delete points layer
        pointsLayerIndex = list(self.viewer.layers).index(pointsLayer)
        del self.viewer.layers[pointsLayerIndex]
        # update shapes layer name
        shapesLayer.name = roiLayerName
        return shapesLayer
    
    def convertShapesLayerToPointsLayer(self, shapesLayer):
        roiCenters = self.getRoiCenters2d(shapesLayer)
        roiSizes = self.getRoiSizes2d(shapesLayer)
        roiLayerName = shapesLayer.name
        pointsLayer = self.viewer.add_points(roiCenters, size=roiSizes, 
            name=shapesLayer.name, affine=shapesLayer.affine, features=shapesLayer.features, 
            edge_width=shapesLayer.edge_width, edge_width_is_relative=False, 
            edge_color=shapesLayer.edge_color, face_color=shapesLayer.face_color, 
            blending=shapesLayer.blending, opacity=shapesLayer.opacity)
        # move points layer to shapes layer
        pointsLayerIndex = list(self.viewer.layers).index(shapesLayer)
        shapesLayerIndex = list(self.viewer.layers).index(shapesLayer)
        self.viewer.layers.move(pointsLayerIndex, shapesLayerIndex)
        # delete shapes layer
        shapesLayerIndex = list(self.viewer.layers).index(shapesLayer)
        del self.viewer.layers[shapesLayerIndex]
        # update points layer name
        pointsLayer.name = roiLayerName
        return pointsLayer
    
    def findPeakRoisInImageLayer(self, layer, minPeakHeight=None, minPeakSeparation=None):
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
        roiShapeType = self.roiShapeComboBox.currentText()
        roiSize = self.roiSizeSpinBox.value()
        roiEdgeWidth = self.roiEdgeWidthSpinBox.value()
        roiEdgeColor = str2rgba(self.roiEdgeColorEdit.text())
        roiFaceColor = str2rgba(self.roiFaceColorEdit.text())
        if roiShapeType == "point":
            return self.viewer.add_points(points, name=name, affine=tform, features=features, 
                size=roiSize, edge_width=roiEdgeWidth, edge_color=roiEdgeColor, edge_width_is_relative=False, 
                face_color=roiFaceColor, blending='translucent_no_depth', opacity=1)
        else:
            shapes = [point + np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * (roiSize / 2) for point in points]
            shapeTypes = [roiShapeType] * n_points
            return self.viewer.add_shapes(shapes, shape_type=shapeTypes, name=name, affine=tform, features=features, 
                edge_width=roiEdgeWidth, edge_color=roiEdgeColor, face_color=roiFaceColor, blending='translucent_no_depth', opacity=1)
    
    def zprojectRoisInImageLayers(self, roisLayer, roiIndices=None, imageLayers=None):
        if not self.isRoisLayer(roisLayer):
            return
        roiCenters = self.getRoiCenters2d(roisLayer)
        roiSizes = self.getRoiSizes2d(roisLayer)
        roiShapeTypes = self.getRoiShapeTypes(roisLayer)
        if roiIndices is not None:
            roiCenters = roiCenters[roiIndices,:]
            roiSizes = roiSizes[roiIndices,:]
            roiShapeTypes = roiShapeTypes[roiIndices]
        n_rois = len(roiCenters)
        roiCentersInWorld = self.transformPoints2dFromLayerToWorld(roiCenters, roisLayer)
        # if self.viewer.grid.enabled:
        #     # get world centers relative to the layer's grid origin
        #     layerGridRowLims, layerGridColLims = self.getLayerWorldGridBoxes()
        #     roisLayerIndex = list(self.viewer.layers).index(roisLayer)
        #     roisLayerGridOrigin = np.array([layerGridRowLims[roisLayerIndex,0], layerGridColLims[roisLayerIndex,0]]).reshape([-1,2])
        #     roiCentersInGrid = roiCentersInWorld - roisLayerGridOrigin
        if imageLayers is None:
            imageLayers = self.imageStackLayers()
        for imageLayer in imageLayers:
            if not self.isImageLayer(imageLayer):
                continue
            # if self.viewer.grid.enabled:
            #     imageLayerIndex = list(self.viewer.layers).index(imageLayer)
            #     imageLayerGridOrigin = np.array([layerGridRowLims[imageLayerIndex,0], layerGridColLims[imageLayerIndex,0]]).reshape([-1,2])
            #     roiCentersInImageLayer = self.transformPoints2dFromWorldToLayer(imageLayerGridOrigin + roiCentersInGrid, imageLayer)
            # else:
            roiCentersInImageLayer = self.transformPoints2dFromWorldToLayer(roiCentersInWorld, imageLayer)
            roiZProjections = np.zeros([n_rois] + list(imageLayer.data.shape[:-2]))
            for i in range(n_rois):
                try:
                    roiMask = getRoiMask(roiSizes[i], roiShapeTypes[i])
                    roiZProjections[i] = zprojectRoiInImage(imageLayer.data, roiCentersInImageLayer[i], roiMask)
                except:
                    # ROI not in image
                    pass
            # show z-projection of first ROI
            imageLayerIndex = list(self.viewer.layers).index(imageLayer)
            self._layerMetadata[imageLayerIndex]['roi_zprojection_plot_data'].setData(roiZProjections[0])
            # indicate world position of displayed ROI z-projection
            row, col = np.round(roiCentersInWorld[0]).astype(int)
            self.currentRoiWorldPositionLabel.setText(f"[{row},{col}]")
            # store z-projections in image layer metadata (only if all ROIs included ==> roiIndices is None)
            if roiIndices is None:
                if 'roi_zprojections' not in imageLayer.metadata:
                    imageLayer.metadata['roi_zprojections'] = {}
                imageLayer.metadata['roi_zprojections'][roisLayer.name] = roiZProjections
    
    # LAYER REGISTRATION
    
    def registerLayers(self, fixedLayer=None, movingLayer=None, transformType=None):
        # TODO: register shapes/points layers
        if fixedLayer is None:
            fixedLayerName = self.fixedLayerComboBox.currentText()
            fixedLayer = self.viewer.layers[fixedLayerName]
        if movingLayer is None:
            movingLayerName = self.movingLayerComboBox.currentText()
            movingLayer = self.viewer.layers[movingLayerName]
        if transformType is None:
            transformType = self.layerRegistrationTransformTypeComboBox.currentText()
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
            msg.setText("Only image-image or points-points layer registration implemented.")
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
        if AffineRegistration is None:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Points registration requires pycpd.")
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
        fixedPoints = fixedLayer.data
        movingPoints = movingLayer.data
        reg = AffineRegistration(X=fixedPoints, Y=movingPoints)
        registeredPoints, (affine, translation) = reg.register()
        tform = np.eye(3)
        tform[:2,:2] = affine
        tform[:2,-1] = translation
        # apply net world transform to moving points
        movingLayer.affine = tform @ fixedLayer.affine.affine_matrix[-3:,-3:]
    
    def copyLayerTransform(self, layer=None):
        if layer is None:
            # first selected layer
            layers = list(self.viewer.layers.selection)
            if len(layers) == 0:
                return
            layer = layers[0]
        self._layerTransformCopy = self.layerToWorldTransform3x3(layer)
    
    def pasteCopiedLayerTransform(self, layer):
        layer.affine = self._layerTransformCopy
    
    def clearLayerTransform(self, layer):
        layer.affine = np.eye(3)
    
    # IMAGE PROCESSING
    
    def zprojectImageLayer(self, layer, method=None, frameSlice=None):
        if not self.isImageLayer(layer) or (layer.ndim <= 2):
            return
        if method is None:
            method = self.zprojectImageOperationComboBox.currentText()
        methods = {"max": np.max, "min": np.min, "std": np.std, "sum": np.sum, "mean": np.mean, "median": np.median}
        func = methods[method]
        if frameSlice is None:
            frameSlice = self.zprojectImageFrameSliceEdit.text().strip()
        if isinstance(frameSlice, str):
            frameSlice = str2slice(frameSlice)
        # n_frames = layer.data.shape[0]
        # if frames is None:
        #     framesText = self.zprojectImageFramesEdit.text().strip()
        #     if framesText == "":
        #         frames = np.arange(n_frames)
        #     else:
        #         slice = framesText.split(':')
        #         try:
        #             start = int(slice[0])
        #         except (IndexError, ValueError):
        #             start = 0
        #         try:
        #             stop = int(slice[1])
        #         except (IndexError, ValueError):
        #             stop = n_frames
        #         try:
        #             step = int(slice[2])
        #         except (IndexError, ValueError):
        #             step = 1
        #         frames = np.arange(start, stop, step)
        projected = func(layer.data[frameSlice], axis=0)
        name = layer.name + f" {method}-proj"
        tform = self.worldToLayerTransform3x3(layer)
        return self.viewer.add_image(projected, name=name, affine=tform, 
            blending=layer.blending, colormap=layer.colormap, opacity=layer.opacity)
    
    def splitImageLayer(self, layer, regions=None):
        if not self.isImageLayer(layer):
            return
        bounds = np.hstack([
            np.zeros([len(layer.data.shape),1], dtype=int), 
            np.array(layer.data.shape, dtype=int).reshape([-1,1])
            ])
        imageAbsPath = self.getImageLayerAbsFilePath(layer)
        if imageAbsPath is not None:
            if 'subimage_slice' in layer.metadata:
                subimage_slice = layer.metadata['subimage_slice'].copy()
            else:
                subimage_slice = bounds.copy()
        tform = self.worldToLayerTransform3x3(layer)
        if regions is None:
            regions = self.splitImageRegionsComboBox.currentText()
        if regions == "Top/Bottom":
            n_rows = layer.data.shape[-2]
            row = int(n_rows / 2)
            top = layer.data[...,:row,:]
            bottom = layer.data[...,-row:,:]
            name = layer.name + " bottom"
            bottom_layer = self.viewer.add_image(bottom, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                bottom_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                rowstart, rowstop = startstop[-2]
                startstop[-2] = [rowstop - row, rowstop]
                bottom_layer.metadata['subimage_slice'] = startstop
            name = layer.name + " top"
            top_layer = self.viewer.add_image(top, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                top_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                rowstart, rowstop = startstop[-2]
                startstop[-2] = [rowstart, rowstart + row]
                top_layer.metadata['subimage_slice'] = startstop
            return top_layer, bottom_layer
        elif regions == "Left/Right":
            n_cols = layer.data.shape[-1]
            col = int(n_cols / 2)
            left = layer.data[...,:,:col]
            right = layer.data[...,:,-col:]
            name = layer.name + " right"
            right_layer = self.viewer.add_image(right, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                right_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                colstart, colstop = startstop[-1]
                startstop[-1] = [colstop - col, colstop]
                right_layer.metadata['subimage_slice'] = startstop
            name = layer.name + " left"
            left_layer = self.viewer.add_image(left, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                left_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                colstart, colstop = startstop[-1]
                startstop[-1] = [colstart, colstart + col]
                left_layer.metadata['subimage_slice'] = startstop
            return left_layer, right_layer
        elif regions == "Quad":
            n_rows = layer.data.shape[-2]
            n_cols = layer.data.shape[-1]
            row = int(n_rows / 2)
            col = int(n_cols / 2)
            topleft = layer.data[...,:row,:col]
            topright = layer.data[...,:row,-col:]
            bottomleft = layer.data[...,-row:,:col]
            bottomright = layer.data[...,-row:,-col:]
            name = layer.name + " bottomright"
            bottomright_layer = self.viewer.add_image(bottomright, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                bottomright_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                rowstart, rowstop = startstop[-2]
                colstart, colstop = startstop[-1]
                startstop[-2] = [rowstop - row, rowstop]
                startstop[-1] = [colstop - col, colstop]
                bottomright_layer.metadata['subimage_slice'] = startstop
            name = layer.name + " bottomleft"
            bottomleft_layer = self.viewer.add_image(bottomleft, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                bottomleft_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                rowstart, rowstop = startstop[-2]
                colstart, colstop = startstop[-1]
                startstop[-2] = [rowstop - row, rowstop]
                startstop[-1] = [colstart, colstart + col]
                bottomleft_layer.metadata['subimage_slice'] = startstop
            name = layer.name + " topright"
            topright_layer = self.viewer.add_image(topright, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                topright_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                rowstart, rowstop = startstop[-2]
                colstart, colstop = startstop[-1]
                startstop[-2] = [rowstart, rowstart + row]
                startstop[-1] = [colstop - col, colstop]
                topright_layer.metadata['subimage_slice'] = startstop
            name = layer.name + " topleft"
            topleft_layer = self.viewer.add_image(topleft, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)
            if imageAbsPath is not None:
                topleft_layer.metadata['image_file_abspath'] = imageAbsPath
                startstop = subimage_slice.copy()
                rowstart, rowstop = startstop[-2]
                colstart, colstop = startstop[-1]
                startstop[-2] = [rowstart, rowstart + row]
                startstop[-1] = [colstart, colstart + col]
                topleft_layer.metadata['subimage_slice'] = startstop
            return topleft_layer, topright_layer, bottomleft_layer, bottomright_layer
    
    def sliceImageLayer(self, layer, slices=None):
        if not self.isImageLayer(layer):
            return
        if slices is None:
            slices = self.sliceImageEdit.text().strip()
        if isinstance(slices, str):
            slices = str2slice(slices)
        # bounds = np.hstack([
        #     np.zeros([len(layer.data.shape),1], dtype=int), 
        #     np.array(layer.data.shape, dtype=int).reshape([-1,1])
        #     ])
        # if cropSlice is None:
        #     cropSlice = self.sliceImageEdit.text().strip()
        #     if cropSlice == "":
        #         return
        # if type(cropSlice) is str:
        #     cropSlice = [[s.strip() for s in lim.split(':')] for lim in cropSlice.strip().split(',')]
        #     if len(cropSlice) == 1:
        #         # frame crop
        #         cropSliceDims = [0]
        #     elif len(cropSlice) == 2:
        #         # row,col crop
        #         cropSliceDims = [-2, -1]
        #     else:
        #         # frame,...,row,col crop
        #         cropSliceDims = list(range(len(layer.data.shape)))
        #     for i in range(len(cropSlice)):
        #         for j in range(len(cropSlice[i])):
        #             if cropSlice[i][j] == "":
        #                 if j < 2:
        #                     cropSlice[i][j] = bounds[cropSliceDims[i],j]
        #                 else:
        #                     cropSlice[i][j] = 1
        #             else:
        #                 cropSlice[i][j] = int(cropSlice[i][j])
        #         if len(cropSlice[i]) == 2:
        #             cropSlice[i].append(1)
        # # cropSlice = [starts column, stops column, steps column]
        # # e.g., startstop = [[frame start, frame stop, frame step], [row start, row stop, row step], [col start, col stop, col step]]
        # cropSlice = np.array(cropSlice, dtype=int)
        # if cropSlice.size % 2 == 0:
        #     cropSlice = cropSlice.reshape([-1,2])
        #     cropSlice = np.hstack([
        #         cropSlice, np.ones([cropSlice.shape[0],1], dtype=int)
        #     ])
        # elif cropSlice.size % 3 == 0:
        #     cropSlice = cropSlice.reshape([-1,3])
        # starts = cropSlice[:,0]
        # stops = cropSlice[:,1]
        # steps = cropSlice[:,2]
        # if len(starts) == 1:
        #     # frame crop
        #     crop = layer.data[starts[0]:stops[0]:steps[0]]
        # elif len(starts) == 2:
        #     # row,col crop
        #     crop = layer.data[...,starts[-2]:stops[-2]:steps[-2],starts[-1]:stops[-1]:steps[-1]]
        # else:
        #     # frame,...,row,col crop
        #     indexer = tuple([slice(i,j,k) for (i,j,k) in zip(starts,stops,steps)])
        #     crop = layer.data[indexer]
        imageSlice = layer.data[slices]
        name = layer.name + " slice"
        tform = self.worldToLayerTransform3x3(layer)
        imageSliceLayer = self.viewer.add_image(imageSlice, name=name, affine=tform, 
            blending=layer.blending, colormap=layer.colormap, opacity=layer.opacity)
        imageAbsPath = self.getImageLayerAbsFilePath(layer)
        if imageAbsPath is None:
            return imageSliceLayer
        imageSliceLayer.metadata['image_file_abspath'] = imageAbsPath
        if 'subimage_slice' in layer.metadata:
            subimage_slice = layer.metadata['subimage_slice'].copy().astype(int)
        else:
            subimage_slice = np.zeros([0,3], dtype=int)
        for i in range(len(slices)):
            start, stop, step = slices[i].start, slices[i].stop, slices[i].step
            if subimage_slice.shape[0] > i:
                prevStart, prevStop, prevStep = subimage_slice[i]
                newStart = prevStart if start is None else prevStart + start
                newStop = prevStop if stop is None else prevStart + stop
                newStep = prevStep if step is None else prevStep * step
                subimage_slice[i] = newStart, newStop, newStep
            else:
                start = 0 if start is None else (start if start >= 0 else layer.data.shape[i] + start)
                stop = layer.data.shape[i] if stop is None else (stop if stop >= 0 else layer.data.shape[i] + stop)
                step = 1 if step is None else step
                subimage_slice = np.vstack([
                    subimage_slice,
                    np.array([[start, stop, step]], dtype=int)
                ])
        #     if subimage_slice.shape[1] == 2:
        #         subimage_slice = np.hstack([
        #             subimage_slice, np.ones([subimage_slice.shape[0],1], dtype=int)
        #         ])
        # else:
        #     subimage_slice = np.hstack([
        #         bounds, np.ones([bounds.shape[0],1], dtype=int)
        #     ])
        # if len(starts) == 1:
        #     # frame crop
        #     cropSlice[0,:2] += subimage_slice[0,0]
        #     subimage_slice[0] = cropSlice
        # elif len(starts) == 2:
        #     # row,col crop
        #     cropSlice[-2:,:2] += subimage_slice[-2:,0].reshape([2,1])
        #     subimage_slice[-2:] = cropSlice
        # else:
        #     # frame,...,row,col crop
        #     cropSlice[:,:2] += subimage_slice[:,0].reshape([-1,1])
        #     subimage_slice = cropSlice
        imageSliceLayer.metadata['subimage_slice'] = subimage_slice
        return imageSliceLayer
    
    def gaussianFilterImageLayer(self, layer, sigma=None):
        if not self.isImageLayer(layer):
            return
        if sigma is None:
            sigma = self.gaussianFilterSigmaSpinBox.value()
        if self.isImageStackLayer(layer):
            # default is to not blur together images in stack (i.e., non-zero sigma in frame dimension)
            if type(sigma) is float:
                sigma = [0]*(layer.ndim - 2) + [sigma, sigma]
            else:
                sigma = [0]*(layer.ndim - len(sigma)) + list(sigma)
            # elif len(sigma) == 1:
            #     sigma = (0, sigma[0], sigma[0])
            # elif len(sigma) == 2:
            #     sigma = (0, sigma[0], sigma[1])
        filtered = filters.gaussian(layer.data, sigma=sigma, preserve_range=True)
        name = layer.name + " gauss-filt"
        tform = self.worldToLayerTransform3x3(layer)
        return self.viewer.add_image(filtered, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)

    # TODO: handle 4d and higher dimension images (apply tophat to last two row,col dimensions)
    def tophatFilterImageLayer(self, layer, diskRadius=None):
        if not self.isImageLayer(layer):
            return
        if diskRadius is None:
            diskRadius = self.tophatFilterDiskRadiusSpinBox.value()
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
        return self.viewer.add_image(filtered, name=name, affine=tform, blending=layer.blending, colormap=layer.colormap)

    # ROI COLOCALIZATION
    
    def visualizeColocalizationOfRoisLayers(self, roisLayer, neighborsLayer, bins=30):
        if neighborsLayer is roisLayer:
            neighborsLayer = None
        if roisLayer is not None:
            roiCenters = self.getRoiCenters2d(roisLayer)
            roiCentersInWorld = self.transformPoints2dFromLayerToWorld(roiCenters, roisLayer)
            roisNNs = distance.squareform(distance.pdist(roiCentersInWorld))
            np.fill_diagonal(roisNNs, np.inf)
            roisNNs = np.min(roisNNs, axis=1)
        if not neighborsLayer is None:
            neighborCenters = self.getRoiCenters2d(neighborsLayer)
            neighborCentersInWorld = self.transformPoints2dFromLayerToWorld(neighborCenters, neighborsLayer)
            neighborsNNs = distance.squareform(distance.pdist(neighborCentersInWorld))
            np.fill_diagonal(neighborsNNs, np.inf)
            neighborsNNs = np.min(neighborsNNs, axis=1)
        if (roisLayer is not None) and (neighborsLayer is not None):
            withinLayerNNs = np.concatenate([roisNNs, neighborsNNs])
            counts, bin_edges = np.histogram(withinLayerNNs, bins=bins)
            self.withinRoiLayersNearestNeighborsHistogram.setData(bin_edges, counts)
            betweenLayerNNs = np.min(np.linalg.norm(roiCentersInWorld[:, None, :] - neighborCentersInWorld[None, :, :], axis=-1), axis=1)
            counts, bin_edges = np.histogram(betweenLayerNNs, bins=bins)
            self.betweenRoiLayersNearestNeighborsHistogram.setData(bin_edges, counts)
        elif roisLayer is not None:
            counts, bin_edges = np.histogram(roisNNs, bins=bins)
            self.withinRoiLayersNearestNeighborsHistogram.setData(bin_edges, counts)
            self.betweenRoiLayersNearestNeighborsHistogram.setData([0, 0], [0])
        elif neighborsLayer is not None:
            counts, bin_edges = np.histogram(neighborsNNs, bins=bins)
            self.withinRoiLayersNearestNeighborsHistogram.setData(bin_edges, counts)
            self.betweenRoiLayersNearestNeighborsHistogram.setData([0, 0], [0])
        else:
            self.withinRoiLayersNearestNeighborsHistogram.setData([0, 0], [0])
            self.betweenRoiLayersNearestNeighborsHistogram.setData([0, 0], [0])
    
    def updateRoisColocalizationPlot(self):
        roisLayerName = self.colocalizeRoisLayerComboBox.currentText()
        neighborsLayerName = self.colocalizeNeighborsLayerComboBox.currentText()
        try:
            roisLayer = self.viewer.layers[roisLayerName]
        except KeyError:
            roisLayer = None
        try:
            neighborsLayer = self.viewer.layers[neighborsLayerName]
        except KeyError:
            neighborsLayer = None
        self.visualizeColocalizationOfRoisLayers(roisLayer, neighborsLayer)
    
    def findColocalizedRois(self):
        if self.viewer.grid.enabled:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Colocalization requires grid view to be disabled.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        roisLayerName = self.colocalizeRoisLayerComboBox.currentText()
        neighborsLayerName = self.colocalizeNeighborsLayerComboBox.currentText()
        try:
            roisLayer = self.viewer.layers[roisLayerName]
        except KeyError:
            roisLayer = None
        try:
            neighborsLayer = self.viewer.layers[neighborsLayerName]
        except KeyError:
            neighborsLayer = None
        if neighborsLayer is roisLayer:
            neighborsLayer = None
        if (roisLayer is None) or (neighborsLayer is None):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Select two points layers for colocalization.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec_()
            return
        roiCenters = self.getRoiCenters2d(roisLayer)
        roiCentersInWorld = self.transformPoints2dFromLayerToWorld(roiCenters, roisLayer)
        neighborCenters = self.getRoiCenters2d(neighborsLayer)
        neighborCentersInWorld = self.transformPoints2dFromLayerToWorld(neighborCenters, neighborsLayer)
        pdists = np.linalg.norm(roiCentersInWorld[:, None, :] - neighborCentersInWorld[None, :, :], axis=-1)
        roisNearestNeighborsDists = pdists.min(axis=1)
        # neighborsNearestPointsDists = pdists.min(axis=0)
        roisNearestNeighborsIndexes = pdists.argmin(axis=1)
        neighborsNearestRoisIndexes = pdists.argmin(axis=0)
        colocalizedRoiAndNeighborIndexes = []
        cutoff = self.colocalizeNearestNeighborCutoffSpinBox.value()
        for i in range(len(roiCentersInWorld)):
            j = roisNearestNeighborsIndexes[i]
            if neighborsNearestRoisIndexes[j] == i:
                if roisNearestNeighborsDists[i] <= cutoff:
                    colocalizedRoiAndNeighborIndexes.append([i, j])
        colocalizedRoiAndNeighborIndexes = np.array(colocalizedRoiAndNeighborIndexes, dtype=int)
        colocalizedRoiIndexes = colocalizedRoiAndNeighborIndexes[:,0]
        colocalizedNeighborIndexes = colocalizedRoiAndNeighborIndexes[:,1]
        colocalizedRoiCentersInWorld = np.reshape(roiCentersInWorld[colocalizedRoiIndexes], [-1, 2])
        colocalizedNeighborCentersInWorld = np.reshape(neighborCentersInWorld[colocalizedNeighborIndexes], [-1, 2])
        colocalizedRoiCentersInWorld = (colocalizedRoiCentersInWorld + colocalizedNeighborCentersInWorld) / 2
        colocalizedRoiCentersInRoisLayer = self.transformPoints2dFromWorldToLayer(colocalizedRoiCentersInWorld, roisLayer)
        name = "colocalized " + roisLayer.name + "-" + neighborsLayer.name
        return self.addRoisLayer(center=colocalizedRoiCentersInRoisLayer, name=name, affine=roisLayer.affine)


# UTILITIES

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


def zprojectRoiInImage(image, roiCenter, roiMask=None) -> np.ndarray:
    # image size
    n_rows, n_cols = image.shape[-2:]

    # pixel (row, col) from subpixel float point
    row, col = np.round(roiCenter).astype(int)

    # find overlap of point mask with image
    if (roiMask is None) or np.all(roiMask.shape == 1):
        # z-project single pixel
        if (0 <= row < n_rows) and (0 <= col < n_cols):
            if image.ndim == 2:
                zproj = image[row,col]
            elif image.ndim == 3:
                zproj = np.squeeze(image[:,row,col])
            else:
                zproj = np.squeeze(image[...,row,col])
        else:
            zproj = np.array([])
    else:
        # z-project overlap between mask and image
        n_mask_rows, n_mask_cols = roiMask.shape
        if n_mask_rows % 2 == 0:
            # even rows
            drows = n_mask_rows / 2
            if roiCenter[0] >= row:
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
            if roiCenter[1] >= col:
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
            roiMask = np.reshape(roiMask[pointMaskInImage], [1] * (image.ndim - 2) + [len(rows), len(cols)])
            dim_starts = [0] * (image.ndim - 2) + [rows[0], cols[0]]
            dim_stops = list(image.shape[:-2]) + [rows[-1] + 1, cols[-1] + 1]
            pointMaskIndexer = tuple([slice(i,j) for (i,j) in zip(dim_starts, dim_stops)])
            zproj = np.squeeze(np.mean(image[pointMaskIndexer] * roiMask, axis=(-2, -1)))
        else:
            zproj = np.array([])
    return zproj


def getRoiMask(roiSize, roiShape="ellipse"):
    try:
        n_rows, n_cols = np.ceil(roiSize).astype(int)
    except:
        n_rows = np.ceil(roiSize).astype(int)
        n_cols = n_rows
    if roiShape in ["point", "ellipse"]:
        rows = np.reshape(np.arange(n_rows, dtype=float), [-1, 1])
        rows -= rows.mean()
        cols = np.reshape(np.arange(n_cols, dtype=float), [1, -1])
        cols -= cols.mean()
        pointMask = rows**2 / (n_rows / 2)**2  + cols**2 / (n_cols / 2)**2 < 1
        return pointMask
    elif roiShape == "rectangle":
        pointMask = np.ones([n_rows, n_cols]) > 0
        return pointMask
    return None
    
    
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


def str2slice(sliceStr):
    if sliceStr.strip() == "":
        return (slice(None),)
    dimSliceStrs = [dimSliceStr.strip() for dimSliceStr in sliceStr.split(',')]
    slices = []  # one slice per dimension
    for dimSliceStr in dimSliceStrs:
        sliceIndexes = [int(idx) if idx.strip != "" else None for idx in dimSliceStr.split(':')]
        slices.append(slice(*sliceIndexes))
    return tuple(slices)


def str2rgba(color):
    color = color.strip()
    if color == "":
        # transparent
        return [0]*4
    elif ',' in color:
        rgb_or_rgba = [float(c) for c in color.split(',')]
        return rgb_or_rgba + [1.0] * (4 - len(rgb_or_rgba))
    else:
        qcolor = QColor(color)
        return [qcolor.redF(), qcolor.greenF(), qcolor.blueF(), qcolor.alphaF()]


# def getComplementaryColor(rgb_or_rgba):
#     r, g, b = rgb_or_rgba[:3]
#     h, s, v = rgb_to_hsv(r, g, b)
#     rgb_or_rgba[:3] = hsv_to_rgb((h + 0.5) % 1, s, v)
#     return rgb_or_rgba


# def array2qimage(array2d):
#     array2d = array2d.astype(np.float32)
#     array2d -= array2d.min()
#     array2d /= array2d.max()
#     array2d *= 255
#     array2d[array2d > 255] = 255
#     array2d[array2d < 0] = 0
#     array2d = array2d.astype(np.uint8)
#     height, width = array2d.shape
#     bytes = array2d.tobytes()
#     qimage = QImage(bytes, width, height, QImage.Format.Format_Grayscale8)
#     return qimage


# def array2qpixmap(array2d):
#     qimage = array2qimage(array2d)
#     qpixmap = QPixmap.fromImage(qimage)
#     return qpixmap


def clearQLayout(layout):
    while layout.count():
        child = layout.takeAt(0)
        if child.widget():
            child.widget().deleteLater()


# RUN AS APPLICATION

if __name__ == "__main__":
    viewer = napari.Viewer()
    ui = CoSMoS_TS_napari_UI(viewer)
    viewer.window.add_dock_widget(ui, name='CoSMoS-TS', area='right')
    napari.run()