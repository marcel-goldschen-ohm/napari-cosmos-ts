""" Main widget for the napari-cosmos-ts plugin.
"""

import os
import numpy as np
import pandas as pd
from napari.viewer import Viewer
from napari.layers import Layer, Image, Points
from napari.utils.events import Event
from qtpy.QtWidgets import QTabWidget
import pyqtgraph as pg


class MainWidget(QTabWidget):
    """ Main widget for the napari-cosmos-ts plugin.
    """
    
    def __init__(self, viewer: Viewer, parent=None):
        QTabWidget.__init__(self, parent=parent)
        self.viewer: Viewer = viewer

        # Layer metadata to be stored separately from layers.
        # This keeps non-serealizable objects (e.g., QObjects) out of the layer metadata dicts.
        # Data that can be serialized and should be saved should be stored in the individual layer metadata dicts.
        self._layer_metadata: list[dict] = []

        # points layer for just the selected projection point 
        self._selected_point_layer = None

        # UI
        self._setup_ui()

        # event handling
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.layers.events.moved.connect(self._on_layer_moved)
        self.viewer.dims.events.current_step.connect(self._on_dim_step_changed)
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_clicked_or_dragged)
        self.viewer.mouse_double_click_callbacks.append(self._on_mouse_doubleclicked)
    
    def export_session(self, filepath: str = None):
        """ Export data to MATLAB .mat file.
        """
        from scipy.io import savemat

        if filepath is None:
            from qtpy.QtWidgets import QFileDialog
            filepath, _filter = QFileDialog.getSaveFileName(self, "Save session", "", "MATLAB files (*.mat)")
            if filepath == "":
                return
        session_abspath = os.path.abspath(filepath)
        session_absdir, session_file = os.path.split(session_abspath)
        
        # session dict
        session = {}
        session['date'] = self._date_edit.text() + " "
        session['ID'] = self._id_edit.text() + " "
        session['users'] = self._users_edit.text() + " "
        session['notes'] = self._notes_edit.toPlainText() + " "
        session['default_point_size'] = self._default_point_size_spinbox.value()
        
        # layer dicts
        session['layers'] = []
        for layer in self.viewer.layers:
            layer_data = {}
            layer_data['name'] = layer.name
            layer_data['affine'] = layer.affine.affine_matrix
            layer_data['opacity'] = layer.opacity
            layer_data['blending'] = layer.blending
            layer_data['visible'] = layer.visible

            if isinstance(layer, Image):
                layer_data['type'] = 'image'
                image_abspath = self._image_layer_abspath(layer)
                if image_abspath is not None:
                    image_relpath = os.path.relpath(image_abspath, start=session_absdir)
                    layer_data['abspath'] = image_abspath
                    layer_data['relpath'] = image_relpath
                if image_abspath is None:
                    # store image data if it does not already exist on disk
                    layer_data['data'] = layer.data
                if 'data' not in layer_data:
                    # if image data is not stored in the session, store shape and dtype
                    layer_data['data_shape'] = layer.data.shape
                    layer_data['data_dtype'] = str(layer.data.dtype)
                layer_data['contrast_limits'] = layer.contrast_limits
                layer_data['gamma'] = layer.gamma
                layer_data['colormap'] = layer.colormap.name
                layer_data['interpolation2d'] = layer.interpolation2d
            
            elif isinstance(layer, Points):
                layer_data['type'] = 'points'
                layer_data['data'] = layer.data
                layer_data['size'] = layer.size
                layer_data['symbol'] = [str(symbol) for symbol in layer.symbol]
                layer_data['face_color'] = layer.face_color
                layer_data['edge_color'] = layer.edge_color
                layer_data['edge_width'] = layer.edge_width
                layer_data['edge_width_is_relative'] = layer.edge_width_is_relative

                if not layer.features.empty:
                    layer_data['features'] = {}
                    for key in layer.features:
                        if key == 'tags':
                            layer.features['tags'] = layer.features['tags'].fillna("")
                            layer.features['tags'] = layer.features['tags'].replace("", " ")
                        layer_data['features'][key] = layer.features[key].to_numpy()
            
            # layer metadata
            if layer.metadata:
                layer_data['metadata'] = layer.metadata
            
            # add layer data to session
            session['layers'].append(layer_data)
        
        # save session to .mat file
        savemat(filepath, session)
    
    def import_session(self, filepath: str = None):
        """ Import data from MATLAB .mat file.
        """
        from scipy.io import loadmat

        if filepath is None:
            from qtpy.QtWidgets import QFileDialog
            filepath, _filter = QFileDialog.getOpenFileName(self, "Open session", "", "MATLAB files (*.mat)")
            if filepath == "":
                return
        session_abspath = os.path.abspath(filepath)
        session_absdir, session_file = os.path.split(session_abspath)

        self.viewer.layers.clear()
        self._selected_point_layer = None

        session = loadmat(filepath, simplify_cells=True)

        for key, value in session.items():
            if key == "date":
                self._date_edit.setText(str(value).strip())
            elif key == "ID":
                self._id_edit.setText(str(value).strip())
            elif key == "users":
                self._users_edit.setText(str(value).strip())
            elif key == "notes":
                self._notes_edit.setPlainText(str(value).strip())
            elif key == "default_point_size":
                self._default_point_size_spinbox.setValue(session['default_point_size'])
            elif key == "layers":
                for layer_data in value:
                    layer = None
                    
                    if layer_data['type'] == 'image':
                        abspath = None
                        if 'relpath' in layer_data:
                            relpath = layer_data['relpath']
                            abspath = os.path.join(session_absdir, relpath)
                        elif 'abspath' in layer_data:
                            abspath = layer_data['abspath']
                        
                        if 'data' in layer_data:
                            image = layer_data['data']
                            layer = self.viewer.add_image(image)
                        elif abspath is not None:
                            try:
                                layer = self.viewer.open(abspath, layer_type="image")[0]
                            except Exception as error:
                                print(error)
                        if layer is None:
                            continue
                        
                        if 'contrast_limits' in layer_data:
                            layer.contrast_limits = layer_data['contrast_limits']
                        if 'gamma' in layer_data:
                            layer.gamma = layer_data['gamma']
                        if 'colormap' in layer_data:
                            layer.colormap = layer_data['colormap']
                        if 'interpolation2d' in layer_data:
                            layer.interpolation2d = layer_data['interpolation2d']
                    
                    elif layer_data['type'] == 'points':
                        points = layer_data['data']
                        layer = self.viewer.add_points(points)
                        
                        if 'size' in layer_data:
                            layer.size = layer_data['size']
                        if 'symbol' in layer_data:
                            layer.symbol = layer_data['symbol']
                        if 'face_color' in layer_data:
                            layer.face_color = layer_data['face_color']
                        if 'edge_color' in layer_data:
                            layer.edge_color = layer_data['edge_color']
                        if 'edge_width' in layer_data:
                            layer.edge_width = layer_data['edge_width']
                        if 'edge_width_is_relative' in layer_data:
                            layer.edge_width_is_relative = layer_data['edge_width_is_relative']
                        
                        n_points = len(layer.data)
                        features = pd.DataFrame({"tags": [""] * n_points})
                        if 'features' in layer_data:
                            for key in layer_data['features']:
                                features[key] = layer_data['features'][key]
                                if key == "tags":
                                    features['tags'] = features['tags'].replace(" ", "")
                        layer.features = features
                    
                    if 'name' in layer_data:
                        layer.name = layer_data['name']
                    if 'affine' in layer_data:
                        layer.affine = layer_data['affine']
                    if 'opacity' in layer_data:
                        layer.opacity = layer_data['opacity']
                    if 'blending' in layer_data:
                        layer.blending = layer_data['blending']
                    if 'visible' in layer_data:
                        layer.visible = layer_data['visible']
                    if 'metadata' in layer_data:
                        layer.metadata = layer_data['metadata']
                    
                    if isinstance(layer, Image):
                        if abspath is not None:
                            layer.metadata['abspath'] = abspath
                        if 'slice' in layer.metadata:
                            slice_str = layer.metadata['slice']
                            layer.data = layer.data[slice_from_str(slice_str)]
                    elif isinstance(layer, Points):
                        if 'is_selected_point_layer' in layer.metadata:
                            self._selected_point_layer = layer
        
        if self._selected_point_layer is not None:
            self.set_projection_point(self._selected_point_layer.data)
        
        self._update_layer_selection_comboboxes()
    
    def split_image_layer(self, layer: Image, regions: str = None) -> list[Image]:
        """ Split image horizontally, vertically, or into quadrants.

        Return each split as a new layer.

        If the original image layer reflects a source image file, the split layers will keep track of the file path and slice in their metadata.
        """
        if not isinstance(layer, Image):
            return

        if regions is None:
            regions = self._split_image_regions_combobox.currentText()
        
        n_rows = layer.data.shape[-2]
        n_cols = layer.data.shape[-1]
        mid_row = int(n_rows / 2)
        mid_col = int(n_cols / 2)

        stack_slice = (slice(None),) * (layer.data.ndim - 2)

        if regions == "Top/Bottom":
            slices = {
                'bottom': stack_slice + np.index_exp[-mid_row:,:],
                'top': stack_slice + np.index_exp[:mid_row,:],
            }
        elif regions == "Left/Right":
            slices = {
                'right': stack_slice + np.index_exp[:,-mid_col:],
                'left': stack_slice + np.index_exp[:,:mid_col],
            }
        elif regions == "Quad":
            slices = {
                'bottomright': stack_slice + np.index_exp[-mid_row:,-mid_col:],
                'bottomleft': stack_slice + np.index_exp[-mid_row:,:mid_col],
                'topright': stack_slice + np.index_exp[:mid_row,-mid_col:],
                'topleft': stack_slice + np.index_exp[:mid_row,:mid_col],
            }
        
        new_layers = []
        for region, region_slice in slices.items():
            new_layer = self.viewer.add_image(
                layer.data[region_slice],
                name = layer.name + f" ({region})",
                affine = layer.affine,
                blending = layer.blending,
                colormap = layer.colormap,
                opacity = layer.opacity
            )
            new_layers.append(new_layer)

        if layer.source.path is not None:
            has_parent_slice = False
            if 'slice' in layer.metadata:
                parent_slice = slice_from_str(layer.metadata['slice'])
                has_parent_slice = True
            for new_layer, image_slice in zip(new_layers, slices.values()):
                if has_parent_slice:
                    image_slice = combine_slices(parent_slice, image_slice, layer.data.shape)
                new_layer.metadata['abspath'] = layer.source.path
                new_layer.metadata['slice'] = str_from_slice(image_slice)

        return new_layers

    def slice_image_layer(self, layer: Image, image_slice: tuple[slice] = None) -> Image:
        """ Return a slice of an image as a new layer.

        If the original image layer reflects a source image file, the split layers will keep track of the file path and slice in their metadata.
        """
        if not isinstance(layer, Image):
            return
        
        if image_slice is None:
            image_slice: str = self._slice_image_edit.text().strip()
        if isinstance(image_slice, str):
            image_slice: tuple[slice] = slice_from_str(image_slice)
        
        new_layer: Image = self.viewer.add_image(
            layer.data[image_slice],
            name = layer.name + " (sliced)",
            affine = layer.affine,
            blending = layer.blending,
            colormap = layer.colormap,
            opacity = layer.opacity
        )

        if layer.source.path is not None:
            if 'slice' in layer.metadata:
                parent_slice = slice_from_str(layer.metadata['slice'])
                image_slice = combine_slices(parent_slice, image_slice, layer.data.shape)
            new_layer.metadata['abspath'] = layer.source.path
            new_layer.metadata['slice'] = str_from_slice(image_slice)
        
        return new_layer
    
    def project_image_layer(self, layer: Image, operation: str = None) -> Image:
        """ Return a projection of an image as a new layer.
        """
        if not isinstance(layer, Image):
            return
        
        if operation is None:
            operation = self._project_image_operation_combobox.currentText()
        operation = operation.lower()
        
        operations = {"max": np.max, "min": np.min, "std": np.std, "sum": np.sum, "mean": np.mean, "median": np.median}
        if operation not in operations:
            raise ValueError(f"Invalid operation: {operation}")
        
        func = operations[operation]
        projected = func(layer.data, axis=0)
        
        new_layer = self.viewer.add_image(
            projected,
            name = layer.name + f" ({operation}-proj)",
            affine = layer.affine.affine_matrix[-3:,-3:],
            blending = layer.blending,
            colormap = layer.colormap,
            opacity = layer.opacity
        )
        return new_layer
    
    def filter_image_layer(self, layer: Image, filter_type: str, **kwargs) -> Image:
        """ Return a filtered image as a new layer.
        """
        if not isinstance(layer, Image):
            return

        if filter_type == "gauss":
            from skimage import filters
            sigma = kwargs.get('sigma', self._gaussian_filter_sigma_spinbox.value())
            # default is to NOT blur together images in stack (i.e., sigma=0 along non-row/col dimensions)
            if type(sigma) is float:
                sigma = [0]*(layer.ndim - 2) + [sigma, sigma]
            else:
                sigma = [0]*(layer.ndim - len(sigma)) + list(sigma)
            filtered = filters.gaussian(layer.data, sigma=sigma, preserve_range=True)
        elif filter_type == "tophat":
            from skimage import morphology
            disk_radius = kwargs.get('disk_radius', self._tophat_filter_disk_radius_spinbox.value())
            disk = morphology.disk(disk_radius)
            # need to filter each image in stack separately
            if layer.ndim == 2:
                filtered = morphology.white_tophat(layer.data, disk)
            elif layer.ndim > 2:
                filtered = np.empty(layer.data.shape)
                stack_shape = layer.data.shape[:-2]
                stack_index_permutations = [(i,) for i in range(stack_shape[0])]
                for dim in range(1, len(stack_shape)):
                    stack_index_permutations = [index + (i,) for i in range(stack_shape[dim]) for index in stack_index_permutations]
                for index in stack_index_permutations:
                    filtered[index] = morphology.white_tophat(layer.data[index], disk)
            else:
                raise ValueError(f"Invalid layer shape: {layer.data.shape}")
        
        new_layer = self.viewer.add_image(
            filtered,
            name = layer.name + f" ({filter_type}-filt)",
            affine = layer.affine,
            blending = layer.blending,
            colormap = layer.colormap,
            opacity = layer.opacity
        )
        return new_layer
    
    def register_layers(self, fixed_layer: Layer = None, moving_layer: Layer = None, transform_type: str = None):
        """ Set transformation of moving layer to align moving layer to fixed layer.

        Implements both image and points registration.
        """
        if fixed_layer is None:
            fixed_layer_name = self._fixed_layer_combobox.currentText()
            fixed_layer = self.viewer.layers[fixed_layer_name]
        if moving_layer is None:
            moving_layer_name = self._moving_layer_combobox.currentText()
            moving_layer = self.viewer.layers[moving_layer_name]
        if transform_type is None:
            transform_type = self._layer_transform_type_combobox.currentText()
        transform_type = transform_type.lower()

        if fixed_layer is moving_layer:
            from qtpy.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Cannot register a layer with itself.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()
            return
        if isinstance(fixed_layer, Image) and isinstance(moving_layer, Image):
            self.register_image_layers(fixed_layer, moving_layer, transform_type)
        elif isinstance(fixed_layer, Points) and isinstance(moving_layer, Points):
            self.register_points_layers(fixed_layer, moving_layer, transform_type)
        else:
            from qtpy.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Only image-image or points-points layer registration implemented.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()

    def register_image_layers(self, fixed_layer: Image, moving_layer: Image, transform_type: str = "affine"):
        """ Set transformation of moving layer to align moving image to fixed image.

        Uses pystackreg for image registration.
        """
        try:
            from pystackreg import StackReg
        except ImportError:
            from qtpy.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Image registration requires pystackreg package.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()
            return
        
        # get current image or frame if layer is an image stack
        fixed_image = fixed_layer.data
        moving_image = moving_layer.data
        if fixed_image.ndim > 2:
            ind = self.viewer.dims.current_step[-fixed_image.ndim:-2]
            fixed_image = fixed_image[*ind,:,:]
        if moving_image.ndim > 2:
            ind = self.viewer.dims.current_step[-moving_image.ndim:-2]
            moving_image = moving_image[*ind,:,:]

        # adjust image to match layer contrast limits
        fixed_image = normalize_image(fixed_image, fixed_layer.contrast_limits)
        moving_image = normalize_image(moving_image, moving_layer.contrast_limits)
        
        # register images
        tform = register_images(fixed_image, moving_image, transform_type)
        
        # apply net world transform to moving image
        moving_layer.affine = tform @ fixed_layer.affine.affine_matrix[-3:,-3:]

    def register_points_layers(self, fixed_layer: Points, moving_layer: Points, transform_type: str = "affine"):
        """ Set transformation of moving layer to align moving points to fixed points.

        Uses pycpd for point registration.
        """
        try:
            from pycpd import RigidRegistration, AffineRegistration
        except ImportError:
            from qtpy.QtWidgets import QMessageBox
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Image registration requires pycpd package.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()
            return
        
        fixed_points = fixed_layer.data[:,-2:]
        moving_points = moving_layer.data[:,-2:]

        # register points
        tform = register_points(fixed_points, moving_points, transform_type)
        
        # apply net world transform to moving points
        moving_layer.affine = tform @ fixed_layer.affine.affine_matrix[-3:,-3:]
    
    def find_image_peaks(self, layer: Image, min_peak_height: float = None, min_peak_separation: float = None) -> Points:
        """ Return position of local peaks in image as new points layer.

        Uses the current visible frame for image stacks.
        """
        if min_peak_height is None:
            min_peak_height = self._min_peak_height_spinbox.value()
        if min_peak_separation is None:
            min_peak_separation = self._min_peak_separation_spinbox.value()
        image = layer.data
        if image.ndim > 2:
            ind = self.viewer.dims.current_step[-image.ndim:-2]
            image = image[*ind,:,:]
        
        points = find_image_peaks(image, min_peak_height, min_peak_separation)
        n_points = len(points)

        new_layer = self.viewer.add_points(
            points,
            name = layer.name + " (peaks)",
            affine = layer.affine.affine_matrix[-3:,-3:],
            symbol = "disc",
            size = [self._default_point_size_spinbox.value()] * n_points,
            face_color = [[1, 1, 0, 0.1]] * n_points,
            edge_color = [[1, 1, 0, 1]] * n_points,
            opacity = 1,
            blending = "translucent_no_depth",
        )
        return new_layer

    def find_colocalized_points(self, layer: Points = None, neighbors_layer: Points = None, nearest_neighbor_cutoff: float = None) -> Points:
        """ Return new points layer with colocalized points between two input points layers.

        For each pair of colocalized points, their mean position is returned.
        """
        if layer is None:
            layer_name = self._coloc_layer_combobox.currentText()
            try:
                layer = self.viewer.layers[layer_name]
            except KeyError:
                return
        if neighbors_layer is None:
            neighbors_layer_name = self._coloc_neighbors_layer_combobox.currentText()
            try:
                neighbors_layer = self.viewer.layers[neighbors_layer_name]
            except KeyError:
                return
        if neighbors_layer is layer:
            return
        if nearest_neighbor_cutoff is None:
            nearest_neighbor_cutoff = self._coloc_nearest_neighbor_cutoff_spinbox.value()

        # points in world coords
        points = self._transform_points2d_from_layer_to_world(layer.data[:,-2:], layer)
        neighbors = self._transform_points2d_from_layer_to_world(neighbors_layer.data[:,-2:], neighbors_layer)

        # colocalized points in world coords
        colocalized = find_colocalized_points(points, neighbors, nearest_neighbor_cutoff)

        # colocalized points layer
        n_points = len(colocalized)
        new_layer = self.viewer.add_points(
            colocalized,
            name = "colocalized",
            symbol = "disc",
            size = [self._default_point_size_spinbox.value()] * n_points,
            face_color = [[1, 0, 1, 0.1]] * n_points,
            edge_color = [[1, 0, 1, 1]] * n_points,
            opacity = 1,
            blending = "translucent_no_depth",
        )
        return new_layer
    
    def set_projection_point(self, worldpt2d: np.ndarray | None, point_size: int | float = None):
        """ Set the world position for the currently visible point projections.

        This will update the point projection plots for all image stack layers.
        """
        if worldpt2d is None:
            # clear selected projection point overlay
            if self._selected_point_layer is not None:
                self.viewer.layers.remove(self._selected_point_layer)
                self._selected_point_layer = None
            
            # clear point projection plots
            for metadata in self._layer_metadata:
                if 'point_projection_data' in metadata:
                    metadata['point_projection_data'].setData([])
            
            # update point projections tab
            self._projection_point_world_label.setText("")
            self._tag_edit.setText("")
            
            return
        
        worldpt2d = np.array(worldpt2d).reshape([1, 2])

        if point_size is None:
            point_size = self._default_point_size_spinbox.value()
        
        # update selected projection point overlay
        if self._selected_point_layer is None:
            self._selected_point_layer = self.viewer.add_points(
                worldpt2d,
                name = "selected point",
                symbol = "disc",
                size = [point_size],
                face_color = [[0, 1, 1, 0.1]],
                edge_color = [[0, 1, 1, 1]],
                opacity = 1,
                blending = "translucent_no_depth",
            )
            self._selected_point_layer.metadata['is_selected_point_layer'] = True
            # because self._selected_point_layer is not yet defined during layer insertion
            self._update_layer_selection_comboboxes(self._selected_point_layer)
        else:
            self._selected_point_layer.data = worldpt2d
            self._selected_point_layer.size = [point_size]
        
        # update point projections tab
        row, col = np.round(worldpt2d).astype(int).flatten()
        self._projection_point_world_label.setText(f"[{row}, {col}]")
        self._tag_edit.setText("")
        
        # project selected point for all imagestack layers
        # use circular mask for point projection
        pixel_size = int(np.round(self._selected_point_layer.size[0]))
        point_mask = make_point_mask(pixel_size, type='circle')
        for layer, metadata in zip(self.viewer.layers, self._layer_metadata):
            if isinstance(layer, Image) and layer.data.ndim == 3:
                if 'point_projection_data' in metadata:
                    # selected point in layer
                    layerpt2d = self._transform_points2d_from_world_to_layer(worldpt2d, layer)
                    # project point
                    xdata = None  # assumed integer frames
                    ydata = project_image_point(layer.data, layerpt2d, point_mask)
                    # sum frames?
                    if 'projection-sum-frames' in metadata:
                        nframes = metadata['projection-sum-frames']
                        if nframes > 1:
                            xdata = np.arange(0, len(ydata), nframes)
                            tmp_ydata = np.zeros(xdata.shape)
                            for i in range(nframes):
                                tmp = ydata[i::nframes]
                                tmp_ydata[:len(tmp)] += tmp
                            ydata = tmp_ydata
                    # update plot
                    if xdata is None:
                        metadata['point_projection_data'].setData(ydata)
                    else:
                        metadata['point_projection_data'].setData(xdata, ydata)
    
    def select_projection_point(self, layer: Points = None, point_index: int = None, ignore_tag_filter: bool = False):
        """ Select an existing point to use as the position for the currently visible point projections.

        The selected point will be used for projection in all image stack layers.
        """
        # add guard because blockSignals() does not work for QSpinBox.valueChanged
        # and we don't want to trigger this function recursively in an infinite loop
        if getattr(self, '_select_projection_point_in_progress', False):
            return
        self._select_projection_point_in_progress = True

        self._projection_points_layer_combobox.blockSignals(True)
        self._projection_point_index_spinbox.blockSignals(True)

        if layer is None:
            layer_name = self._projection_points_layer_combobox.currentText()
            if layer_name != "":
                layer = self.viewer.layers[layer_name]
        else:
            self._projection_points_layer_combobox.setCurrentText(layer.name)
        
        if layer is None:
            self._n_projection_points_label.setText("")
            self._projection_point_index_spinbox.clear()
        else:
            n_points = layer.data.shape[0]
            self._n_projection_points_label.setText(f"{n_points} pts")
            self._projection_point_index_spinbox.setMaximum(n_points - 1)

            if point_index is None:
                if self._projection_point_index_spinbox.text() != "":
                    point_index = self._projection_point_index_spinbox.value()
            else:
                self._projection_point_index_spinbox.setValue(point_index)
                point_index = self._projection_point_index_spinbox.value()
            
            if n_points == 0:
                point_index = None
            
            if point_index is None:
                self._projection_point_index_spinbox.clear()
        
        if layer is not None and point_index is not None:
            # tag filter
            if (not ignore_tag_filter) and self._tag_filter_checkbox.isChecked() and ('tags' in layer.features):
                tag_filter = [tag.strip() for tag in self._tag_filter_edit.text().split(",")]
                tags = [tag.strip() for tag in layer.features['tags'][point_index].split(",")]
                if not any(tag in tags for tag in tag_filter):
                    found_point = False
                    n_points = len(layer.data)
                    prev_point_index = getattr(self, '_selected_point_index', None)
                    if prev_point_index is None or prev_point_index <= point_index:
                        search_direction = ["forward", "backward"]
                    else:
                        search_direction = ["backward", "forward"]
                    for direction in search_direction:
                        if found_point:
                            break
                        if direction == "forward":
                            for i in range(point_index + 1, n_points):
                                tags = [tag.strip() for tag in layer.features['tags'][i].split(",")]
                                if any(tag in tags for tag in tag_filter):
                                    point_index = i
                                    found_point = True
                                    break
                        elif direction == "backward":
                            for i in reversed(range(point_index)):
                                tags = [tag.strip() for tag in layer.features['tags'][i].split(",")]
                                if any(tag in tags for tag in tag_filter):
                                    point_index = i
                                    found_point = True
                                    break
                    if found_point:
                        self._projection_point_index_spinbox.setValue(point_index)

            # project point
            layerpt2d = layer.data[point_index,-2:]
            worldpt2d = self._transform_points2d_from_layer_to_world(layerpt2d, layer)
            point_size = layer.size[point_index]
            self.set_projection_point(worldpt2d, point_size)

            # show point tags
            try:
                tags: str = layer.features['tags'][point_index]
            except (KeyError, IndexError):
                tags = ""
            self._tag_edit.setText(tags)
        
        self._projection_points_layer_combobox.blockSignals(False)
        self._projection_point_index_spinbox.blockSignals(False)

        # keep track of currently selected point
        self._selected_point_index = point_index
        
        # remove guard
        self._select_projection_point_in_progress = False
    
    def update_point_projections(self):
        is_index_selected = self._projection_point_index_spinbox.text().strip() != ''
        if is_index_selected:
            index = self._projection_point_index_spinbox.value()
            layer_name = self._projection_points_layer_combobox.currentText()
            try:
                layer = self.viewer.layers[layer_name]
                self.select_projection_point(layer, index, ignore_tag_filter=True)
                return
            except KeyError:
                pass
        
        # no index selected, check for selected point layer
        if self._selected_point_layer is not None:
            worldpt2d = self._selected_point_layer.data
            self.set_projection_point(worldpt2d)
            return
    
    def set_point_tags(self, layer: Points = None, point_index: int = None, tags: str = None):
        """ Set tags for a point.

        The tags string will be stored in the 'tags' feature column of the layer.
        """
        if layer is None:
            layer_name = self._projection_points_layer_combobox.currentText()
            try:
                layer = self.viewer.layers[layer_name]
            except KeyError:
                return
        if point_index is None:
            if self._projection_point_index_spinbox.text() == "":
                return
            point_index = self._projection_point_index_spinbox.value()
        
        if tags is None:
            tags = self._tag_edit.text().strip()

        if not 'tags' in layer.features:
            layer.features['tags'] = [""] * len(layer.data)
        
        layer.features.loc[point_index, 'tags'] = tags
    
    def _layers(self, include_selected_point_layer: bool = False):
        """ Return a list of all layers.

        Layer order is reversed so that the list corresponds to the layers in top-to-bottom order as seen in the viewer.

        By default, the selected point layer is excluded from the list.
        """
        layers = [layer for layer in reversed(self.viewer.layers)]
        if not include_selected_point_layer:
            if self._selected_point_layer in layers:
                layers.remove(self._selected_point_layer)
        return layers
    
    def _selected_layers(self, include_selected_point_layer: bool = False):
        """ Return a list of all selected layers.

        By default, the selected point layer is excluded from the list.
        """
        return [layer for layer in self._layers(include_selected_point_layer) if layer in self.viewer.layers.selection]
    
    def _image_layers(self):
        """ Return a list of image layers.
        """
        return [layer for layer in self._layers() if isinstance(layer, Image)]
    
    def _imagestack_layers(self):
        """ Return a list of image stack layers.
        """
        return [layer for layer in self._layers() if isinstance(layer, Image) and layer.data.ndim == 3]
    
    def _points_layers(self, include_selected_point_layer: bool = False):
        """ Return a list of points layers.

        By default, the selected point layer is excluded from the list.
        """
        return [layer for layer in self._layers(include_selected_point_layer) if isinstance(layer, Points)]
    
    def _image_layer_abspath(self, layer: Image) -> str | None:
        """ Return the source image file path.

        It is not possible to set the layer.source.path attribute directly, so in some cases the path may be stored in the layer metadata.
        """
        abspath = None
        if layer.source.path is not None:
            abspath = os.path.abspath(layer.source.path)
        if abspath is None:
            if 'abspath' in layer.metadata:
                abspath = os.path.abspath(layer.metadata['abspath'])
        return abspath
    
    def _on_layer_inserted(self, event: Event):
        """ Callback for layer insertion event.
        """
        from qtpy.QtGui import QColor
        from qtpy.QtCore import QSize

        layer = event.value
        layer_index = event.index

        # insert layer metadata
        metadata = {}
        self._layer_metadata.insert(layer_index, metadata)

        # image stack
        if isinstance(layer, Image) and layer.data.ndim > 2:
            # point projection plot
            plot = self._new_plot()
            plot.getAxis('left').setWidth(82)
            plot.setMinimumSize(QSize(150, 50))
            plot.setLabels(left=layer.name)
                
            # xlink point projection plots
            other_plots = [meta['point_projection_plot'] for meta in self._layer_metadata if 'point_projection_plot' in metadata]
            if other_plots:
                plot.setXLink(other_plots[0])
            
            # position plot to match layer order
            plot_index = self._imagestack_layers().index(layer)
            self._point_projection_plots_layout.insertWidget(plot_index, plot)
            self._point_projection_plots_layout.setStretch(plot_index, 1)
            
            # point projection data
            data = plot.plot([], pen=pg.mkPen(QColor(0, 114, 189), width=1))
            
            # current frame vertical line
            frame_index = self._current_frame()
            vline = plot.addLine(x=frame_index, pen=pg.mkPen('y', width=1))

            metadata['point_projection_plot'] = plot
            metadata['point_projection_data'] = data
            metadata['point_projection_vline'] = vline
        
        # update layer selection lists
        self._update_layer_selection_comboboxes(layer)
        
        # handle events for new layer
        layer.events.name.connect(self._on_layer_name_changed)
        layer.events.visible.connect(self._on_layer_visibility_changed)
        layer.events.mode.connect(self._on_layer_mode_changed)

    def _on_layer_removed(self, event: Event):
        """ Callback for layer removal event.
        """
        layer = event.value
        layer_index = event.index

        # remove layer metadata
        metadata = self._layer_metadata.pop(layer_index)
        
        # image stack
        if isinstance(layer, Image) and layer.data.ndim > 2:
            # remove point projection plot
            if 'point_projection_plot' in metadata:
                plot = metadata['point_projection_plot']
                self._point_projection_plots_layout.removeWidget(plot)
                plot.deleteLater()
        
        # selected projection point overlay
        if self._selected_point_layer is layer:
            self._selected_point_layer = None
            self.set_projection_point(None)
        
        # update layer selection lists
        self._update_layer_selection_comboboxes(layer)

    def _on_layer_moved(self, event: Event):
        """ Callback for layer move event.
        """
        layer = event.value
        old_layer_index = event.index
        new_layer_index = event.new_index

        # move layer metadata to match layer move
        metadata = self._layer_metadata.pop(old_layer_index)
        self._layer_metadata.insert(new_layer_index, metadata)
        
        # image stack
        if isinstance(layer, Image) and layer.data.ndim > 2:
            # reorder point projection plots
            if 'point_projection_plot' in metadata:
                plot = metadata['point_projection_plot']
                self._point_projection_plots_layout.removeWidget(plot)
                plot_index = self._imagestack_layers().index(layer)
                self._point_projection_plots_layout.insertWidget(plot_index, plot)
                self._point_projection_plots_layout.setStretch(plot_index, 1)

        # update layer selection lists
        self._update_layer_selection_comboboxes(layer)

    def _on_layer_name_changed(self, event: Event):
        """ Callback for layer name change event.
        """
        layer_index = event.index
        layer = self.viewer.layers[layer_index]
        metadata = self._layer_metadata[layer_index]
        
        # image stack
        if isinstance(layer, Image) and layer.data.ndim > 2:
            # rename point projection plot
            if 'point_projection_plot' in metadata:
                plot = metadata['point_projection_plot']
                plot.setLabels(left=layer.name)

        # update layer selection lists
        self._update_layer_selection_comboboxes(layer)

    def _on_layer_visibility_changed(self, event: Event):
        """ Callback for layer visibility change event.
        """
        layer_index = event.index
        layer = self.viewer.layers[layer_index]
        metadata = self._layer_metadata[layer_index]
        
        # image stack
        if isinstance(layer, Image) and layer.data.ndim > 2:
            # show/hide point projection plot
            if 'point_projection_plot' in metadata:
                plot = metadata['point_projection_plot']
                plot.setVisible(layer.visible)

    def _on_layer_mode_changed(self, event: Event):
        """ Callback for layer mode change event.
        """
        layer_index = event.index
        layer = self.viewer.layers[layer_index]
        metadata = self._layer_metadata[layer_index]

        if isinstance(layer, Image):
            if event.mode == "transform":
                # !!! If any image layers have more dimensions than this layer, transform will throw an error.
                # To avoid this, temporarily expand this image to have the max number of dimensions.
                # We'll convert back upon exiting transform mode.
                image_layers = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
                ndim = [layer.data.ndim for layer in image_layers]
                maxdim = np.max(ndim)
                npad = maxdim - layer.data.ndim
                if npad > 0:
                    layer.data = np.expand_dims(layer.data, axis=tuple(range(npad)))
                    metadata['tmp_npad'] = npad
            else:
                # remove any temp dimensions added for transform mode
                if 'tmp_npad' in metadata:
                    npad = metadata['tmp_npad']
                    del metadata['tmp_npad']
                    layer.data = np.squeeze(layer.data, axis=tuple(range(npad)))
    
    def _on_dim_step_changed(self, event: Event):
        """ Callback for frame change event.
        """
        try:
            # [frame, row, col]
            frame = event.value[-3]
        except IndexError:
            frame = 0
        
        # update current frame vertical line in point projection plots
        for metadata in self._layer_metadata:
            if 'point_projection_vline' in metadata:
                vline = metadata['point_projection_vline']
                vline.setValue(frame)

    def _on_mouse_clicked_or_dragged(self, viewer: Viewer, event: Event):
        """ Callback for mouse press/drag/release events.
        """
        from qtpy.QtCore import Qt

        if viewer.layers.selection.active.mode != "pan_zoom":
            return
        
        # mouse press event
        # only process left-click events
        # ignore initial mouse press event (we'll use the mouse release event instead)
        if (event.type == 'mouse_press') and (event._button == Qt.MouseButton.LeftButton):
            yield
        else:
            return

        # mouse move event
        # if mouse dragged (beyond a tiny bit), ignore subsequent mouse release event
        # i.e., ignore click when dragging
        n_move_events = 0
        while event.type == 'mouse_move':
            n_move_events += 1
            if n_move_events <= 3:
                yield
            else:
                return
            
        # mouse release event
        # if we get here, then mouse was clicked without dragging (much)
        # i.e., process click event
        if event.type == 'mouse_release':
            mouse_worldpt2d = event.position[-2:]  # (row, col)
            
            # Find closest visible point to mouse click.
            # If mouse is within the point, then select the point for projection.
            # Ignore the layer for the selected projection point.
            visible_points_layers = [layer for layer in self._points_layers() if layer.visible and layer.data.size > 0]
            for layer in visible_points_layers:
                mouse_layerpt2d = self._transform_points2d_from_world_to_layer(mouse_worldpt2d, layer)
                layerpts2d = layer.data[:,-2:]
                radii = layer.size / 2
                square_dists = np.sum((layerpts2d - mouse_layerpt2d)**2, axis=1)
                layerpt_indexes = np.argsort(square_dists)
                for index in layerpt_indexes:
                    if square_dists[index] <= radii[index]**2:
                        self.select_projection_point(layer, index, ignore_tag_filter=True)
                        return
            
            # Did NOT click on an existing point.
            # self._projection_point_index: int | None = None
            self._projection_point_index_spinbox.clear()
            
            # Use the mouse location as the projection point.
            self.set_projection_point(mouse_worldpt2d)

    def _on_mouse_doubleclicked(self, viewer: Viewer, event: Event):
        """ Callabck for mouse double-click event.
        """
        # viewer.reset_view()
        pass

    def _apply_to_all_layers(self, func, *args, **kwargs):
        """ Apply a function to all layers.

        The function should take a layer as its first argument.
        """
        for layer in self._layers():
            func(layer, *args, **kwargs)
    
    def _apply_to_selected_layers(self, func, *args, **kwargs):
        """ Apply a function to selected layers.

        The function should take a layer as its first argument.
        """
        for layer in self._selected_layers():
            func(layer, *args, **kwargs)
    
    def _current_frame(self) -> int:
        """ Return the current frame index.
        """
        try:
            return self.viewer.dims.current_step[-3]
        except IndexError:
            return 0
    
    def _copy_active_layer_transform(self):
        """ Copy the active layer transform.
        """
        layer: Layer = self.viewer.layers.selection.active
        self._copied_layer_transform = layer.affine
    
    def _paste_copied_layer_transform(self, layer: Layer):
        """ Set the layer transform to the copied transform.
        """
        if hasattr(self, '_copied_layer_transform'):
            layer.affine = self._copied_layer_transform
    
    def _clear_layer_transform(self, layer: Layer):
        """ Clear the layer transform.
        """
        layer.affine = np.eye(layer.ndim)
    
    def _transform_points2d_from_layer_to_world(self, layerpts2d: np.ndarray, layer: Layer) -> np.ndarray:
        """ Transform points from layer to world coordinates.
        """
        layerpts2d = np.array(layerpts2d).reshape([-1, 2])
        if layer.ndim == 2:
            layerpts = layerpts2d
        else:
            n_points = layerpts2d.shape[0]
            layerpts = np.zeros([n_points, layer.ndim])
            layerpts[:,-2:] = layerpts2d
        worldpts2d = np.zeros(layerpts2d.shape)
        for i, layer_pt in enumerate(layerpts):
            worldpts2d[i] = layer.data_to_world(layer_pt)[-2:]
        return worldpts2d

    def _transform_points2d_from_world_to_layer(self, worldpts2d: np.ndarray, layer: Layer) -> np.ndarray:
        """ Transform points from world to layer coordinates.
        """
        worldpts2d = np.array(worldpts2d).reshape([-1, 2])
        if layer.ndim == 2:
            worldpts = worldpts2d
        else:
            n_points = worldpts2d.shape[0]
            worldpts = np.zeros([n_points, layer.ndim])
            worldpts[:,-2:] = worldpts2d
        layerpts2d = np.zeros(worldpts2d.shape)
        for i, world_pt in enumerate(worldpts):
            layerpts2d[i] = layer.world_to_data(world_pt)[-2:]
        return layerpts2d
    
    def _setup_ui(self):
        """ Setup the main widget UI.
        """
        self._setup_metadata_tab()
        self._setup_file_tab()
        self._setup_image_tab()
        self._setup_points_tab()
        self._setup_layer_registration_tab()

        # mimic layer insertion event in order to setup components for existing layers
        for i, layer in enumerate(self.viewer.layers):
            event = Event("", value=layer, index=i)
            self._on_layer_inserted(event)
    
    def _setup_metadata_tab(self, title: str = "Meta"):
        """ Metadata UI.

        Includes session date, ID, users, and notes.
        """
        from qtpy.QtWidgets import QFormLayout, QLineEdit, QTextEdit, QWidget

        self._date_edit = QLineEdit()
        self._id_edit = QLineEdit()
        self._users_edit = QLineEdit()
        self._notes_edit = QTextEdit()

        tab = QWidget()
        form = QFormLayout(tab)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.addRow("Date", self._date_edit)
        form.addRow("ID", self._id_edit)
        form.addRow("Users", self._users_edit)
        form.addRow("Notes", self._notes_edit)
        self.addTab(tab, title)

    def _setup_file_tab(self, title: str = "File"):
        """ File UI.

        Includes session import/export.
        """
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QWidget, QLabel

        msg = QLabel("!!! Session stores the relative path to each image stack file, NOT the data itself. It is up to you to maintain this file structure.\nSession .mat files can be accessed in MATLAB.")
        msg.setWordWrap(True)

        self._open_session_button = QPushButton("Open .mat session file")
        self._open_session_button.pressed.connect(self.import_session)

        self._save_session_button = QPushButton("Save session as .mat file")
        self._save_session_button.pressed.connect(self.export_session)

        inner = QVBoxLayout()
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(5)
        inner.addWidget(msg)
        inner.addSpacing(10)
        inner.addWidget(self._open_session_button)
        inner.addWidget(self._save_session_button)
        inner.addStretch()

        outer = QHBoxLayout()
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(5)
        outer.addLayout(inner)
        outer.addStretch()

        tab = QWidget()
        tab.setLayout(outer)
        self.addTab(tab, title)
    
    def _setup_image_tab(self, title: str = "Image"):
        """ Image UI.

        Includes image processing operations such as splitting, slicing, projecting, and filtering.
        """
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QFormLayout, QGroupBox, QPushButton, QComboBox, QDoubleSpinBox, QLineEdit, QLabel, QTabWidget

        self._split_image_button = QPushButton("Split Image")
        self._split_image_button.pressed.connect(lambda: self._apply_to_selected_layers(self.split_image_layer))

        self._split_image_regions_combobox = QComboBox()
        self._split_image_regions_combobox.addItems(["Top/Bottom", "Left/Right", "Quad"])
        self._split_image_regions_combobox.setCurrentText("Top/Bottom")

        self._slice_image_button = QPushButton("Slice Image")
        self._slice_image_button.pressed.connect(lambda: self._apply_to_selected_layers(self.slice_image_layer))

        self._slice_image_edit = QLineEdit()
        self._slice_image_edit.setPlaceholderText("start:stop[:step], for each dim...")

        self._project_image_button = QPushButton("Project Image")
        self._project_image_button.pressed.connect(lambda: self._apply_to_selected_layers(self.project_image_layer))

        self._project_image_operation_combobox = QComboBox()
        self._project_image_operation_combobox.addItems(["Max", "Min", "Std", "Sum", "Mean", "Median"])
        self._project_image_operation_combobox.setCurrentText("Mean")

        self._gaussian_filter_button = QPushButton("Gaussian Filter")
        self._gaussian_filter_button.pressed.connect(lambda: self._apply_to_selected_layers(self.filter_image_layer, filter_type="gauss"))

        self._gaussian_filter_sigma_spinbox = QDoubleSpinBox()
        self._gaussian_filter_sigma_spinbox.setValue(1)

        self._tophat_filter_button = QPushButton("Tophat Filter")
        self._tophat_filter_button.pressed.connect(lambda: self._apply_to_selected_layers(self.filter_image_layer, filter_type="tophat"))

        self._tophat_filter_disk_radius_spinbox = QDoubleSpinBox()
        self._tophat_filter_disk_radius_spinbox.setValue(3)

        tab = QTabWidget()
        for tab_title in ["Split/Slice", "Project", "Filter"]:
            msg = QLabel("Operations are applied to all selected image layers.\nResults are returned in new layers.")
            msg.setWordWrap(True)

            inner = QVBoxLayout()
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(5)
            inner.addWidget(msg)
            if tab_title == "Split/Slice":
                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow(self._split_image_button)
                form.addRow("Regions", self._split_image_regions_combobox)
                inner.addWidget(group)

                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
                form.addRow(self._slice_image_button)
                form.addRow("Slice", self._slice_image_edit)
                inner.addWidget(group)
            elif tab_title == "Project":
                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow(self._project_image_button)
                form.addRow("Projection", self._project_image_operation_combobox)
                inner.addWidget(group)
            elif tab_title == "Filter":
                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow(self._gaussian_filter_button)
                form.addRow("Sigma", self._gaussian_filter_sigma_spinbox)
                inner.addWidget(group)
                
                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow(self._tophat_filter_button)
                form.addRow("Disk radius", self._tophat_filter_disk_radius_spinbox)
                inner.addWidget(group)
            inner.addStretch()

            outer = QHBoxLayout()
            outer.setContentsMargins(5, 5, 5, 5)
            outer.setSpacing(5)
            outer.addLayout(inner)
            outer.addStretch()

            tab2 = QWidget()
            tab2.setLayout(outer)
            tab.addTab(tab2, tab_title)

        self.addTab(tab, title)
    
    def _setup_points_tab(self, title: str = "Points"):
        """ Points UI.

        Includes point settings, colocalization, and point projection.
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QFormLayout, QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox, QComboBox, QTabWidget, QGridLayout, QLabel, QCheckBox, QLineEdit

        self._default_point_size_spinbox = QDoubleSpinBox()
        self._default_point_size_spinbox.setValue(8)

        self._default_point_mask_grid = QGridLayout()
        self._default_point_mask_grid.setContentsMargins(5, 5, 5, 5)
        self._default_point_mask_grid.setSpacing(2)
        self._update_default_point_mask_grid()
        self._default_point_size_spinbox.valueChanged.connect(self._update_default_point_mask_grid)

        self._find_peaks_button = QPushButton("Find peaks in all selected image layers")
        self._find_peaks_button.pressed.connect(lambda: self._apply_to_selected_layers(self.find_image_peaks))

        self._min_peak_height_spinbox = QDoubleSpinBox()
        self._min_peak_height_spinbox.setMinimum(0)
        self._min_peak_height_spinbox.setMaximum(65000)
        self._min_peak_height_spinbox.setValue(10)

        self._min_peak_separation_spinbox = QDoubleSpinBox()
        self._min_peak_separation_spinbox.setMinimum(1)
        self._min_peak_separation_spinbox.setMaximum(65000)
        self._min_peak_separation_spinbox.setValue(self._default_point_size_spinbox.value())

        self._find_colocalized_points_button = QPushButton("Find colocalized points")
        self._find_colocalized_points_button.pressed.connect(self.find_colocalized_points)

        self._coloc_layer_combobox = QComboBox()
        self._coloc_layer_combobox.currentTextChanged.connect(lambda text: self._update_points_colocalization_plot())
        
        self._coloc_neighbors_layer_combobox = QComboBox()
        self._coloc_neighbors_layer_combobox.currentTextChanged.connect(lambda text: self._update_points_colocalization_plot())

        self._coloc_nearest_neighbor_cutoff_spinbox = QDoubleSpinBox()
        self._coloc_nearest_neighbor_cutoff_spinbox.setValue(self._default_point_size_spinbox.value() / 2)

        self._coloc_hist_binwidth_spinbox = QDoubleSpinBox()
        self._coloc_hist_binwidth_spinbox.setMinimum(1)
        self._coloc_hist_binwidth_spinbox.setValue(self._default_point_size_spinbox.value() / 2)
        self._coloc_hist_binwidth_spinbox.valueChanged.connect(lambda value: self._update_points_colocalization_plot())

        self._coloc_plot = self._new_plot()
        self._coloc_plot.setLabels(left="Counts", bottom="Nearest Neighbor Distance")
        legend = pg.LegendItem()
        legend.setParentItem(self._coloc_plot.getPlotItem())
        legend.anchor((1,0), (1,0))
        self._within_layers_nearest_neighbors_histogram = pg.PlotCurveItem([0, 0], [0], name="within layers", stepMode='center', pen=pg.mkPen([98, 143, 176, 80], width=1), fillLevel=0, brush=(98, 143, 176, 80))
        self._between_layers_nearest_neighbors_histogram = pg.PlotCurveItem([0, 0], [0], name="between layers", stepMode='center', pen=pg.mkPen([255, 0, 0, 80], width=1), fillLevel=0, brush=(255, 0, 0, 80))
        self._coloc_plot.addItem(self._within_layers_nearest_neighbors_histogram)
        self._coloc_plot.addItem(self._between_layers_nearest_neighbors_histogram)
        legend.addItem(self._within_layers_nearest_neighbors_histogram, "within layers")
        legend.addItem(self._between_layers_nearest_neighbors_histogram, "between layers")

        self._point_projection_plots_layout: QVBoxLayout = QVBoxLayout()
        self._point_projection_plots_layout.setContentsMargins(0, 0, 0, 0)
        self._point_projection_plots_layout.setSpacing(0)

        self._projection_points_layer_combobox = QComboBox()
        self._projection_points_layer_combobox.currentTextChanged.connect(lambda text: self.select_projection_point())

        self._projection_point_index_spinbox = QSpinBox()
        self._projection_point_index_spinbox.valueChanged.connect(lambda value: self.select_projection_point())

        self._projection_point_world_label = QLabel()
        self._n_projection_points_label = QLabel()
        
        self._tag_edit = QLineEdit()
        self._tag_edit.editingFinished.connect(self.set_point_tags)

        self._tag_filter_checkbox = QCheckBox("Tag filter")
        self._tag_filter_checkbox.stateChanged.connect(lambda state: self._update_tag_filter())

        self._tag_filter_edit = QLineEdit()
        self._tag_filter_edit.editingFinished.connect(self._update_tag_filter)

        self._projection_settings_button = QPushButton("Settings")
        self._projection_settings_button.pressed.connect(self._edit_projection_settings)

        tab = QTabWidget()
        for tab_title in ["Point", "Find", "Colocalize", "Projection"]:
            inner = QVBoxLayout()
            inner.setContentsMargins(0, 0, 0, 0)
            inner.setSpacing(5)
            if tab_title == "Point":
                msg = QLabel("Point projections use a circular mask centered on the point with pixel diameter equal to the selected point's size. For projecting an arbitrary position, the default size below will be used.")
                msg.setWordWrap(True)

                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow(msg)
                form.addRow("Default point size", self._default_point_size_spinbox)
                inner.addWidget(group)

                inner.addSpacing(10)
                inner.addWidget(QLabel("Default point projection mask"))
                inner.addLayout(self._default_point_mask_grid)
            elif tab_title == "Find":
                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow(self._find_peaks_button)
                form.addRow("Min peak height", self._min_peak_height_spinbox)
                form.addRow("Min peak separation", self._min_peak_separation_spinbox)
                inner.addWidget(group)
            elif tab_title == "Colocalize":
                grid = QGridLayout()
                grid.setContentsMargins(0, 0, 0, 0)
                grid.setSpacing(0)

                group = QGroupBox()
                form = QFormLayout(group)
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow(self._find_colocalized_points_button)
                form.addRow("Points layer", self._coloc_layer_combobox)
                form.addRow("Neighbors points layer", self._coloc_neighbors_layer_combobox)
                form.addRow("Nearest neighbor cutoff", self._coloc_nearest_neighbor_cutoff_spinbox)
                grid.addWidget(group, 0, 0)

                form = QFormLayout()
                form.setContentsMargins(5, 5, 5, 5)
                form.setSpacing(5)
                form.addRow("Histogram bin width", self._coloc_hist_binwidth_spinbox)
                grid.addLayout(form, 1, 0)
                grid.addWidget(self._coloc_plot, 2, 0, 1, 2)
                
                inner.addLayout(grid)
            elif tab_title == "Projection":
                grid = QGridLayout()
                grid.setContentsMargins(0, 0, 0, 0)
                grid.setSpacing(5)
                layer_label = QLabel("Points layer")
                layer_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                index_label = QLabel("Point index")
                index_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                tag_label = QLabel("Tags")
                tag_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                grid.addWidget(layer_label, 0, 0)
                grid.addWidget(self._projection_points_layer_combobox, 0, 1)
                grid.addWidget(self._n_projection_points_label, 0, 2)
                grid.addWidget(self._tag_filter_checkbox, 0, 3)
                grid.addWidget(self._tag_filter_edit, 0, 4)
                grid.addWidget(index_label, 1, 0)
                grid.addWidget(self._projection_point_index_spinbox, 1, 1)
                grid.addWidget(self._projection_point_world_label, 1, 2)
                grid.addWidget(tag_label, 1, 3)
                grid.addWidget(self._tag_edit, 1, 4)
                grid.addWidget(self._projection_settings_button, 1, 5)
                inner.addLayout(grid)
                inner.addLayout(self._point_projection_plots_layout)
            
            if tab_title == "Projection":
                tab2 = QWidget()
                tab2.setLayout(inner)
                tab.addTab(tab2, tab_title)
            else:
                inner.addStretch()

                outer = QHBoxLayout()
                outer.setContentsMargins(5, 5, 5, 5)
                outer.setSpacing(5)
                outer.addLayout(inner)
                outer.addStretch()

                tab2 = QWidget()
                tab2.setLayout(outer)
                tab.addTab(tab2, tab_title)

        self.addTab(tab, title)
    
    def _setup_layer_registration_tab(self, title: str = "Align"):
        """ Layer registration UI.
        """
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QFormLayout, QGroupBox, QComboBox, QPushButton, QLabel

        msg = QLabel("Registration sets the layer transform without altering the layer data.")
        msg.setWordWrap(True)

        self._fixed_layer_combobox = QComboBox()
        self._moving_layer_combobox = QComboBox()

        self._layer_transform_type_combobox = QComboBox()
        self._layer_transform_type_combobox.addItems(["Translation", "Rigid Body", "Affine"])
        self._layer_transform_type_combobox.setCurrentText("Affine")

        self._register_layers_button = QPushButton("Register Layers")
        self._register_layers_button.clicked.connect(lambda checked: self.register_layers())

        self._copy_layer_transform_button = QPushButton("Copy selected layer transform")
        self._copy_layer_transform_button.clicked.connect(lambda checked: self._copy_active_layer_transform())

        self._paste_layer_transform_button = QPushButton("Paste copied transform to all selected layers")
        self._paste_layer_transform_button.clicked.connect(lambda checked: self._apply_to_selected_layers(self._paste_copied_layer_transform))

        self._clear_layer_transform_button = QPushButton("Clear transform from all selected layers")
        self._clear_layer_transform_button.clicked.connect(lambda checked: self._apply_to_selected_layers(self._clear_layer_transform))

        layer_registration_group = QGroupBox()
        form = QFormLayout(layer_registration_group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._register_layers_button)
        form.addRow("Fixed Layer", self._fixed_layer_combobox)
        form.addRow("Moving Layer", self._moving_layer_combobox)
        form.addRow("Transform", self._layer_transform_type_combobox)

        inner = QVBoxLayout()
        inner.setContentsMargins(0, 0, 0, 0)
        inner.setSpacing(5)
        inner.addWidget(msg)
        inner.addWidget(layer_registration_group)
        inner.addWidget(self._copy_layer_transform_button)
        inner.addWidget(self._paste_layer_transform_button)
        inner.addWidget(self._clear_layer_transform_button)
        inner.addStretch()

        outer = QHBoxLayout()
        outer.setContentsMargins(5, 5, 5, 5)
        outer.setSpacing(5)
        outer.addLayout(inner)
        outer.addStretch()

        tab = QWidget()
        tab.setLayout(outer)
        self.addTab(tab, title)
    
    def _new_plot(self):
        """ Return new plot object with default settings.
        """
        from qtpy.QtGui import QColor
        from qtpy.QtCore import QSize

        pen = pg.mkPen(QColor.fromRgbF(0.15, 0.15, 0.15), width=1)
        plot = pg.PlotWidget(pen=pen)
        plot.setBackground(QColor(240, 240, 240))
        plot.showGrid(x=True, y=True, alpha=0.3)
        # hack to stop grid from clipping axis tick labels
        for axis in ['left', 'bottom']:
            plot.getAxis(axis).setGrid(False)
        for axis in ['right', 'top']:
            plot.getAxis(axis).setStyle(showValues=False)
            plot.showAxis(axis)
        for axis in ['left', 'bottom', 'right', 'top']:
            plot.getAxis(axis).setTextPen(QColor.fromRgbF(0.15, 0.15, 0.15))
        plot.getAxis('right').setWidth(10)
        plot.getAxis('top').setHeight(10)
        plot.setMinimumSize(QSize(50, 50))
        return plot
    
    def _update_layer_selection_comboboxes(self, changed_layer: Layer = None):
        """ Update all UI combo boxes for layer selection.
        """
        layer_names = [layer.name for layer in self._layers()]
        self._refresh_combobox(self._fixed_layer_combobox, layer_names)
        self._refresh_combobox(self._moving_layer_combobox, layer_names)

        if (changed_layer is None) or isinstance(changed_layer, Points):
            points_layer_names = [layer.name for layer in self._points_layers()]
            self._refresh_combobox(self._coloc_layer_combobox, points_layer_names)
            self._refresh_combobox(self._coloc_neighbors_layer_combobox, points_layer_names)
            self._refresh_combobox(self._projection_points_layer_combobox, points_layer_names)
    
    def _refresh_combobox(self, combobox: 'QComboBox', items: list[str]):
        """ Reset a combo box with new items.

        Keep previous selection if possible.
        """
        current_text = combobox.currentText()
        current_index = combobox.currentIndex()
        combobox.clear()
        if items:
            combobox.addItems(items)
            if current_text in items:
                combobox.setCurrentText(current_text)
            elif 0 <= current_index < len(items):
                combobox.setCurrentIndex(current_index)
    
    def _update_default_point_mask_grid(self):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QLabel

        clear_layout(self._default_point_mask_grid)
        point_size = self._default_point_size_spinbox.value()
        point_mask = make_point_mask(point_size, type='circle')
        for i, row in enumerate(point_mask):
            for j, value in enumerate(row):
                label = QLabel(str(float(value)))
                bg = int(255 * value)
                fg = 0 if bg >= 128 else 255
                label.setStyleSheet(f"QLabel {{ background-color : rgb({bg}, {bg}, {bg}); color : rgb({fg}, {fg}, {fg}); }}")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter)
                label.setFixedSize(30, 30)
                self._default_point_mask_grid.addWidget(label, i, j)
    
    def _update_points_colocalization_plot(self, layer: Points = None, neighbors_layer: Points = None, nearest_neighbor_cutoff: float = None, binwidth: float = None):
        """ Update within layer and between layers nearest neighbor histograms for selected points layers.
        """
        from scipy.spatial import distance

        if layer is None:
            layer_name = self._coloc_layer_combobox.currentText()
            try:
                layer = self.viewer.layers[layer_name]
            except KeyError:
                pass
        if neighbors_layer is None:
            neighbors_layer_name = self._coloc_neighbors_layer_combobox.currentText()
            try:
                neighbors_layer = self.viewer.layers[neighbors_layer_name]
            except KeyError:
                pass
        if neighbors_layer is layer:
            neighbors_layer = None
        if nearest_neighbor_cutoff is None:
            nearest_neighbor_cutoff = self._coloc_nearest_neighbor_cutoff_spinbox.value()
        if binwidth is None:
            binwidth = self._coloc_hist_binwidth_spinbox.value()
        
        if layer is not None:
            points = self._transform_points2d_from_layer_to_world(layer.data[:,-2:], layer)
            points_pairwise_distances = distance.squareform(distance.pdist(points))
            np.fill_diagonal(points_pairwise_distances, np.inf)
            points_nearest_neighbors = np.min(points_pairwise_distances, axis=1).flatten()
            points_nearest_neighbors = points_nearest_neighbors[~np.isinf(points_nearest_neighbors)]
        
        if neighbors_layer is not None:
            neighbors = self._transform_points2d_from_layer_to_world(neighbors_layer.data[:,-2:], neighbors_layer)
            neighbors_pairwise_distances = distance.squareform(distance.pdist(neighbors))
            np.fill_diagonal(neighbors_pairwise_distances, np.inf)
            neighbors_nearest_neighbors = np.min(neighbors_pairwise_distances, axis=1).flatten()
            neighbors_nearest_neighbors = neighbors_nearest_neighbors[~np.isinf(neighbors_nearest_neighbors)]

        if layer is not None and neighbors_layer is not None:
            within_layer_nearest_neighbors = np.concatenate([points_nearest_neighbors, neighbors_nearest_neighbors])
            bin_edges = np.arange(0, np.max(within_layer_nearest_neighbors) + binwidth, binwidth)
            counts, bin_edges = np.histogram(within_layer_nearest_neighbors, bins=bin_edges)
            self._within_layers_nearest_neighbors_histogram.setData(bin_edges, counts)

            between_layer_nearest_neighbors = np.min(np.linalg.norm(points[:, None, :] - neighbors[None, :, :], axis=-1), axis=1)
            bin_edges = np.arange(0, np.max(between_layer_nearest_neighbors) + binwidth, binwidth)
            counts, bin_edges = np.histogram(between_layer_nearest_neighbors, bins=bin_edges)
            self._between_layers_nearest_neighbors_histogram.setData(bin_edges, counts)
        elif layer is not None:
            bin_edges = np.arange(0, np.max(points_nearest_neighbors) + binwidth, binwidth)
            counts, bin_edges = np.histogram(points_nearest_neighbors, bins=bin_edges)
            self._within_layers_nearest_neighbors_histogram.setData(bin_edges, counts)
            self._between_layers_nearest_neighbors_histogram.setData([0, 0], [0])
        elif neighbors_layer is not None:
            bin_edges = np.arange(0, np.max(neighbors_nearest_neighbors) + binwidth, binwidth)
            counts, bin_edges = np.histogram(neighbors_nearest_neighbors, bins=bin_edges)
            self._within_layers_nearest_neighbors_histogram.setData(bin_edges, counts)
            self._between_layers_nearest_neighbors_histogram.setData([0, 0], [0])
        else:
            self._within_layers_nearest_neighbors_histogram.setData([0, 0], [0])
            self._between_layers_nearest_neighbors_histogram.setData([0, 0], [0])

    def _update_tag_filter(self):
        if self._tag_filter_checkbox.isChecked():
            self.select_projection_point()
    
    def _edit_projection_settings(self):
        from qtpy.QtWidgets import QDialog, QVBoxLayout, QGroupBox, QFormLayout, QDialogButtonBox, QSpinBox

        dlg = QDialog(self)
        dlg.setWindowTitle("Point Projection Settings")
        vbox = QVBoxLayout(dlg)

        group = QGroupBox("Sum Frames")
        form = QFormLayout(group)
        sum_frames_spinboxes = {}
        for layer, metadata in zip(reversed(self.viewer.layers), reversed(self._layer_metadata)):
            if isinstance(layer, Image) and layer.data.ndim == 3:
                spinbox = QSpinBox()
                spinbox.setMinimum(1)
                spinbox.setMaximum(layer.data.shape[-3])
                if 'projection-sum-frames' in metadata:
                    spinbox.setValue(metadata['projection-sum-frames'])
                form.addRow(layer.name, spinbox)
                sum_frames_spinboxes[layer.name] = spinbox
        vbox.addWidget(group)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        vbox.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return
        
        for layer, metadata in zip(reversed(self.viewer.layers), reversed(self._layer_metadata)):
            if isinstance(layer, Image) and layer.data.ndim == 3:
                spinbox = sum_frames_spinboxes[layer.name]
                if spinbox.value() != 1:
                    metadata['projection-sum-frames'] = spinbox.value()
                elif 'projection-sum-frames' in metadata:
                    del metadata['projection-sum-frames']
        
        self.update_point_projections()


def clear_layout(layout: 'qtpy.QtWidgets.QLayout'):
    from qtpy.QtWidgets import QLayout

    for i in reversed(range(layout.count())):
        item = layout.itemAt(i)
        if isinstance(item, QLayout):
            clear_layout(item)
        elif item.widget():
            item.widget().setParent(None)
        else:
            layout.removeItem(item)


def slice_from_str(slice_str: str) -> tuple[slice]:
    """ Convert string to slice.
    """
    if slice_str.strip() == "":
        return (slice(None),)
    slice_strs = [dim_slice_str.strip() for dim_slice_str in slice_str.split(',')]
    slices = []  # one slice per dimension
    for slice_str in slice_strs:
        slice_indexes = [int(idx) if len(idx.strip()) > 0 else None for idx in slice_str.split(':')]
        slices.append(slice(*slice_indexes))
    return tuple(slices)


def str_from_slice(slices: tuple[slice]) -> str:
    """ Convert slice to string.
    """
    slice_strs = []
    for dim_slice in slices:
        start = str(dim_slice.start) if dim_slice.start is not None else ""
        stop = str(dim_slice.stop) if dim_slice.stop is not None else ""
        step = str(dim_slice.step) if dim_slice.step is not None else ""
        slice_str = start + ':' + stop + ':' + step
        if slice_str.endswith(':'):
            slice_str = slice_str[:-1]
        slice_strs.append(slice_str)
    slice_str = ','.join(slice_strs)
    while slice_str.endswith(',:'):
        slice_str = slice_str[:-2]
    return slice_str


def combine_slices(parent_slice: tuple[slice], child_slice: tuple[slice], parent_shape: tuple[int]):
    """ Return net slice equivalent to two successive slices.
    """
    n_parent_slices = len(parent_slice)
    n_child_slices = len(child_slice)
    combined_slices = []
    for i in range(min(n_parent_slices, n_child_slices)):
        parent_start = parent_slice[i].start if parent_slice[i].start is not None else 0
        # parent_stop = parent_slices[i].stop if parent_slices[i].stop is not None else parent_shape[i]
        parent_step = parent_slice[i].step if parent_slice[i].step is not None else 1
        child_start = child_slice[i].start if child_slice[i].start is not None else 0
        child_stop = child_slice[i].stop if child_slice[i].stop is not None else parent_shape[i]
        child_step = child_slice[i].step if child_slice[i].step is not None else 1
        start = parent_start + child_start
        stop = parent_start + child_stop
        step = parent_step * child_step
        if start == 0:
            start = None
        if stop == parent_shape[i]:
            stop = None
        if step == 1:
            step = None
        combined_slices.append(slice(start, stop, step))
    if n_child_slices > n_parent_slices:
        combined_slices.extend(child_slice[n_parent_slices:])
    return combined_slices


def normalize_image(image: np.ndarray, limits: tuple[float] = None) -> np.ndarray:
    """ Return normalized image in [0.0, 1.0].
    """
    image = image.astype(float)
    if limits is None:
        cmin, cmax = image.min(), image.max()
    else:
        cmin, cmax = limits
    image -= cmin
    image /= cmax
    image[image < 0] = 0
    image[image > 1] = 1
    return image


def register_images(fixed_image: np.ndarray, moving_image: np.ndarray, transform_type: str = "affine") -> np.ndarray:
    """ Return transformation that aligns moving image to fixed image.

    Uses pystackreg for image registration.
    """
    from pystackreg import StackReg

    transform_types = {
        "translation": StackReg.TRANSLATION,
        "rigid body": StackReg.RIGID_BODY,
        "affine": StackReg.AFFINE,
    }
    type = transform_types[transform_type]
    sreg = StackReg(type)
    tform = sreg.register(fixed_image, moving_image)
    return tform


def register_points(fixed_points: np.ndarray, moving_points: np.ndarray, transform_type: str = "affine") -> np.ndarray:
    """ Return transformation that aligns moving points to fixed points.

    Uses pycpd for point registration.
    """
    if transform_type == "rigid body":
        from pycpd import RigidRegistration
        reg = RigidRegistration(X=fixed_points, Y=moving_points)
        registered_points, (scale, rotate, translate) = reg.register()
        tform = np.eye(3)
        tform[:2,:2] = rotate
        tform[:2,-1] = translate
        # TODO: scale?
        return tform
    
    if transform_type == "affine":
        from pycpd import AffineRegistration
        reg = AffineRegistration(X=fixed_points, Y=moving_points)
        registered_points, (affine, translate) = reg.register()
        tform = np.eye(3)
        tform[:2,:2] = affine
        tform[:2,-1] = translate
        return tform


def find_image_peaks(image: np.ndarray, min_peak_height: float = None, min_peak_separation: float = 3) -> np.ndarray:
    """ Return position of local peaks in image.
    """
    from skimage import morphology, measure

    pixel_radius = max(1, np.ceil(min_peak_separation / 2))
    if min_peak_height is None:
        peak_mask = morphology.local_maxima(image, connectivity=pixel_radius, indices=False, allow_borders=False)
    else:
        disk = morphology.disk(pixel_radius)
        peak_mask = morphology.h_maxima(image, h=min_peak_height, footprint=disk) > 0
        peak_mask[:,0] = False
        peak_mask[0,:] = False
        peak_mask[:,-1] = False
        peak_mask[-1,:] = False
    label_image = measure.label(peak_mask)
    rois = measure.regionprops(label_image)
    n_rois = len(rois)
    points = np.zeros((n_rois, 2))
    for i, roi in enumerate(rois):
        points[i] = roi.centroid
    return points


def find_colocalized_points(points: np.ndarray, neighbors: np.ndarray, nearest_neighbor_cutoff: float) -> np.ndarray:
    """ Return colocalized points between two input points sets.

    For each pair of colocalized points, their mean position is returned.
    """
    pairwise_distances = np.linalg.norm(points[:, None, :] - neighbors[None, :, :], axis=-1)
    nearest_neighbor_distances = pairwise_distances.min(axis=1)
    points_indices = np.where(nearest_neighbor_distances <= nearest_neighbor_cutoff)[0]
    n_points = points_indices.size
    if n_points == 0:
        return np.array([])
    neighbors_indices = pairwise_distances[points_indices].reshape([n_points,-1]).argmin(axis=1)
    colocalized_points = (points[points_indices] + neighbors[neighbors_indices]) / 2
    return colocalized_points


def project_image_point(image: np.ndarray, point2d, point_mask2d: np.ndarray = None) -> np.ndarray:
    """ Return the point projection for the input image.

    If a mask is given, the projection is the per-frame mean of the pixels in the mask scaled by the mask.
    """
    # project single pixel
    if (point_mask2d is None) or np.all(point_mask2d.shape == 1):
        row, col = np.round(point2d).astype(int).flatten()
        try:
            return np.squeeze(image[...,row,col])
        except IndexError:
            return np.array([])
    
    # project mean of pixels in mask
    row, col = point2d.flatten()
    h, w = point_mask2d.shape
    rows = np.round(row - h/2 + np.arange(h)).astype(int)
    cols = np.round(col - w/2 + np.arange(w)).astype(int)
    i = np.where((rows >= 0) & (rows < image.shape[-2]))[0]
    j = np.where((cols >= 0) & (cols < image.shape[-1]))[0]
    if i.size == 0 or j.size == 0:
        return np.array([])
    rows = rows[i].reshape([-1,1])
    cols = cols[j].reshape([1,-1])
    mask = point_mask2d[i.reshape([-1,1]),j.reshape([1,-1])]
    mask = mask.reshape((1,) * (image.ndim - 2) + mask.shape)
    return np.squeeze(np.mean(image[...,rows,cols] * mask, axis=(-2, -1)))


def make_point_mask(size: int | float, type: str = 'circle'):
    if type == 'circle':
        radius = size / 2
        max_pixel_offset = (size - 1) / 2
        pixel_offsets = np.arange(-max_pixel_offset, max_pixel_offset + 1, dtype=float)
        mask = (pixel_offsets.reshape([-1,1])**2 + pixel_offsets.reshape([1,-1])**2) <= radius**2
        return mask
    
    if type == 'square':
        return np.ones((size, size), dtype=bool)


if __name__ == "__main__":
    """ This is mostly for debugging.
    
    Typically, the plugin is run from the plugin menu in the napari viewer.
    """
    import napari

    viewer = Viewer()
    plugin = MainWidget(viewer)
    viewer.window.add_dock_widget(plugin, name='napari-cosmos-ts', area='right')

    viewer.add_image(
        np.random.random([1000, 512, 512]),
        contrast_limits=[0.8, 1],
    )

    viewer.add_image(
        np.random.random([1000, 512, 512]),
        contrast_limits=[0.8, 1],
    )

    viewer.add_image(
        np.random.random([512, 512]),
        contrast_limits=[0.8, 1],
    )

    n_points = 100
    viewer.add_points(
        np.random.random([n_points, 2]) * 512,
        symbol = "disc",
        size = [8] * n_points,
        face_color = [[1, 1, 0, 0.1]] * n_points,
        edge_color = [[1, 1, 0, 1]] * n_points,
        opacity = 1,
        blending = "translucent_no_depth",
    )

    n_points = 50
    viewer.add_points(
        np.random.random([n_points, 2]) * 512,
        symbol = "disc",
        size = [8] * n_points,
        face_color = [[1, 1, 0, 0.1]] * n_points,
        edge_color = [[1, 1, 0, 1]] * n_points,
        opacity = 1,
        blending = "translucent_no_depth",
    )

    napari.run()