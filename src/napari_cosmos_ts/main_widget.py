

import os
import numpy as np
import napari
from napari.viewer import Viewer
from napari.layers import Layer, Image, Points
from napari.utils.events import Event
from qtpy.QtWidgets import QTabWidget
import pyqtgraph as pg


class MainWidget(QTabWidget):
    """ 
    """
    
    def __init__(self, viewer: Viewer, parent=None):
        QTabWidget.__init__(self, parent=parent)
        self.viewer: Viewer = viewer

        # Layer metadata to be stored separately from layers.
        # This keeps non-serealizable objects (e.g., QObjects) out of the layer metadata dicts.
        # Data that can be serialized and should be saved should be stored in the individual layer metadata dicts.
        self._layer_metadata: list[dict] = []

        # UI
        self._setup_ui()

        # event handling
        self.viewer.layers.events.inserted.connect(self._on_layer_inserted)
        self.viewer.layers.events.removed.connect(self._on_layer_removed)
        self.viewer.layers.events.moved.connect(self._on_layer_moved)
        self.viewer.dims.events.current_step.connect(self._on_dim_step_changed)
        self.viewer.mouse_drag_callbacks.append(self._on_mouse_clicked_or_dragged)
        self.viewer.mouse_double_click_callbacks.append(self._on_mouse_doubleclicked)
    
    @property
    def selected_point_layer(self) -> Points | None:
        return getattr(self, '_selected_point_layer', None)
    
    @selected_point_layer.setter
    def selected_point_layer(self, layer: Points | None):
        self._selected_point_layer = layer
    
    def export_session(self, filepath: str = None):
        """ Export data to MATLAB .mat file.
        """
        from scipy.io import savemat

        if filepath is None:
            from qtpy.QtWidgets import QFileDialog
            filepath, _filter = QFileDialog.getSaveFileName(self, "Save data in MATLAB file format.", "", "MATLAB files (*.mat)")
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
        
        # layer data
        session['layers'] = []
        for layer in self.viewer.layers:
            layer_data = {}
            layer_data['name'] = layer.name
            layer_data['affine'] = layer.affine.affine_matrix
            layer_data['opacity'] = layer.opacity
            layer_data['blending'] = layer.blending
            layer_data['visible'] = layer.visible

            # image layer
            if isinstance(layer, Image):
                layer_data['type'] = 'image'
                image_abspath = self._image_layer_abspath(layer)
                if image_abspath is not None:
                    image_relpath = os.path.relpath(image_abspath, start=session_absdir)
                    layer_data['file_abspath'] = image_abspath
                    layer_data['file_relpath'] = image_relpath
                if image_abspath is None:
                    # store image data if it does not already exist on disk
                    layer_data['image'] = layer.data
                if 'image' not in layer_data:
                    # if image data is not stored in the session, store shape and dtype
                    layer_data['image_shape'] = layer.data.shape
                    layer_data['image_dtype'] = str(layer.data.dtype)
                layer_data['contrast_limits'] = layer.contrast_limits
                layer_data['gamma'] = layer.gamma
                layer_data['colormap'] = layer.colormap.name
                layer_data['interpolation2d'] = layer.interpolation2d
            
            # points layer
            elif isinstance(layer, Points):
                layer_data['type'] = 'points'
                layer_data['points'] = layer.data
                layer_data['size'] = layer.size
                layer_data['symbol'] = layer.symbol
                if not layer.features.empty:
                    layer_data['features'] = {}
                    for key in layer.features:
                        if key == 'tags':
                            layer.features['tags'].fillna("", inplace=True)
                            layer.features['tags'].replace("", " ", inplace=True)
                        layer_data['features'][key] = layer.features[key].to_numpy()
                layer_data['face_color'] = layer.face_color
                layer_data['edge_color'] = layer.edge_color
                layer_data['edge_width'] = layer.edge_width
                layer_data['edge_width_is_relative'] = layer.edge_width_is_relative
            
            # layer metadata
            if layer.metadata:
                layer_data['metadata'] = layer.metadata
            
            # add layer data to session
            session['layers'].append(layer_data)
        
        # roiLayer = self.selectedRoisLayer()
        # roiIndex = self.selectedRoiIndex()
        # if roiLayer is not None and roiIndex is not None:
        #     mdict['selected_roi_layer'] = roiLayer.name
        #     mdict['selected_roi_index'] = roiIndex
        
        # save session to .mat file
        savemat(filepath, session)
    
    def import_session(self, filepath: str = None):
        """ Import data from MATLAB .mat file.
        """
        pass # TODO
    
    def split_image_layer(self, layer: Image, regions: str = None):
        """
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
        """
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
        """
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
        """
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
        """
        """
        from qtpy.QtWidgets import QMessageBox

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
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Only image-image or points-points layer registration implemented.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()

    def register_image_layers(self, fixed_layer: Image, moving_layer: Image, transform_type: str = "affine"):
        """
        """
        try:
            from pystackreg import StackReg
        except ImportError:
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Image registration requires pystackreg package.")
            msg.setStandardButtons(QMessageBox.Close)
            msg.exec()
            return
        
        # get current image or frame if layer is an image stack
        fixed_image = fixed_layer.data[...,:,:]
        moving_image = moving_layer.data[...,:,:]

        # adjust image to match layer contrast limits
        fixed_image = normalize_image(fixed_image, fixed_layer.contrast_limits)
        moving_image = normalize_image(moving_image, moving_layer.contrast_limits)
        
        # register images
        tform = register_images(fixed_image, moving_image, transform_type)
        
        # apply net world transform to moving image
        moving_layer.affine = tform @ fixed_layer.affine.affine_matrix[-3:,-3:]

    def register_points_layers(self, fixed_layer: Points, moving_layer: Points, transform_type: str = "affine"):
        """
        """
        try:
            from pycpd import RigidRegistration, AffineRegistration
        except ImportError:
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
    
    def set_projection_point(self, worldpt2d: np.ndarray = None, layer: Points = None, point_index: int = None):
        """ 
        """
        if worldpt2d is None:
            if layer is not None and point_index is not None:
                layerpt2d = layer.data[point_index,-2:]
                worldpt2d = self._transform_points2d_from_layer_to_world(layerpt2d, layer)
        
        if worldpt2d is None:
            # clear selected projection point overlay
            if self.selected_point_layer is not None:
                self.viewer.layers.remove(self.selected_point_layer)
                self.selected_point_layer = None
            return
        
        # update selected projection point overlay
        if self.selected_point_layer is None:
            self.selected_point_layer = self.viewer.add_points(
                worldpt2d,
                name = "selected point",
                # size = self._default_point_size_spinbox.value(),
                symbol = "disc",
                face_color = "cyan",
                edge_color = "cyan",
                # edge_width = self._default_point_edgewidth_spinbox.value(),
                # edge_width_is_relative = False,
                opacity = 0.7,
                blending = "translucent_no_depth",
            )
        else:
            self.selected_point_layer.data = worldpt2d
    
    def _layers(self, include_selected_point_layer: bool = False):
        layers = [layer for layer in reversed(self.viewer.layers)]
        if not include_selected_point_layer:
            if self.selected_point_layer in layers:
                layers.remove(self.selected_point_layer)
        return layers
    
    def _image_layers(self):
        return [layer for layer in reversed(self.viewer.layers) if isinstance(layer, Image)]
    
    def _imagestack_layers(self):
        return [layer for layer in reversed(self.viewer.layers) if isinstance(layer, Image) and layer.data.ndim > 2]
    
    def _points_layers(self, include_selected_point_layer: bool = False):
        layers = [layer for layer in reversed(self.viewer.layers) if isinstance(layer, Points)]
        if not include_selected_point_layer:
            if self.selected_point_layer in layers:
                layers.remove(self.selected_point_layer)
        return layers
    
    def _image_layer_abspath(self, layer: Image) -> str | None:
        """
        """
        abspath = os.path.abspath(layer.source.path)
        if abspath is None:
            if 'image_absfilepath' in layer.metadata:
                abspath = os.path.abspath(layer.metadata['image_absfilepath'])
        return abspath
    
    def _on_layer_inserted(self, event: Event):
        """ 
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
        """ 
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
        if self.selected_point_layer is layer:
            self.selected_point_layer = None

    def _on_layer_moved(self, event: Event):
        """ 
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
        """ 
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
        """ 
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
        """ 
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
        """ 
        """
        if viewer.layers.selection.active.mode != "pan_zoom":
            return
        
        # mouse press event
        # ignore initial mouse press event (we'll use the mouse release event instead)
        if event.type == 'mouse_press':
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
            visible_points_layers = [layer for layer in reversed(self.viewer.layers) if isinstance(layer, Points) and layer.visible and layer.data.size > 0]
            ignore_layer = getattr(self, '_selected_point_overlay_layer', None)
            if ignore_layer in visible_points_layers:
                visible_points_layers.remove(ignore_layer)
            for layer in visible_points_layers:
                mouse_layerpt2d = self._transform_points2d_from_world_to_layer(mouse_worldpt2d, layer)
                layerpts2d = layer.data[:,-2:]
                layerptsizes = layer.size
                square_dists = np.sum((layerpts2d - mouse_layerpt2d)**2, axis=1)
                layerpt_indexes = np.argsort(square_dists)
                for index in layerpt_indexes:
                    if square_dists[index] <= layerptsizes[index]**2:
                        self.set_projection_point(layer=layer, point_index=index)
                        return
            
            # Use the mouse location as the projection point.
            self.set_projection_point(worldpt2d=mouse_worldpt2d)

    def _on_mouse_doubleclicked(self, viewer: Viewer, event: Event):
        """ 
        """
        # viewer.reset_view()
        pass

    def _apply_to_all_layers(self, func, *args, **kwargs):
        layers = list(self.viewer.layers)
        for layer in layers:
            func(layer, *args, **kwargs)
    
    def _apply_to_selected_layers(self, func, *args, **kwargs):
        layers = list(self.viewer.layers.selection)
        for layer in layers:
            func(layer, *args, **kwargs)
    
    # def _update_point_projection_plots(self):
    #     """ 
    #     """
    #     plots = [metadata['point_projection_plot'] for metadata in self._layer_metadata if 'point_projection_plot' in metadata]
        
    #     # xlink
    #     if plots:
    #         plots[0].setXLink(None)
    #     for i in range(1, len(plots)):
    #         plots[i].setXLink(plots[0])
    
    def _current_frame(self) -> int:
        """ 
        """
        try:
            return self.viewer.dims.current_step[-3]
        except IndexError:
            return 0
    
    def _copy_active_layer_transform(self):
        """
        """
        layer: Layer = self.viewer.layers.selection.active
        self._copied_layer_transform = layer.affine
    
    def _paste_copied_layer_transform(self, layer: Layer):
        """
        """
        if hasattr(self, '_copied_layer_transform'):
            layer.affine = self._copied_layer_transform
    
    def _clear_layer_transform(self, layer: Layer):
        """
        """
        layer.affine = np.eye(layer.ndim)
    
    def _transform_points2d_from_layer_to_world(self, layerpts2d: np.ndarray, layer: Layer) -> np.ndarray:
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
        """ 
        """
        self._setup_metadata_tab()
        self._setup_file_tab()
        self._setup_image_processing_tab()
        self._setup_layer_registration_tab()
        self._setup_points_tab()
        self._setup_point_projections_tab()

        # mimic layer insertion event in order to setup components for existing layers
        for i, layer in enumerate(self.viewer.layers):
            event = Event("", value=layer, index=i)
            self._on_layer_inserted(event)
    
    def _setup_metadata_tab(self, title: str = "Meta"):
        from qtpy.QtWidgets import QFormLayout, QLineEdit, QTextEdit, QWidget

        self._date_edit = QLineEdit()
        self._id_edit = QLineEdit()
        self._users_edit = QLineEdit()
        self._notes_edit = QTextEdit()

        tab = QWidget()
        tab_layout = QFormLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        tab_layout.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        tab_layout.addRow("Date", self._date_edit)
        tab_layout.addRow("ID", self._id_edit)
        tab_layout.addRow("User(s)", self._users_edit)
        tab_layout.addRow("Notes", self._notes_edit)
        self.addTab(tab, title)

    def _setup_file_tab(self, title: str = "File"):
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QPushButton, QWidget

        self._open_session_button = QPushButton("Open .mat session file")
        self._open_session_button.clicked.connect(lambda checked: self.import_session())

        self._save_session_button = QPushButton("Save session as .mat file")
        self._save_session_button.clicked.connect(lambda checked: self.export_session())

        tab_inner_layout = QVBoxLayout()
        tab_inner_layout.setContentsMargins(0, 0, 0, 0)
        tab_inner_layout.setSpacing(5)
        tab_inner_layout.addWidget(self._open_session_button)
        tab_inner_layout.addWidget(self._save_session_button)
        tab_inner_layout.addStretch()

        tab = QWidget()
        tab_layout = QHBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        tab_layout.addLayout(tab_inner_layout)
        tab_layout.addStretch()
        self.addTab(tab, title)
    
    def _setup_image_processing_tab(self, title: str = "Image"):
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QFormLayout, QGroupBox, QPushButton, QComboBox, QDoubleSpinBox, QLineEdit, QLabel

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

        usage_message = QLabel("Operations are applied to all selected image layers.\nResults are returned in new layers.")
        usage_message.setWordWrap(True)

        split_image_group = QGroupBox()
        form = QFormLayout(split_image_group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._split_image_button)
        form.addRow("regions", self._split_image_regions_combobox)

        slice_image_group = QGroupBox()
        form = QFormLayout(slice_image_group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._slice_image_button)
        form.addRow("slice", self._slice_image_edit)

        project_image_group = QGroupBox()
        form = QFormLayout(project_image_group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._project_image_button)
        form.addRow("projection", self._project_image_operation_combobox)

        gaussian_filter_group = QGroupBox()
        form = QFormLayout(gaussian_filter_group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._gaussian_filter_button)
        form.addRow("sigma", self._gaussian_filter_sigma_spinbox)

        tophat_filter_group = QGroupBox()
        form = QFormLayout(tophat_filter_group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._tophat_filter_button)
        form.addRow("disk radius", self._tophat_filter_disk_radius_spinbox)

        tab_inner_layout = QVBoxLayout()
        tab_inner_layout.setContentsMargins(0, 0, 0, 0)
        tab_inner_layout.setSpacing(5)
        tab_inner_layout.addWidget(usage_message)
        tab_inner_layout.addWidget(split_image_group)
        tab_inner_layout.addWidget(slice_image_group)
        tab_inner_layout.addWidget(project_image_group)
        tab_inner_layout.addWidget(gaussian_filter_group)
        tab_inner_layout.addWidget(tophat_filter_group)
        tab_inner_layout.addStretch()

        tab = QWidget()
        tab_layout = QHBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        tab_layout.addLayout(tab_inner_layout)
        tab_layout.addStretch()
        self.addTab(tab, title)
    
    def _setup_layer_registration_tab(self, title: str = "Align"):
        """
        """
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QFormLayout, QGroupBox, QComboBox, QPushButton, QLabel

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

        user_message = QLabel("Registration sets the layer affine transform without altering the layer data.")
        user_message.setWordWrap(True)

        layer_registration_group = QGroupBox()
        form = QFormLayout(layer_registration_group)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._register_layers_button)
        form.addRow("fixed Layer", self._fixed_layer_combobox)
        form.addRow("moving Layer", self._moving_layer_combobox)
        form.addRow("transform", self._layer_transform_type_combobox)

        tab_inner_layout = QVBoxLayout()
        tab_inner_layout.setContentsMargins(0, 0, 0, 0)
        tab_inner_layout.setSpacing(5)
        tab_inner_layout.addWidget(user_message)
        tab_inner_layout.addWidget(layer_registration_group)
        tab_inner_layout.addWidget(self._copy_layer_transform_button)
        tab_inner_layout.addWidget(self._paste_layer_transform_button)
        tab_inner_layout.addWidget(self._clear_layer_transform_button)
        tab_inner_layout.addStretch()

        tab = QWidget()
        tab_layout = QHBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        tab_layout.addLayout(tab_inner_layout)
        tab_layout.addStretch()
        self.addTab(tab, title)
    
    def _setup_points_tab(self, title: str = "Points"):
        """
        """
        from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget, QFormLayout, QGroupBox, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox

        # self._default_point_size_spinbox = QSpinBox()
        # self._default_point_size_spinbox.setValue(8)

        # self._default_point_edgewidth_spinbox = QDoubleSpinBox()
        # self._default_point_edgewidth_spinbox.setValue(1)
        # self._default_point_edgewidth_spinbox.setSingleStep(0.25)
        # self._default_point_edgewidth_spinbox.setDecimals(2)

        # self._set_selected_point_edgewidth_button = QPushButton("Set edge width of selected points to default")

        self._find_peaks_button = QPushButton("Find peaks in all selected image layers")
        self._find_peaks_button.clicked.connect(lambda checked: self._apply_to_selected_layers(self.find_image_peaks))

        self._min_peak_height_spinbox = QDoubleSpinBox()
        self._min_peak_height_spinbox.setMinimum(0)
        self._min_peak_height_spinbox.setMaximum(65000)
        self._min_peak_height_spinbox.setValue(10)

        self._min_peak_separation_spinbox = QDoubleSpinBox()
        self._min_peak_separation_spinbox.setMinimum(1)
        self._min_peak_separation_spinbox.setMaximum(65000)
        self._min_peak_separation_spinbox.setValue(5)

        self._find_colocalized_points_button = QPushButton("Find colocalized points")
        self._find_colocalized_points_button.clicked.connect(lambda checked: self.find_colocalized_points())

        self._coloc_layer_combobox = QComboBox()
        # self._coloc_layer_combobox.currentTextChanged.connect(self.updateRoisColocalizationPlot)
        
        self._coloc_neighbors_layer_combobox = QComboBox()
        # self._coloc_neighbors_layer_combobox.currentTextChanged.connect(self.updateRoisColocalizationPlot)

        self._coloc_nearest_neighbor_cutoff_spinbox = QDoubleSpinBox()
        self._coloc_nearest_neighbor_cutoff_spinbox.setMinimum(0)
        self._coloc_nearest_neighbor_cutoff_spinbox.setMaximum(1000)
        self._coloc_nearest_neighbor_cutoff_spinbox.setSingleStep(0.5)
        self._coloc_nearest_neighbor_cutoff_spinbox.setDecimals(2)
        self._coloc_nearest_neighbor_cutoff_spinbox.setValue(5)

        self._coloc_plot = self._new_plot()
        self._coloc_plot.setLabels(left="Counts", bottom="Nearest Neighbor Distance")
        legend = pg.LegendItem()
        legend.setParentItem(self._coloc_plot.getPlotItem())
        legend.anchor((1,0), (1,0))
        self._within_layers_nearest_neighbors_histogram = pg.PlotCurveItem([0, 0], [0], name="within layers", 
            stepMode='center', pen=pg.mkPen([98, 143, 176, 80], width=1), fillLevel=0, brush=(98, 143, 176, 80))
        self._between_layers_nearest_neighbors_histogram = pg.PlotCurveItem([0, 0], [0], name="between layers", 
            stepMode='center', pen=pg.mkPen([255, 0, 0, 80], width=1), fillLevel=0, brush=(255, 0, 0, 80))
        self._coloc_plot.addItem(self._within_layers_nearest_neighbors_histogram)
        self._coloc_plot.addItem(self._between_layers_nearest_neighbors_histogram)
        legend.addItem(self._within_layers_nearest_neighbors_histogram, "within layers")
        legend.addItem(self._between_layers_nearest_neighbors_histogram, "between layers")

        # points_group = QGroupBox()
        # form = QFormLayout(points_group)
        # form.setContentsMargins(5, 5, 5, 5)
        # form.setSpacing(5)
        # form.addRow("Default point size", self._default_point_size_spinbox)
        # form.addRow("Default point edge width", self._default_point_edgewidth_spinbox)
        # form.addRow(self._set_selected_point_edgewidth_button)

        find_peaks_group = QGroupBox()
        form = QFormLayout(find_peaks_group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._find_peaks_button)
        form.addRow("Min peak height", self._min_peak_height_spinbox)
        form.addRow("Min separation", self._min_peak_separation_spinbox)

        colocalize_group = QGroupBox()
        form = QFormLayout(colocalize_group)
        form.setContentsMargins(5, 5, 5, 5)
        form.setSpacing(5)
        form.addRow(self._find_colocalized_points_button)
        form.addRow("Points layer", self._coloc_layer_combobox)
        form.addRow("Neighbors points layer", self._coloc_neighbors_layer_combobox)
        form.addRow("Nearest neighbor cutoff", self._coloc_nearest_neighbor_cutoff_spinbox)
        form.addRow(self._coloc_plot)

        tab_inner_layout = QVBoxLayout()
        tab_inner_layout.setContentsMargins(0, 0, 0, 0)
        tab_inner_layout.setSpacing(5)
        # tab_inner_layout.addWidget(points_group)
        tab_inner_layout.addWidget(find_peaks_group)
        tab_inner_layout.addWidget(colocalize_group)
        tab_inner_layout.addStretch()

        tab = QWidget()
        tab_layout = QHBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        tab_layout.addLayout(tab_inner_layout)
        tab_layout.addStretch()
        self.addTab(tab, title)
    
    def _setup_point_projections_tab(self, title: str = "Point Z-Proj"):
        """ 
        """
        from qtpy.QtWidgets import QVBoxLayout, QWidget

        self._point_projection_plots_layout: QVBoxLayout = QVBoxLayout()
        self._point_projection_plots_layout.setContentsMargins(0, 0, 0, 0)
        self._point_projection_plots_layout.setSpacing(0)

        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(5, 5, 5, 5)
        tab_layout.setSpacing(5)
        tab_layout.addLayout(self._point_projection_plots_layout)
        self.addTab(tab, title)
    
    def _new_plot(self):
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
        layer_names = [layer.name for layer in self._layers()]
        self._refresh_combobox(self._fixed_layer_combobox, layer_names)
        self._refresh_combobox(self._moving_layer_combobox, layer_names)

        if (changed_layer is None) or isinstance(changed_layer, Points):
            points_layer_names = [layer.name for layer in self._points_layers()]
            self._refresh_combobox(self._coloc_layer_combobox, points_layer_names)
            self._refresh_combobox(self._coloc_neighbors_layer_combobox, points_layer_names)
    
    def _refresh_combobox(self, combobox: 'QComboBox', items: list[str]):
        current_text = combobox.currentText()
        current_index = combobox.currentIndex()
        combobox.clear()
        if items:
            combobox.addItems(items)
            if current_text in items:
                combobox.setCurrentText(current_text)
            elif 0 <= current_index < len(items):
                combobox.setCurrentIndex(current_index)


def slice_from_str(slice_str: str) -> tuple[slice]:
    if slice_str.strip() == "":
        return (slice(None),)
    slice_strs = [dim_slice_str.strip() for dim_slice_str in slice_str.split(',')]
    slices = []  # one slice per dimension
    for slice_str in slice_strs:
        slice_indexes = [int(idx) if len(idx.strip()) > 0 else None for idx in slice_str.split(':')]
        slices.append(slice(*slice_indexes))
    return tuple(slices)


def str_from_slice(slices: tuple[slice]) -> str:
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


if __name__ == "__main__":
    import napari

    viewer = Viewer()

    plugin = MainWidget(viewer)
    viewer.window.add_dock_widget(plugin, name='CoSMoS-TS', area='right')

    # from aicsimageio import AICSImage
    # from aicsimageio.readers import BioformatsReader
    # fp = "/Users/marcel/Downloads/img/2019-08-22 Tax4-GFP posA-3 10nM fcGMP to fcGMP+10uM cGMP ex532nm60mW100ms.tif"
    # fp = "/Users/marcel/Downloads/img/ZMW_loc17_fcGFP_532nm_80mW_1_MMStack_Pos0.ome.tif"
    # fp = "/Users/marcel/Downloads/img/fcGFP_637nm_60mW_1_MMStack_Default.ome.tif"
    # # img = AICSImage(fp, reader=BioformatsReader)
    # img = BioformatsReader(fp)
    # img = tifffile.memmap(fp)
    # img = viewer.open(path=fp, layer_type="image")[0].data
    # print(img.shape)
    viewer.add_image(np.random.random([1000, 512, 512]))
    # viewer.open(path=fp, layer_type="image")

    napari.run()