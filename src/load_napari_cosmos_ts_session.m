function data = load_napari_cosmos_ts_session(filepath)

    data = struct();
    
    hinfo = hdf5info(filepath);

    data.attrs = get_attrs(hinfo.GroupHierarchy.Attributes);

    for i = 1:length(hinfo.GroupHierarchy.Groups)
        group = hinfo.GroupHierarchy.Groups(i);
        if group.Name == "/Layers"
            data.layers = {};
            for j = 1:length(group.Groups)
                layer_group = group.Groups(j);
                layer = struct();
                layer.name = string(layer_group.Name);
                if ~isempty(layer_group.Datasets)
                    layer_dataset = layer_group.Datasets(1);
                    layer.data = hdf5read(filepath, layer_dataset.Name);
                end
                layer.attrs = get_attrs(layer_group.Attributes);
                data.layers{end+1} = layer;
            end
        elseif group.Name == "/Point Projections"
            data.point_projections = {};
            for j = 1:length(group.Datasets)
                point_projections_dataset = group.Datasets(j);
                point_projections = struct();
                point_projections.data = hdf5read(filepath, point_projections_dataset.Name);
                point_projections.attrs = get_attrs(point_projections_dataset.Attributes);
                data.point_projections{end+1} = point_projections;
            end
        end
    end
    
%     data.layers = {};
%     data.point_projections = {};
%     for i = 1:length(hinfo.GroupHierarchy.Datasets)
%         dataset = hinfo.GroupHierarchy.Datasets(i);
%         layer = struct();
%         layer.name = string(dataset.Name);
%         layer.attrs = get_attrs(dataset.Attributes);
%         if layer.attrs.type == "Image"
%             % layer.data = hdf5read(filepath, layer.name);
%         elseif layer.attrs.type == "Points"
%             layer.data = hdf5read(filepath, layer.name);
%         elseif layer.attrs.type == "Points Projections"
%             layer.data = hdf5read(filepath, layer.name);
%             data.point_projections{end+1} = layer;
%             continue
%         end
%         data.layers{end+1} = layer;
%     end

end


function attrs = get_attrs(AttributesInfo)
    
    attrs = struct();
    
    for i = 1:length(AttributesInfo)
        attr = AttributesInfo(i);
        name = split(string(attr.Name), '/');
        name = strip(name(end));
        value = attr.Value;
        if string(class(value)) == "hdf5.h5string"
            try
                value = string(value.Data);
            catch
                value = "";
            end
        elseif string(class(value)) == "hdf5.h5enum"
            value = value.Data;
        end
        attrs.(name) = value;
    end

end

