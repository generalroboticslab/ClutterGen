# Importing necessary modules from Blender Python API and other standard libraries.
from bpy.types import (
    Operator,
    OperatorFileListElement,
    Panel
)
from bpy.props import (
    StringProperty,
    CollectionProperty
)
from bpy_extras.io_utils import ImportHelper
import bpy
import pickle
from os.path import splitext, join, basename, exists
from os import listdir
from mathutils import Vector
import math


def load_pkl(filepath=None):
    # Setting up some initial variables.
    # These include the directory of the files, file extension, frame skipping, and maximum frames.
    skip_frames = 1
    max_frames_limit = 100000

    # Getting the current Blender context which contains data like current scene, selected objects etc.
    context = bpy.context

    # Iterating through each file in the directory.
    # Constructing the full file path.
    assert exists(filepath), f"File does not exist: {filepath}"
    print(f'Processing {filepath}')

    max_frames = 0
    # Opening the .pkl (pickle) file which contains serialized Python object data.
    with open(filepath, 'rb') as pickle_file:
        data = pickle.load(pickle_file)

        # Creating a new collection in Blender for each file. 
        # Collections are used to organize objects in the scene.
        collection_name = splitext(basename(filepath))[0]
        collection = bpy.data.collections.new(collection_name)
        context.scene.collection.children.link(collection)
        context.view_layer.active_layer_collection = \
            context.view_layer.layer_collection.children[-1]

        # Iterating through each object in the loaded data.
        for obj_key in data:
            pybullet_obj = data[obj_key]

            # Ensuring the object type is a mesh.
            assert pybullet_obj['type'] == 'mesh'
            assert pybullet_obj['mesh_path'] is not None \
                and exists(pybullet_obj['mesh_path']), \
                f"Mesh file {pybullet_obj['mesh_path']} does not exist."

            # Extracting the file extension of the mesh.
            extension = pybullet_obj['mesh_path'].split(".")[-1].lower()

            # Handling different mesh formats by using appropriate Blender import operators.
            if 'obj' in extension:
                bpy.ops.wm.obj_import(
                    filepath=pybullet_obj['mesh_path'], forward_axis='Y', up_axis='Z')
            elif 'dae' in extension:
                bpy.ops.wm.collada_import(
                    filepath=pybullet_obj['mesh_path'])
            elif 'stl' in extension:
                bpy.ops.wm.stl_import(
                    filepath=pybullet_obj['mesh_path'])
            else:
                print("Unsupported File Format:{}".format(extension))
                continue

            # Deleting any imported lights and cameras as they are not needed.
            parts = 0
            final_objs = []
            for import_obj in context.selected_objects:
                bpy.ops.object.select_all(action='DESELECT')
                import_obj.select_set(True)
                if 'Camera' in import_obj.name \
                        or 'Light' in import_obj.name\
                        or 'Lamp' in import_obj.name:
                    bpy.ops.object.delete(use_global=True)
                else:
                    # Applying scale to the imported object.
                    scale = pybullet_obj['mesh_scale']
                    if scale is not None:
                        import_obj.scale.x = scale[0]
                        import_obj.scale.y = scale[1]
                        import_obj.scale.z = scale[2]
                    final_objs.append(import_obj)
                    parts += 1
            
            # Deselecting all objects and then re-selecting the final objects.
            bpy.ops.object.select_all(action='DESELECT')
            for obj in final_objs:
                if obj.type == 'MESH':
                    obj.select_set(True)
            
            # Joining selected objects into a single object.
            if len(context.selected_objects):
                context.view_layer.objects.active =\
                    context.selected_objects[0]
                bpy.ops.object.join()

            # Renaming the final joined object to match the key from the data.
            blender_obj = context.view_layer.objects.active
            blender_obj.name = obj_key

            # Keyframing the motion of the imported object based on the data.
            for frame_count, frame_data in enumerate(pybullet_obj['frames']):
                if frame_data is None:
                    continue
                if frame_count % skip_frames != 0:
                    continue
                if max_frames_limit > 1 and frame_count > max_frames_limit:
                    print('Exceed max frame count')
                    break
                percentage_done = frame_count / len(pybullet_obj['frames'])
                print(f'\r[{percentage_done*100:.01f}% | {obj_key}]',
                        '#' * int(60*percentage_done), end='')
                
                # Setting position and orientation of the object and corrosponding frame position.
                pos = frame_data['position']
                orn = frame_data['orientation']
                frame_number = frame_count // skip_frames
                context.scene.frame_set(frame_number)
                
                # Applying position and rotation to the object.
                blender_obj.location.x = pos[0]
                blender_obj.location.y = pos[1]
                blender_obj.location.z = pos[2]
                blender_obj.rotation_mode = 'QUATERNION'
                blender_obj.rotation_quaternion.x = orn[0]
                blender_obj.rotation_quaternion.y = orn[1]
                blender_obj.rotation_quaternion.z = orn[2]
                blender_obj.rotation_quaternion.w = orn[3]

                # Directly inserting keyframes for both rotation and location.
                blender_obj.keyframe_insert(data_path="location", frame=frame_number)
                blender_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)

            # Updating the maximum frame count.
            max_frames = max(max_frames, frame_number)

    # Indicating the script has finished execution.
    print('FINISHED')

    return collection, max_frames
