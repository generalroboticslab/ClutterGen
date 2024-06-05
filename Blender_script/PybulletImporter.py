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
from os import listdir, makedirs
from mathutils import Vector, Quaternion
import math
import time
import random
import numpy as np
import re
import json
import argparse



def load_pkl(filepath=None, load_all=True):
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
            loaded_frames = pybullet_obj['frames'] if load_all else [pybullet_obj['frames'][-1]]
            for frame_count, frame_data in enumerate(loaded_frames):
                if frame_data is None:
                    continue
                if frame_count % skip_frames != 0:
                    continue
                if max_frames_limit > 1 and frame_count > max_frames_limit:
                    print('Exceed max frame count')
                    break
                percentage_done = frame_count / len(loaded_frames)
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


def load_actor_visualization(filepath=None):
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
        print(f"Loading {filepath}")
        data = pickle.load(pickle_file)

    _, example_data = data["prob_pos_heatmap"][0]
    num_visual_objs = example_data["World_2_ObjBboxCenter_xyz_i"].shape[0]

    # Add visualization object
    visual_objs = []
    context = bpy.context
    collection_name = "actor_visual"
    collection = bpy.data.collections.new(collection_name)
    context.scene.collection.children.link(collection)
    context.view_layer.active_layer_collection = \
        context.view_layer.layer_collection.children[-1]
    bpy.ops.object.select_all(action='DESELECT')
    for i in range(num_visual_objs):
        location = [-1000, -1000, 0]
        vis_obj = add_primitive_object(object_type='CUBE', location=location, scale=[1, 1, 1.], use_emission=True, texture_path=None, alpha=0.3)
        visual_objs.append(vis_obj)

    for frame_index, actor_visual_data in data["prob_pos_heatmap"]:
        World_2_ObjBboxCenter_xyz_i = actor_visual_data["World_2_ObjBboxCenter_xyz_i"]
        using_act_prob_i = actor_visual_data["using_act_prob_i"]
        visual_voxel_half_extents = actor_visual_data["visual_voxel_half_extents"]
        assert World_2_ObjBboxCenter_xyz_i.shape[0] == num_visual_objs
        for i, obj in enumerate(visual_objs):
            rgba_color = [1, 1, 0, using_act_prob_i[i]]
            xyz_pos = World_2_ObjBboxCenter_xyz_i[i]
            context.view_layer.objects.active = obj  # Correctly set each obj as active
            # Assume you want to vary the frame for demonstration; adjust as needed
            change_object_color(obj, color=rgba_color)
            # TODO Change object scaling to match the scale at this frame
            # Set initial location and rotation
            obj.location = xyz_pos
            obj.rotation_mode = 'QUATERNION'
            obj.rotation_quaternion = (1, 0, 0, 0)
            obj.keyframe_insert(data_path="location", frame=frame_index)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_index)
            obj.scale = visual_voxel_half_extents
            obj.keyframe_insert(data_path="scale", frame=frame_index)

            # Access the animation data of the object to change interpolation mode
            if obj.animation_data and obj.animation_data.action:
                for fcurve in obj.animation_data.action.fcurves:
                    for keyframe_point in fcurve.keyframe_points:
                        keyframe_point.interpolation = 'CONSTANT'
        # Indicating the script has finished execution.
    print('FINISHED')

    return collection, max_frames


def change_object_color(obj, color=(1.0, 1.0, 1.0, 1.0)):
    """
    Changes the color, transparency, and emission of a given Blender object. Ensures the object does not generate shadows.
    
    Parameters:
        obj (bpy.types.Object): The Blender object to change the properties of.
        color (tuple): A tuple representing the RGBA color and transparency.
                       Each component should be a float between 0.0 and 1.0.
    """
    
    # Ensure the object has a material
    if not obj.data.materials:
        # Create a new material if the object has none
        mat = bpy.data.materials.new(name="NewMaterial")
        obj.data.materials.append(mat)
    else:
        # Get the first material of the object
        mat = obj.data.materials[0]
    
    # Ensure the material uses nodes and clear existing nodes
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Create necessary nodes
    emission = nodes.new(type='ShaderNodeEmission')
    emission.inputs['Color'].default_value = color  # Set color
    emission.inputs['Strength'].default_value = 1.0  # Adjust as needed

    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    material_output = nodes.new(type='ShaderNodeOutputMaterial')

    # Set up node links
    mix_shader.inputs['Fac'].default_value = 1.0 - color[3]
    mat.node_tree.links.new(mix_shader.inputs[1], emission.outputs['Emission'])
    mat.node_tree.links.new(mix_shader.inputs[2], transparent.outputs['BSDF'])
    mat.shadow_method = 'NONE'

    mat.node_tree.links.new(material_output.inputs['Surface'], mix_shader.outputs['Shader'])
    
    # Update the blend method for transparency
    mat.blend_method = 'BLEND' if color[3] < 1.0 else 'OPAQUE'


def set_blender_engine(render_engine='BLENDER_EEVEE'):
    """
    Sets the render engine to either EEVEE or Cycles.
    """
    # https://github.com/nytimes/rd-blender-docker/issues/3
    assert render_engine in ['BLENDER_EEVEE', 'CYCLES'], f"render_engine must be either BLENDER_EEVEE or CYCLES, but got {render_engine}"
    bpy.data.scenes[0].render.engine = render_engine
    if render_engine == 'CYCLES':
        engine_key = render_engine.lower()
        # Set the device_type
        bpy.context.preferences.addons[engine_key].preferences.compute_device_type = "OPTIX"  # "CUDA" or "OPTIX" for NVIDIA RTX cards
        bpy.context.preferences.addons[engine_key].preferences.memory_cache_limit = 10240
        # Set the device and feature set
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons[engine_key].preferences.refresh_devices()
        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons[engine_key].preferences.get_devices()
        # Enable all CPU and GPU devices
        for device in bpy.context.preferences.addons[engine_key].preferences.devices:
            # print(f"Device {device.name} is {device.type}")
            # Useless, can not control!
            device.use = True

    print(f"Render Engine is set to {render_engine}")


def delete_collection(collections=None, specific_name=None):
    """
    Deletes all collections.
    """
    deleting_collections = collections if collections is not None else bpy.data.collections
    for collection in deleting_collections:
        # Iterate through all scenes and unlink the collection
        if specific_name is not None and specific_name not in collection.name: continue

        for scene in bpy.data.scenes:
            if collection.name in scene.collection.children:
                scene.collection.children.unlink(collection)
        
        # Delete all objects in the collection
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Delete the collection
        bpy.data.collections.remove(collection)
    print("Collections Deleting is Done")


def delete_scene_objects():
    """
    Deletes all objects in the scene.
    """
    for obj in bpy.context.scene.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    print("Scene Objects Deleting is Done")


def render_animation(output_path, encoder='H264', container='MPEG4', resolution=(1920, 1080), 
                     start_frame=1, end_frame=250, skip_frames=1, quality=90, frame_rate=24, samples=2048, animation=True):
    """
    Renders an animation in Blender to a specified path with various customizable settings.

    :param output_path: The path where the rendered animation will be saved.
    :param encoder: The video format for the render ('H264', 'FFmpeg', etc.).
    :param resolution: A tuple containing the resolution (width, height) of the render.
    :param skip_frames: Number of frames to skip while rendering.
    :param start_frame: The starting frame of the animation.
    :param end_frame: The ending frame of the animation.
    :param quality: Quality of the render (Percentage).
    :param frame_rate: Frame rate of the animation.
    """

    # Set the render resolution
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    # Set the frame rate
    bpy.context.scene.render.fps = frame_rate

    # Set the output path
    bpy.context.scene.render.filepath = output_path

    # Set the output format
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG' if animation else 'PNG'
    # Set the codec
    bpy.context.scene.render.ffmpeg.format = container
    bpy.context.scene.render.ffmpeg.codec = encoder

    # Set the quality
    bpy.context.scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    bpy.context.scene.render.ffmpeg.video_bitrate = quality

    # Set the keyframe skipping
    bpy.context.scene.frame_start = start_frame
    bpy.context.scene.frame_end = end_frame
    bpy.context.scene.frame_step = skip_frames

    # Disable denoising for Cycles
    if bpy.context.scene.render.engine == 'CYCLES':
        print("Setting up Cycles")
        bpy.context.scene.cycles.use_denoising = False
        # uncheck the noise threshold
        bpy.context.scene.cycles.use_adaptive_sampling = False
        # Change number of samples for Cycles
        bpy.context.scene.cycles.samples = samples
        # Set light path settings
        bpy.context.scene.cycles.max_bounces = 6
        bpy.context.scene.cycles.transmission_bounces = 6
        bpy.context.scene.cycles.transparent_max_bounces = 6
        # Set the film transparency
        bpy.context.scene.render.film_transparent = True
    
    # Change the number of samples for EEVEE
    if bpy.context.scene.render.engine == 'BLENDER_EEVEE':
        bpy.context.scene.eevee.taa_samples = 64
        bpy.context.scene.eevee.film_transparent_glass = True

    # Render the animation (https://blender.stackexchange.com/questions/283640/why-does-bpy-ops-render-renderanimation-true-work-but-fails-when-animation-is)
    bpy.ops.render.render(animation=animation, write_still=True)

    print(f"Rendered animation to {output_path}")
    return


def animate_camera_focus_rotate(focus_point, camera_start_location, rotation_angle, frame_start, frame_end):
    """
    Creates a camera that focuses on a specific point and rotates around it.

    Args:
    - focus_point: Tuple (x, y, z) for the point the camera will focus on.
    - camera_start_location: Tuple (x, y, z) for the initial camera location.
    - rotation_angle: Total angle in degrees for camera rotation.
    - frame_start: Start frame for the animation.
    - frame_end: End frame for the animation.
    """

    # Scene setup
    scene = bpy.context.scene

    # Add camera
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    scene.collection.objects.link(camera)
    scene.camera = camera

    # Set camera location and focus point
    camera.location = Vector(camera_start_location)
    direction = camera.location - Vector(focus_point)
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Add an empty axes
    axes = bpy.data.objects.new("Axes", None)
    scene.collection.objects.link(axes)
    axes.location = focus_point

    # Parent camera to axes
    camera.parent = axes

    # Insert keyframes for rotation
    if frame_start < frame_end:
        axes.rotation_mode = 'XYZ'
        for frame in range(frame_start, frame_end + 1):
            fraction = (frame - frame_start) / (frame_end - frame_start)
            axes.rotation_euler[2] = math.radians(fraction * rotation_angle)
            axes.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Set scene frame range
        scene.frame_start = frame_start
        scene.frame_end = frame_end


def add_light(name="Light", type='POINT', location=(0, 0, 0), energy=1000, color=(1, 1, 1)):
    """
    Adds a light to the Blender scene with adjustable properties.

    :param name: Name of the light.
    :param type: Type of light ('POINT', 'SUN', 'SPOT', 'AREA').
    :param location: Location of the light.
    :param energy: Power of the light.
    :param color: Color of the light.
    """
    bpy.ops.object.light_add(type=type, location=location)
    light = bpy.context.object
    light.name = name
    light.data.energy = energy
    light.data.color = color


def add_primitive_object(object_type='CUBE', location=(0, 0, 0), scale=(1, 1, 1), rotation=(0, 0, 0.), color=[1., 1., 1.], alpha=1.0, texture_path=None, use_emission=False, noshadow=False):
    """
    Adds a primitive object to the Blender scene with optional texture, emission shader, and transparency.

    :param object_type: Type of the primitive ('CUBE', 'PLANE', 'SPHERE', etc.).
    :param location: Location of the object.
    :param scale: Scale of the object.
    :param texture_path: Path to the texture file.
    :param use_emission: If True, uses an emission shader for a shadeless effect.
    :param alpha: Alpha (transparency) value of the object (0.0 to 1.0).
    """

    # Add primitive object
    bpy.ops.mesh.primitive_cube_add(location=location, scale=scale, rotation=rotation) if object_type == 'CUBE' else None
    bpy.ops.mesh.primitive_plane_add(location=location, size=scale[0], scale=scale) if object_type == 'PLANE' else None
    bpy.ops.mesh.primitive_uv_sphere_add(location=location, scale=scale) if object_type == 'SPHERE' else None

    obj = bpy.context.object

    # Create material
    mat = bpy.data.materials.new(name="Material")
    obj.data.materials.append(mat)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()  # Clear existing nodes

    # Add necessary nodes
    emission = nodes.new(type='ShaderNodeEmission')
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    material_output = nodes.new(type='ShaderNodeOutputMaterial')
    # Set color and alpha value
    color = list(color) + [1.0]  # Red color with 50% transparency

    # Set up node links
    if use_emission:
        emission.inputs['Strength'].default_value = 1.0
        emission.inputs['Color'].default_value = color
        mix_shader.inputs['Fac'].default_value = 1.0 - alpha
        mat.node_tree.links.new(mix_shader.inputs[1], emission.outputs['Emission'])
        mat.node_tree.links.new(mix_shader.inputs[2], transparent.outputs['BSDF'])
    else:
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        mix_shader.inputs['Fac'].default_value = 1.0 - alpha
        mat.node_tree.links.new(mix_shader.inputs[1], bsdf.outputs['BSDF'])
        mat.node_tree.links.new(mix_shader.inputs[2], transparent.outputs['BSDF'])
        bsdf.inputs['Base Color'].default_value = color

    mat.node_tree.links.new(material_output.inputs['Surface'], mix_shader.outputs['Shader'])

    # Add texture if provided
    if texture_path:
        tex_image = nodes.new('ShaderNodeTexImage')
        tex_image.image = bpy.data.images.load(texture_path)
        if use_emission:
            mat.node_tree.links.new(emission.inputs['Color'], tex_image.outputs['Color'])
        else:
            mat.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # Adjust material blend method for transparency
    mat.blend_method = 'BLEND' if alpha < 1.0 else 'OPAQUE'

    # Disable shadows for the object
    if noshadow:
        # print obj attributes
        # mat.shadow_method = 'NONE'
        obj.visible_shadow = False
        # obj.visible_transmission = False
        # obj.visible_diffuse = False
        # obj.visible_glossy = False
        # obj.visible_volume_scatter = False

    return obj


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [atoi(c) for c in re.split(r'(\d+)', text)]


##############################################################################################################
##############################################################################################################
                        ########################### Main Area#############################
##############################################################################################################
# How to improve the rendering speed?
# https://irendering.net/how-to-improve-speed-of-blenders-cycles-x-rendering-engine/
# Can not use if __name__ == "__main__" here. Blender seems does not support it.
# Only if you create the add-on operator, you can use it.

blender_traj_directory = "eval_res/Union/blender"
actor_vis_traj_directory = "eval_res/Union/trajectories"
json_file_directory = "eval_res/Union/Json"
# evalUniName_lst = ["Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5__EVAL_best_Scene_table_1_QRHalfExt_0.3_0.3_0.35_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy_EVAL_best_Scene_table_100_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_10_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_21_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_32_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_49_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_52_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_54_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_55_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0_EVAL_best_Scene_table_83_objRange_10_10",
#                    "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial3_entropy0._EVAL_best_Scene_table_2_objRange_10_10"]

# evalUniName_lst = ["Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_ma_EVAL_best_TableHalfExtents0.6_0.7_0.35_Scene_table_QRHalfExt_0.3_0.35_0.35_RQRPos_RQREulerZ_RsrkQRHalfExt_RexpQRHalfExt_objRange_10_10",
#                "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Ep_EVAL_best_TableHalfExtents0.6_0.7_0.35_Scene_table_QRHalfExt_0.3_0.35_0.35_RexpQRHalfExt_objRange_10_10",
#                "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Ep_EVAL_best_TableHalfExtents0.6_0.7_0.35_Scene_table_QRHalfExt_0.3_0.35_0.35_RsrkQRHalfExt_objRange_10_10",
#                "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2R_EVAL_best_TableHalfExtents0.6_0.7_0.35_Scene_table_QRHalfExt_0.3_0.35_0.35_RQREulerZ_objRange_10_10",
#                "Union_2024_05_17_154518_Sync_Beta_group4_real_objects_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Repl_EVAL_best_TableHalfExtents0.6_0.7_0.35_Scene_table_QRHalfExt_0.3_0.35_0.35_RQRPos_objRange_10_10"]


evalUniName_lst = ["Union_03-12_23:40Sync_Beta_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab60_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_s_EVAL_best_Scene_table_1_QRHalfExt_0.2_0.3_0.35_objRange_10_10",
                   "Union_03-12_23:40Sync_Beta_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab60_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0.01_seed123456_EVAL_best_Scene_table_100_objRange_10_10",
                   "Union_2024_04_22_144343_Sync_Beta_group1_studying_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entrop_EVAL_best_Scene_table_10_objRange_10_10",
                   "Union_2024_04_22_144343_Sync_Beta_group1_studying_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entrop_EVAL_best_Scene_table_21_objRange_10_10",
                   "Union_2024_04_22_144351_Sync_Beta_group2_office_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq_EVAL_best_Scene_table_32_QRHalfExt_0.3_0.35_0.35_objRange_10_10",
                   "Union_2024_04_22_144351_Sync_Beta_group2_office_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy0_EVAL_best_Scene_table_49_objRange_10_10",
                   "Union_2024_04_22_144403_Sync_Beta_group3_kitchen_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy_EVAL_best_Scene_table_52_objRange_10_10",
                   "Union_2024_04_22_144403_Sync_Beta_group3_kitchen_table_table_PCExtractor_Rand_ObjPlace_Goal_maxObjNum10_maxPool10_maxScene1_maxStab40_contStab20_Epis2Replaceinf_Weight_rewardPobj100.0_seq5_step80_trial5_entropy_EVAL_best_Scene_table_54_objRange_10_10"]

render_nums = 3
specific_eps = None # 36, 30, 27, 9, 18, 22, 26
animation = False
multiple_lights = True
add_plane = False
add_qr_region = False
load_all = False
set_blender_engine(render_engine='CYCLES') # 'BLENDER_EEVEE', 'CYCLES'
# For randomizing qr region
QrSceneCenter2Camera = [-2, 0., 3]
focus_offset = (0.07, 0, -0.1)
resolution = (1920, 1080)

for evalUniName in evalUniName_lst:
    json_file_path = join("eval_res/Union/Json", evalUniName+".json")
    meta_json_file_path = join("eval_res/Union/trajectories", evalUniName, "10Objs_meta_data.json")
    blender_traj_directory = join("eval_res/Union/blender", evalUniName)
    actor_vis_traj_directory = join("eval_res/Union/trajectories", evalUniName)
    blender_filename_ext = ".pkl"
    actor_vis_filename_ext = ".pkl"
    # Listing all files in the specified directory.
    blender_file_paths = []
    trajectory_file_paths = []
    if specific_eps is not None:
        blender_filepaths = [join(blender_traj_directory, filename) for filename in listdir(blender_traj_directory) if filename.endswith(blender_filename_ext) and f"{specific_eps}eps" in filename]
        actor_filepaths = [join(actor_vis_traj_directory, filename) for filename in listdir(actor_vis_traj_directory) if filename.endswith(actor_vis_filename_ext) and f"{specific_eps}eps" in filename]
    else:
        blender_filepaths = sorted([join(blender_traj_directory, filename) for filename in listdir(blender_traj_directory) if filename.endswith(blender_filename_ext) and "success" in filename], key=natural_keys)[:render_nums]
        actor_filepaths = sorted([join(actor_vis_traj_directory, filename) for filename in listdir(actor_vis_traj_directory) if filename.endswith(actor_vis_filename_ext) and "success" in filename], key=natural_keys)[:render_nums]

    save_folder = join(blender_traj_directory, "render_results")
    makedirs(save_folder, exist_ok=True)

    # Read the json file
    with open(json_file_path, 'r') as f:
        json_data = json.load(f)
    with open(meta_json_file_path, 'r') as f:
        meta_json_data = json.load(f)

    QueryRegion_halfext = json_data["QueryRegion_halfext"]
    QueryRegion_pos = json_data["QueryRegion_pos"]
    QueryRegion_euler_z = json_data["QueryRegion_euler_z"]
    qr_scene_halfexts = meta_json_data["qr_scene_pose"][2][7:]
    qr_regions = meta_json_data["qr_region_dict"] if "qr_region_dict" in meta_json_data else None

    for i, filepath in enumerate(blender_filepaths):
        filename = basename(filepath)
        eps_index = filename.split("_")[1].replace("eps", "")
        qr_region = qr_regions[eps_index] if qr_regions is not None else None
        delete_collection(specific_name=None)
        delete_scene_objects()

        # Add a plane as the ground with wooden texture
        if add_plane:
            add_primitive_object(object_type='PLANE', location=(0, 0, 0), scale=(1000, 1000, 1000), texture_path=None)

        # Add QR region, red color with transparency
        if add_qr_region and qr_region is not None:
            qr_region_location = qr_region[:3]
            qr_region_location[2] = qr_region_location[2] + qr_scene_halfexts[2]
            # adjust the quaternion convention from [x, y, z, w] to [w, x, y, z]
            blender_quat = [qr_region[6], *qr_region[3:6]]
            qr_euler = Quaternion(blender_quat).to_euler()
            qr_region_halfexts = qr_region[7:]
            add_primitive_object(object_type='CUBE', location=qr_region_location, scale=qr_region_halfexts, rotation=qr_euler, texture_path=None, use_emission=False, color=[1.0, 0.0, 0.0], alpha=0.5, noshadow=True)
        
        # Add a table; The size might be different from the original size!
        if json_data["specific_scene"] == "table":
            add_primitive_object(object_type='CUBE', location=(0, 0, qr_scene_halfexts[2]), scale=qr_scene_halfexts, color=[1., 1., 1.], alpha=1., use_emission=False, texture_path=None)

        # Load the data from the pickle file.
        _, max_frame = load_pkl(filepath, load_all=load_all)

        # Add visualization object for actor visualization
        if len(actor_filepaths) > 0:
            print(f"Loading actor visualization from {actor_filepaths[i]}")
            load_actor_visualization(actor_filepaths[i])

        # Adjust the camera and light
        if "table" in json_data["specific_scene"]:
            camera_start_location = (*QrSceneCenter2Camera[:2], QrSceneCenter2Camera[2] + qr_scene_halfexts[2])
            focus_point=(focus_offset[0], focus_offset[1], qr_scene_halfexts[2]+focus_offset[2]) # Focus on the table center; given some offset
            light_location1 = (0, 0, 5); light_location2 = (5, 0, 1); light_location3 = (-5, 0, 1); light_location4 = (0, 5, 5); light_location5 = (0, -5, 5)
            light_color = (1., 1., 1.) # (1., 0.85, 0.72)
        elif json_data["specific_scene"] == "storage_furniture_5":
            camera_start_location = (0, 5., 1.8)
            focus_point = (0, 0, 0.35)
            light_location = (-4., 0, 4)
            light_color = (1., 0.85, 0.72)
        elif json_data["specific_scene"] == "storage_furniture_6":
            camera_start_location = (0, 4.5, 1.8)
            focus_point = (0, 0, 0.45)
            light_location = (-3., 0, 2)
            light_color = (1., 0.85, 0.72)
        else:
            raise ValueError(f"Unknown specific_scene: {json_data['specific_scene']}")

        # Add light
        add_light(location=light_location1, energy=5000, color=light_color)
        if multiple_lights:
            # Add light
            add_light(location=light_location2, energy=2000, color=light_color)
            add_light(location=light_location3, energy=2000, color=light_color)
            add_light(location=light_location4, energy=500, color=light_color)
            add_light(location=light_location5, energy=500, color=light_color)

        # Add camera
        animate_camera_focus_rotate(
            focus_point=focus_point,
            camera_start_location=camera_start_location,
            rotation_angle=90,
            # frame_start=1,
            # frame_end=max_frame,
            frame_start=0,
            frame_end=0
        )

        # Render the animation
        skip_frames = 5
        frame_rate = 240 // skip_frames
        start_frame = max_frame if not animation else 1
        max_frame = max_frame if not animation else max_frame + 100

        start_time = time.time()
        output_name = basename(filepath).replace(blender_filename_ext, '.mp4') \
            if animation else basename(filepath).replace(blender_filename_ext, '.png')
        render_animation(f'{save_folder}/{output_name}', 
                        encoder='H264', 
                        resolution=resolution, 
                        skip_frames=skip_frames, 
                        start_frame=start_frame, 
                        end_frame=max_frame, 
                        quality=80, 
                        frame_rate=frame_rate,
                        animation=animation)
        print(f"Total time: {time.time() - start_time:.2f}s")

    print(f"{evalUniName} All Done!")
print("All All Done!")