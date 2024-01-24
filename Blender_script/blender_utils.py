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
        bpy.context.preferences.addons[engine_key].preferences.compute_device_type = "CUDA"
        bpy.context.preferences.addons[engine_key].preferences.memory_cache_limit = 10240
        # Set the device and feature set
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons[engine_key].preferences.refresh_devices()
        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons[engine_key].preferences.get_devices()
        # Enable all CPU and GPU devices
        for device in bpy.context.preferences.addons[engine_key].preferences.devices:
            print(f"Device {device.name} is {device.type}")
            # Useless, can not control!
            device.use = True


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
                     start_frame=1, end_frame=250, skip_frames=1, quality=90, frame_rate=24):
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
    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
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
    # Render the animation (https://blender.stackexchange.com/questions/283640/why-does-bpy-ops-render-renderanimation-true-work-but-fails-when-animation-is)
    bpy.ops.render.render(animation=True, write_still=False)
    
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


def add_primitive_object(object_type='CUBE', location=(0, 0, 0), scale=(1, 1, 1), texture_path=None, use_emission=False, alpha=1.0):
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
    bpy.ops.mesh.primitive_cube_add(location=location, scale=scale) if object_type == 'CUBE' else None
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
    emission.inputs['Strength'].default_value = 1.0
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    mix_shader = nodes.new(type='ShaderNodeMixShader')
    material_output = nodes.new(type='ShaderNodeOutputMaterial')

    # Set up node links
    if use_emission:
        mix_shader.inputs['Fac'].default_value = 1.0 - alpha
        mat.node_tree.links.new(mix_shader.inputs[1], emission.outputs['Emission'])
        mat.node_tree.links.new(mix_shader.inputs[2], transparent.outputs['BSDF'])
        mat.shadow_method = 'NONE'
    else:
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        mix_shader.inputs['Fac'].default_value = 1.0 - alpha
        mat.node_tree.links.new(mix_shader.inputs[1], bsdf.outputs['BSDF'])
        mat.node_tree.links.new(mix_shader.inputs[2], transparent.outputs['BSDF'])

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