from Blender_script.blender_utils import *
from Blender_script.PybulletImporter import load_pkl


# Example usage
# Can not use if __name__ == "__main__" here. Blender seems does not support it.
# Only if you create the add-on operator, you can use it.


set_blender_engine(render_engine='BLENDER_EEVEE')
directory = "eval_res/Union/trajectories/Union_01-19_14:43_Transformer_Tanh_Rand_ObjPlace_QRRegion_Goal_maxObjNum5_maxPool10_maxScene1_maxStable60_contStable20_maxQR1Scene_Epis2Replaceinf_Weight_rewardPobj100.0_seq10_EVAL_best_objRange_5_5"
filename_ext = ".pkl"
# Listing all files in the specified directory.
file_name = '5Objs_4eps_failure.pkl'
# filepath = [file_name for file_name in listdir(directory) if file_name.endswith(filename_ext)][0]
filepath = join(directory, file_name)

delete_collection(specific_name=None)
delete_scene_objects()

add_primitive_object(object_type='CUBE', location=(0, 0, 0.35), scale=(0.4, 0.5, 0.35), texture_path=None)
# Add a plane as the ground with wooden texture
add_primitive_object(object_type='PLANE', location=(0, 0, 0), scale=(1000, 1000, 1000), texture_path=None)

# Add visualization object
# add_primitive_object(object_type='SPHERE', location=(0, 0, 1.0), scale=(0.1, 0.1, 0.1), use_emission=True, texture_path=None, alpha=0.3)

# Add light
add_light(location=(0, 0, 5), energy=5000, color=(1., 1., 1.))
# Add camera
animate_camera_focus_rotate(
    focus_point=(0, 0, 0.35),
    camera_start_location=(3, 0., 3.),
    rotation_angle=90,
    frame_start=1,
    frame_end=900
)

# Load the data from the pickle file.
_, max_frame = load_pkl(filepath)

skip_frames = 10
frame_rate = 240 // skip_frames
render_animation('test_res/output.mp4', encoder='H264', resolution=(1280, 720), 
                 skip_frames=skip_frames, start_frame=0, end_frame=max_frame+480, quality=80, frame_rate=frame_rate)