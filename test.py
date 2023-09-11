from isaacgym_utils import *

urdf_path = "assets/objects/ycb_objects_origin_at_center_vhacd/urdfs/banana.urdf"
robot = urdf.Robot.from_xml_file(urdf_path)
# Iterate through links or joints to access attributes
for link_name, link in robot.links.items():
    # Get scaling of the link
    scaling = link.visual.origin.xyz if link.visual is not None else None
    
    # Get friction parameters (if defined)
    if link.collision and link.collision.surface:
        friction = link.collision.surface.friction if link.collision.surface.friction else None
    else:
        friction = None

    # Print the attributes
    print(f"Link: {link_name}")
    print(f"Scaling: {scaling}")
    print(f"Friction: {friction}")
    print("\n")
getUrdfStates(urdf_path)