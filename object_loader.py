import os
import shutil


# write a python file to change the object name to be uniform

class ObjectManager:
    def __init__(self, assets_path) -> None:
        self.assets_path = assets_path
        self.meshes_path = os.path.join(self.assets_path, "meshes")
        self.urdf_path = os.path.join(self.assets_path, "urdfs")


    def uniformalize_object_mesh(self):
        all_obj_folders = os.listdir(self.meshes_path)
        for obj_folder in all_obj_folders:
            obj_mesh_files = os.listdir(os.path.join(self.meshes_path, obj_folder))
            for obj_mesh_name in obj_mesh_files:
                if obj_mesh_name.endswith(".obj") and obj_mesh_name not in ["collision.obj", "textured.obj"]:
                    os.rename(os.path.join(self.meshes_path, obj_folder, obj_mesh_name), os.path.join(self.meshes_path, obj_folder, "collision"+".obj"))
                    print(f"Rename {obj_mesh_name} to collision.obj")
                if obj_mesh_name.endswith(".mtl") and obj_mesh_name not in ["textured.mtl"]:
                    os.rename(os.path.join(self.meshes_path, obj_folder, obj_mesh_name), os.path.join(self.meshes_path, obj_folder, "textured"+".mtl"))
                    print(f"Rename {obj_mesh_name} to textured.mtl")
            if "collision.obj" not in obj_mesh_files:
                # copy file from textured.obj to collision.obj
                shutil.copy(os.path.join(self.meshes_path, obj_folder, "textured.obj"), os.path.join(self.meshes_path, obj_folder, "collision"+".obj"))
                print(f"Missing collision.obj in {obj_folder}; Copy textured.obj to collision.obj")
            if "textured.obj" not in obj_mesh_files:
                # copy file from collision.obj to textured.obj
                shutil.copy(os.path.join(self.meshes_path, obj_folder, "collision.obj"), os.path.join(self.meshes_path, obj_folder, "textured"+".obj"))
                print(f"Missing textured.obj in {obj_folder}; Copy collision.obj to textured.obj")
            if "collision.obj" not in obj_mesh_files and "textured.obj" not in obj_mesh_files:
                print(f"Missing both collision.obj and textured.obj in {obj_folder}")
    
    
    # write a python file to change the mesh path of the urdf file to new path
    def uniformalize_object_urdf(self):
        all_obj_urdfs = os.listdir(self.urdf_path)
        for obj_urdf in all_obj_urdfs[:1]:
            with open(obj_urdf, 'r') as urdf_file:
                urdf_lines = urdf_file.readlines()
                # Change a line in the file
                


                        
if __name__=="__main__":
    assets_path = "assets/objects/ycb_objects_origin_at_center_vhacd"
    obj_manager = ObjectManager(assets_path)
    obj_manager.uniformalize_object_mesh()
    # obj_manager.uniformalize_object_urdf()