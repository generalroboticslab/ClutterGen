import openai
import os
from collections import deque

"""
Prompt Rules
API key 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

I want to ask you to write a training environment description.
The training task is to ask the robot arm to pick the arrow_box and put it on the other box.

The available object descriptions include its name and their 3D bounding box dimension x, y, z:
{"box": [0.05, 0.05, 0.05], "table": [0.6, 1.0, 0.4], "arrow_box": [0.05, 0.05, 0.05]}.

You need to generate objects' positions, orientations, and scaling (if you want), based on the difficulty I provide to you ["Easy", "Medium", "Hard"], and number of envs I provided to you.
For example, If I provide you: [Difficulty: Easy, Number of Envs: 2],
You should generate a list of dictionary: You should output like this format: [{"box":[[0.1, 0.1, 0.05], [0.0, 0.0, 0.0, 1.0], 1], "table":[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], 1], "arrow_box":[[-0.1, -0.1, 0.05], [0.0, 0.0, 0.0, 1.0], 1]}, {Env2 description}] 
The length of list should be equal to the number of envs.
Positions is a list with three items each is float number, orientations is a list with four items which represent a quaternion, scaling is an int number to show whether you want to make the current object bigger or smaller.

Finally, the two boxes should be on the table and no objects collide with each other. The robot arm is centered at [0, 0, 0]

"""

class GPT:
    def __init__(self, prompt_head=None, api_key="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") -> None:
        openai.api_key = api_key
        self.prompt_head = prompt_head if prompt_head is not None \
        else """I will give you a task description and a description of what I am seeing. 
        I will also ask your question, you can only answer the question by choosing objects in Available_objects. 
        The answer format for you is strict! Please only output the content AFTER 'Your answer'.
        The structure example is below:
         Task: Rearrange the table.
         Scene: ["table", "cube", "cuboid"]. 
         Available_objects: ["cube", "cuboid", "bowl"]
         Question: What possible objects can exist on the table in daily life?
         Your answer: ["bowl", "cube", "cuboid"]
         
         You should answer as the format: "Your answer: []" if you think there are no objects suitable for the question."""
        self.message_head = {"role": "system", "content": self.prompt_head}
        self.maximum_msg_num = 100
        self.messages = deque([], maxlen=self.maximum_msg_num)
        

    def query_scene_objects(self, available_objects=["cube_arrow", "cuboid", "bowl"]):
        task = "Rearrange the table"
        scene = ["table", "cube_arrow", "cuboid"]
        available_objects = available_objects
        question = "What possible objects can exist on the table in daily life?" # other or not other
        
        message =   f"""
                    Task: {task}
                    Scene: {scene}
                    Available_objects: {available_objects}
                    Question: {question}
                    """
        
        self.messages.appendleft(self.message_head)
        self.messages.append({"role": "user", "content": message})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=list(self.messages)
        )
        reply = chat.choices[0].message.content
        self.messages.popleft()
        self.messages.append({"role": "assistant", "content": reply})

        choose_objs = []
        try:
            choose_objs = eval('[' + reply.split(sep="[")[1])
            print(choose_objs)
        except:
            print(f"Invalid Reply: {reply}")
        return choose_objs
    
    
    def text_visual(self):
        print(f"{self.message_head['role']}: {self.message_head['content']}")
        for dialog in self.messages:
            print(f"{dialog['role']}: {dialog['content']}")


if __name__=="__main__":
    assets_path = "assets/objects/ycb_objects_origin_at_center_vhacd/urdfs"
    all_obj_names = os.listdir(assets_path)
    all_obj_names = [name.split(sep=".")[0] for name in all_obj_names]
    chat = GPT()
    chat.query_scene_objects(all_obj_names)
    chat.text_visual()
