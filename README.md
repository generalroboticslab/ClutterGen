Sicne Issac Gym requires python<3.9: [Install python3.8](https://linuxgenie.net/how-to-install-python-3-8-on-ubuntu-22-04/)

```
python3.8 -m venv venv --system-site-packages
source venv/bin/activate
pip install -r requirements.txt
```

```
# Python.h: No such file or directory
sudo apt-get install python3.8-dev
```


# ClutterGen: A Cluttered Scene Generator for Robot Learning
<span style="font-size:17px; display: block; text-align: left;">
    <a href=# target="_blank" style="text-decoration: underline;">[Project Page]</a> 
    <a href=# target="_blank" style="text-decoration: underline;">[Video]</a>
    <a href=# target="_blank" style="text-decoration: underline;">[arXiv]</a> <br>
</span>


### Abstract
ClutterGen, a physically compliant simulation scene generator capable of producing highly diverse, cluttered, and stable scenes for robot learning. Generating such scenes is challenging as each object must adhere to physical laws like gravity and collision. As the number of objects increases, finding valid poses becomes more difficult, necessitating significant human engineering effort, which limits the diversity of the scenes. To overcome these challenges, we propose a reinforcement learning method that can be trained with physics-based reward signals provided by the simulator. Our experiments demonstrate that ClutterGen can generate cluttered object layouts with up to ten objects on confined table surfaces. Additionally, our policy design explicitly encourages the diversity of the generated scenes for open-ended generation. Our real-world robot results show that ClutterGen can be directly used for clutter rearrangement and stable placement policy training.

<p align="center">
    <img src="paper/final/png/teaser_2.png" width="600"> <br>
    <em>(a/b) When the number of objects in the environment increases, the difficulty of creating such a stable setup also increases. The traditional heuristic method cannot create a simulation scene above 7 objects, while ClutterGen consistently achieves high success rates. (c) Diverse, cluttered, and stable simulation setups created by ClutterGen.</em>
</p>

## Content
<span style="font-size:17px; display: block; text-align: center;">
    <a href="#prerequisites">Prerequisites</a> <br>
    <a href="#Training">Training</a> <br>
    <a href="#Evaluation">Testing</a> <br>
    <a href="#Real-robot">Real-robot</a> <br>
    <a href="#BibTex">BibTex</a> <br>
</span>

## Prerequisites

Please clone the repository first, (Link needs to be changed)
```sh
git clone https://github.com/generalroboticslab/RoboSensai.git
cd RoboSensai
git checkout multienv_sg_simple_temp
```

We use [`conda`](https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html) to create a new environment with python3.8.
```sh
conda create -n ClutterGen python=3.8
```

Activate the environment,
```sh
conda activate ClutterGen
```

Install the required packages,
```sh
pip install -r requirements.txt
```

## Training
We use the `wandb` package to log the training stats, which will ask you to log in the account if you set `--collect_data True`. You can create a free account on [wandb](https://wandb.ai). <br>

To train full _ClutterGen_ model, run the following command:
```sh
python ppo_discrete_asyn.py \
--collect_data \
--result_dir train_res \
--random_target \
--random_target_init \
--handed_search \
--num_envs 1300 \
--filter_contact \
--add_random_noise \
--use_contact_force \
--constrain_ws \
--quiet True \
--rendering \
--graphics_device_id 0
# The id of the GPU might be varied in 0, 1 or 2 based on your machine; If you can not see it rendering successfully, please try to change the id
# We suggest to not use the rendering option for training; set --rendering False
```

After the training is done, your checkpoint will be saved in the `/[--result_dir]/Random/checkpoint` and your environment description file will be saved in the `/[--result_dir]/Random/Json`. <br>
To train the stage 2, find the checkpoint name which is same as the checkpoint folder name, `.json` file name (no suffix), and the name shows on `wandb`.  By running the following command, the finger will be trained to retrieve the object from the granular media. <br>

```sh
python ppo_discrete_asyn.py \
--collect_data \
--random_target \
--random_target_init \
--handed_search \
--num_envs 10 \
--filter_contact \
--add_random_noise \
--use_contact_force \
--constrain_ws \
--add_gms \
--checkpoint [CHECKPOINT NAME] \
--index_episode [CHECKPOINT EPISODE] \
--reward_steps 1000 \
--min_force_filter 3. \
--quiet True \
--rendering \
--graphics_device_id 0
# The id of the GPU might be varied in 0, 1 or 2 based on your machine
# We suggest to not use the rendering option for training --rendering False
```

A quick example that used our pre-trained model is shown below. <br>
```sh
python ppo_discrete_asyn.py \
--collect_data  \
--random_target \
--random_target_init \
--handed_search \
--num_envs 1 \
--filter_contact \
--add_random_noise \
--use_contact_force \
--constrain_ws \
--add_gms \
--result_dir saved_res \
--checkpoint Random_01-18_22:51_P2G_FineTune_FC_CPU_Rand_target_targInit_With_force_Use_search_Add_noise_filter_limitws_gms_F_filter3.0N_Weight_succ800.0_pena10.0_pos20.0_rot-1.57 \
--index_episode best \
--reward_steps 1000 \
--min_force_filter 3. \
--quiet True \
--rendering \
--graphics_device_id 0
# The id of the GPU might be varied in 0, 1 or 2 based on your machine
# We suggest to not use the rendering option for training --rendering False
```

To train the _GEOTACT_ model in one stage from scratch, run the following command,
```sh
python ppo_discrete_asyn.py \
--collect_data \
--result_dir train_res \
--random_target \
--random_target_init \
--handed_search \
--num_envs 10 \
--filter_contact \
--add_random_noise \
--use_contact_force \
--constrain_ws \
--add_gms \
--reward_steps 1000 \
--min_force_filter 3. \
--seed 123456 \
--quiet True \
--rendering \
--graphics_device_id 0
# The id of the GPU might be varied in 0, 1 or 2 based on your machine
# We suggest to not use the rendering option for training --rendering False
```

## Evaluation

We have provided a script to evaluate the saved model checkpoints `evaluation.py`. We also provide some pre-trained models under `/saved_res`. All evaluation results will be stored in `/eval_res` if `--collect_data True`.  

To enjoy the pre-trained _GEOTACT_ model,

```sh
python evaluation.py \
--collect_data \
--real False \
--num_envs 10 \
--num_trials 1000 \
--specific_target [Specific Target Name] \
--checkpoint [CHECKPOINT NAME] \
--index_episode [CHECKPOINT EPISODE] \
--rendering \
--graphics_device_id 0
# The id of the GPU might be varied in 0, 1 or 2 based on your machine
# We suggest to not use the rendering option for training
# --specific_target is in['cube', 'cuboid', 'ball', 'potted_meat_can', 'tomato_soup_can', 'pentagram','L-shape']; Default is None
```

A quick example that used our pre-trained model is shown below. <br>
```sh
python evaluation.py \
--collect_data False \
--real False \
--num_envs 1 \
--num_trials 1000 \
--specific_target None \
--result_dir saved_res \
--checkpoint Random_01-18_22:51_P2G_FineTune_FC_CPU_Rand_target_targInit_With_force_Use_search_Add_noise_filter_limitws_gms_F_filter3.0N_Weight_succ800.0_pena10.0_pos20.0_rot-1.57 \
--index_episode best \
--rendering \
--graphics_device_id 1
# The id of the GPU might be varied in 0, 1 or 2 based on your machine
```

_Note: If you meet the problem of out of GPU memory, you can decrease the number of `num_envs`_

## Real-robot

We use the UR5 robotic arm and the [DISCO tactile finger](https://arxiv.org/abs/2004.00685) developed in [ROAM Lab](https://roam.me.columbia.edu/) at Columbia University. The DISCO finger is not publicly available. But if you are interested in getting one, please contact [Jingxi Xu](https://jxu.ai). We use the [ur_rtde](https://sdurobotics.gitlab.io/ur_rtde/) package for UR5 control. We also provide the CAD models of the finger coupler that connects the finger base to the UR5 arm.

When you have the UR5 and DISCO finger set up, run the following code to run the evaluation on the real robot.

### Real-robot Related (Optional)
We here list the packages that are required for the real-robot experiments. If you are not interested in the real-robot experiments, you can skip this part. <br>

**GroundDINO**
[optinal] If you want to use the real robot, you need to install the `ur_rtde` package. Please follow the instructions [here](https://sdurobotics.gitlab.io/ur_rtde/).

```sh
python evaluation.py \
--real True \
--num_envs 1 \
--num_trials 10 \
--checkpoint [CHECKPOINT NAME] \
--index_episode [CHECKPOINT EPISODE] \
--rendering \
--graphics_device_id 0
# The id of the GPU might be varied in 0, 1 or 2 based on your machine
```

## BibTeX

If you find this repo useful, please consider citing,

```
@article{xu2024tactile,
  title={Tactile-based Object Retrieval From Granular Media},
  author={Xu, Jingxi and Jia, Yinsen and Yang, Dongxiao and Meng, Patrick and Zhu, Xinyue and Guo, Zihan and Song, Shuran and Ciocarlie, Matei},
  journal={arXiv preprint arXiv:2402.04536},
  year={2024}
}
```