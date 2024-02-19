# Contrastive Initial State Buffer for Reinforcement Learning

<p align="center">
 <a href="https://youtu.be/RB7mDq2fhho">
  <img src="doc/thumbnail.png" alt="youtube_video" width="800"/>
 </a>
</p>

This is the code for the ICRA24 paper **Contrastive Initial State Buffer for Reinforcement Learning**
([PDF](https://rpg.ifi.uzh.ch/docs/Arxiv23_Messikommer.pdf)) by [Nico Messikommer](https://messikommernico.github.io/), [Yunlong Song](https://yun-long.github.io/), and [Davide Scaramuzza](http://rpg.ifi.uzh.ch/people_scaramuzza.html).
For an overview of our method, check out our [video](https://youtu.be/RB7mDq2fhho).

If you use any of this code, please cite the following publication:

```bibtex
@Article{Messikommer24icra,
  author  = {Nico Messikommer and Yunlong Song and Davide Scaramuzza},
  title   = {Contrastive Initial State Buffer for Reinforcement Learning},
  journal = {2024 IEEE International Conference on Robotics and Automation (ICRA)},
  year    = {2024},
}
```

## Abstract

In Reinforcement Learning, the trade-off between exploration and exploitation poses a complex challenge for achieving efficient learning from limited samples.
While recent works have been effective in leveraging past experiences for policy updates, they often overlook the potential of reusing past experiences for data collection.
Independent of the underlying RL algorithm, we introduce the concept of a Contrastive Initial State Buffer, which strategically selects states from past experiences and uses them to initialize the agent in the environment in order to guide it toward more informative states.
We validate our approach on two complex robotic tasks without relying on any prior information about the environment: (i) locomotion of a quadruped robot traversing challenging terrains and (ii) a quadcopter drone racing through a track.
The experimental results show that our initial state buffer achieves higher task performance than the nominal baseline while also speeding up training convergence.

## Content

This repository contains the code for the Contrastive Initial State Buffer (CL-Buffer), which can be installed as a Python library. 
The repository does not contain the code for training an RL agent in an environment (Drone Racing or Legged Locomotion).
However, with the given toy example (```toy_example.py```), it is straightforward to implement the CL-Buffer in an existing RL framework.

## Installation

1. If desired, a conda environment can be created using the following command:

```bash
conda create -n <env_name>
```

2.  If needed, the dependencies for the ```toy_example.py``` script can be installed via the requirements.txt file.
```bash
pip install -r requirements.txt
```
<br>    
    Dependencies:
    <ul>
        <li>PyTorch</li>
        <li>Numpy</li>
        <li>Fast Pytorch Kmeans</li>
    </ul><br>
    
3. Install the CL-Buffer library by running the following command inside the directory where the ```setup.py``` file is located.
```bash
pip install .
```


## Usage
Installing the initial state buffer as a library makes it possible to import the buffer using the import statement directly
```bash
from initial_buffer.algorithms.projection_buffer import ProjectionBuffer
```

The ```ProjectionBuffer``` class includes three sampling methods: ['network', 'observations', 'random']. 
The CL-Buffer corresponds to the 'network' sampling strategy.
For the explanation of the other sampling strategies, we refer to the paper.

There are multiple hyperparameters that can be set for the training of the buffer; see the arguments in the ```__init__``` function for the ```ProjectionBuffer``` class in ```initial_buffer/algorithms/projection_buffer.py```. 
Generally, we noticed that the initial state clustering is not affected much by parameters in a similar range as the default parameters.

For a toy example, please have a look at the ```toy_example.py``` script. 
It includes template functions for adding visited experiences to the buffer, training the buffer, and using the buffer for the selection of states.
The visited state buffer is not included in the ```toy_example.py``` since it highly depends on the underlying environment. 
However, the visited state buffer can be implemented relatively easily using a simple array/dict/list storing the states, observations, dones, and rewards of the collected experiences.

