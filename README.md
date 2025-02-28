**The Bandit Whisperer: Communication Learning for Restless Bandits**
==================================

Applying Reinforcement Learning (RL) to Restless Multi-Arm Bandits (RMABs) offers a promising avenue for addressing allocation problems with resource constraints and temporal dynamics. However, classic RMAB models largely overlook the challenges of (systematic) data errors - a common occurrence in real-world scenarios due to factors like varying data collection protocols and intentional noise for differential privacy. We demonstrate that conventional RL algorithms used to train RMABs can struggle to perform well in such settings. To solve this problem, we propose the first communication learning approach in RMABs, where we study which arms, when involved in communication, are most effective in mitigating the influence of such systematic data errors. In our setup, the arms receive Q-function parameters from similar arms as messages to guide behavioral policies, steering Q-function updates. We learn communication strategies by considering the joint utility of messages across all pairs of arms and using a Q-network architecture that decomposes the joint utility. Both theoretical and empirical evidence validate the effectiveness of our method in significantly improving RMAB performance across diverse problems.



## Setup

Main file is `agent_oracle.py`

- Clone the repo:
- Install the repo:
- `pip3 install -e .`
- Create the directory structure:
- `bash make_dirs.sh`

To run Synthetic dataset from the paper, run 
`bash run/jobs.run_synthetic.sh`

Code adapted from https://github.com/killian-34/RobustRMAB, the github. The authors thank Jackson Killian for discussions.
