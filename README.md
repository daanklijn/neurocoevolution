# Deep NeuroCoevolution
This repository contains the code for the paper [A Coevolutionary Approach to Deep Multi-agent Reinforcement Learning](https://arxiv.org/pdf/2104.05610.pdf).

[Paper](https://arxiv.org/pdf/2104.05610.pdf)

[Poster](https://raw.githubusercontent.com/daanklijn/neurocoevolution/main/poster.png)

[Youtube video](https://www.youtube.com/watch?v=GOpzjJzpWrw)



# Running the experiments

1. Install the dependencies using either `poetry install` or `pip install -r requirements.txt`. 
2. Download the required Atari ROMs by running the `AutoROM` command.
3. If you want to use Weights and Biases for dashboarding, make sure you set the `WANDB_API_KEY`.
4. Specify the config file you want to use in `main.py`. You might want to use one of the configs in the `configs` folder. 
5. Run `main.py`.

If you used Weights and Biases, the metrics and videos should appear in your dashboard:

<img width="1513" alt="Screenshot 2021-05-11 at 17 41 28" src="https://user-images.githubusercontent.com/27863547/117844633-21f20800-b280-11eb-883f-0990ebab8f98.png">

