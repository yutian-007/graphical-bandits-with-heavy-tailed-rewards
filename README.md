# Stochastic Graphical Bandits with Heavy-Tailed Rewards

This repository is the official implement of Stochastic Graphical Bandits with Heavy-Tailed Rewards (UAI 2023).

## Instructions

### 1. How to run

- Environment: Linux, Ubuntu-20.04.
- Python version: 3.8.13.
- Install packages: numpy, matplotlib, abc, copy, time.
- Run `python3 run.py` to excute main function.

### 2. Other Specific Functions

- ArmClass.py -> class of arms, including class `BinomialArm` and `ParetoArm`.
- feedback_graph.py -> class of feedback graph, including class `FeedbackGraph`.
- UCB.py -> implements of two UCB-type algorithms, including our methods RUNE-TEM, RUNE-MoM and UCB-N in "Stéphane Caron, Branislav Kveton, Marc Lelarge, and Smriti Bhagat. Leveraging side observations in stochastic bandits. In Proceedings of the 28th Conference on Uncertainty in Artificial Intelligence, page 142–151, 2012.".
- AAE.py -> implements of two elimination-based algorithms, including our methods RAAE-TEM, RAAE-MoM and AAE-AlphaSample in "Alon Cohen, Tamir Hazan, and Tomer Koren. Online learning with feedback graphs without the graphs. In Proceedings of the 33rd International Conference on Machine Learning, pages 811–819, 2016.".
