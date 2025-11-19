
---
# Restricted Boltzmann Machine 
---
## Overview

This project demonstrates the learning mechanism of Boltzmann machine. Using MNIST digit dataset, the model was tasked to learn the patterns of the handwritten digits and reconstruct the damaged images from what the model learned.

## How to Run
```
# Git command clonning repo
git clone https://github.com/markna627/rbm.git
cd rbm

# Dependency installation
pip install -r requirements.txt

# Default parameters
train.py --epochs 50 --cdk 5
```
Available Arguments:
```
--epochs   (int)  
--cdk      (int)  sampling steps for contrastive divergence k value
```
Colab demo is available:

[Here](https://colab.research.google.com/drive/1lZktsWvnVs3kk3NC7WKeX9ptFxZovOar?usp=sharing)




## Example:
The images with noises were reconstructed after training, and below are few example results. 

![Reconstructed Image of 6](<img width="618" height="291" alt="Screenshot 2025-11-18 at 8 59 33 PM" src="https://github.com/user-attachments/assets/bb526ccf-77ba-4436-abca-61986f353127" />)

![Reconstructed Image of 3](<img width="612" height="285" alt="Screenshot 2025-11-18 at 8 59 49 PM" src="https://github.com/user-attachments/assets/dd9a789f-e9a5-4f3c-ade2-90a2a300dc1c" />)
## Notes
* The model was designed so that there are only connections between visible and hidden nodes, hence Restricted Boltzmann Machine for simpler training step.
* This implmentation closely follows the Boltzmann machine and its learning algorithm proposed by Geoffrey E. Hinton, the nobel laureate for his work on this matter.
* Mathematical derivations for the update rule is introduced in the provided Colab [link](https://colab.research.google.com/drive/1lZktsWvnVs3kk3NC7WKeX9ptFxZovOar?usp=sharing).

## Related Works

* Hinton, G. E., & Sejnowski, T. J. (1985). A Learning Algorithm for Boltzmann Machines.
Cognitive Science, 9(1), 147–169. 





