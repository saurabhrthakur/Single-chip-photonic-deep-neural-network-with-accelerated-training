# Replication and Analysis of a Photonic Deep Neural Network

This repository contains the source code and documentation for my final-year B.Sc. thesis project at the University of Delhi. The project focuses on the replication and analysis of the groundbreaking paper: "[Single chip photonic deep neural network with accelerated training](https://arxiv.org/abs/2208.01623)" by Saumil Bandyopadhyay, Dirk Englund, et al.

## 🎯 Project Goals

The primary objectives of this project are:
1.  **Replication:** To faithfully replicate the experimental results of the Fully-Integrated Coherent Optical Neural Network (FICONN) architecture, including:
    * The standard backpropagation-trained digital benchmark model.
    * The novel *in-situ* training algorithm based on stochastic optimization.
2.  **Analysis & Contribution:** To move beyond replication by conducting a sensitivity analysis on the *in-situ* training method. This involves studying how the model's performance and robustness are affected by changes in key hyperparameters and initial conditions.

## 🛠️ Architecture Overview

The FICONN architecture is an optical neural network implemented on a photonic integrated circuit. Its key components, simulated in this project, include:
* **Coherent Matrix Multiplication Units (CMXUs):** Implemented as unitary matrices using the Clements decomposition.
* **Nonlinear Optical Function Units (NOFUs):** Modeled using a `tanh` activation function as per the paper's digital benchmark.

## 💾 Dataset

The model is trained on the **Hillenbrand vowel classification dataset**, which consists of acoustic measurements of 12 different vowels. The input features are the first three formants (F1, F2, F3) at two different time points, creating a 6-dimensional input vector.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[saurabhrthakur]/[Single-chip-photonic-deep-neural-network-with-accelerated-training].git
    cd [Single-chip-photonic-deep-neural-network-with-accelerated-training]
    ```
2.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the training script:**
    * To run the digital benchmark model:
        ```bash
        python train.py --mode digital
        ```
    * To run the *in-situ* training model:
        ```bash
        python train.py --mode insitu
        ```

## 📈 Current Status

*This section should be updated as you make progress.*
* **Digital Benchmark Model:** [e.g., Achieved X% test accuracy after 10,000 epochs, successfully replicating the overfitting behavior described in the paper.]
* **In-Situ Training Model:** [e.g., Implementation complete. Currently debugging the training loop and investigating performance on the dataset.]
* **Sensitivity Analysis:** [e.g., Planned experiments include analyzing the impact of weight initialization and perturbation size `δ`.]

## 🙏 Acknowledgments

This project would not be possible without the foundational work of the original authors: Saumil Bandyopadhyay, Alexander Sludds, Stefan Krastanov, Ryan Hamerly, Nicholas Harris, Darius Bunandar, Matthew Streshinsky, Michael Hochberg, and Dirk Englund.
