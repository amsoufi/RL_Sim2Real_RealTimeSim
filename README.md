# RL with and without Real-Time Simulation for Kinova Gen3

This project explores reinforcement learning (RL) applied to robotic manipulation using a Kinova Gen3 robot, under two conditions:
- **NoRTSim/**: Training with deterministic, fixed-timestep simulation.
- **RTSim/**: Training with real-time simulation, introducing dynamics noise and delay.

Pretrained models are available in the `Test/` directory for evaluation purposes.

This work is associated with [this paper](https://ieeexplore.ieee.org/abstract/document/10196019).

Much inspiration and implementation guidelines were taken from [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/).

---

## 📁 Project Structure

- `NoRTSim/` – Training with deterministic (fixed timestep) simulation.
- `RTSim/` – Training with real-time simulation.
- `Test/` – Pretrained agents evaluation.

Each environment contains its own `main.py` as the starting point.

> **Note:** The `kinova_sim/` modules differ across environments and are **not unified**, as they have custom URDFs and control logic.

---

## 🚀 Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/amsoufi/RL_Sim2Real_RealTimeSim.git
   cd RL_Sim2Real_RealTimeSim
   ```

2. **Install the requirements**:
   This project was developed and tested with Python 3.6

   ```bash
   pip install -r requirements.txt
   ```

3. **Train an agent**:

   - Without real-time simulation:
     ```bash
     cd repo/NoRTSim
     python main.py
     ```

   - With real-time simulation:
     ```bash
     cd repo/RTSim
     python main.py
     ```

4. **Test a pretrained agent**:

   ```bash
   cd repo/Test
   python main.py
   ```

   ➡️ **Choosing Different Models**:  
   Inside `Test/main.py`, you can manually change which pretrained model to load by editing the call:
   ```python
   ac = torch.load('Agents/ppo_model_RTR.pt')
   ```
   Select the appropriate model from `Agents/`.

---

## 🎛 Domain Randomization

You can modify simulation properties for domain randomization by editing:
- The robot URDF files inside `kinova_sim/resources/`
- The code inside `kinova_sim/resources/robot.py`

Variables like `J1`, `J2`, etc., can be exposed and modified dynamically.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you find this project useful, please cite:

> A. M. S. Enayati et al., "[Facilitating Sim-to-Real by Intrinsic Stochasticity of Real-Time Simulation in Reinforcement Learning for Robot Manipulation](https://ieeexplore.ieee.org/abstract/document/10196019)", IEEE, 2023.

---
