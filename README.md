# Double-Pendulum
Simulating and visualizing chaotic dynamics of the double pendulum using Python. Includes animations, energy analysis, phase space plots, and earthquake response modeling.

# Chaos in the Double Pendulum

## 🧪 Description
A computational and visual study of deterministic chaos in the classical double pendulum. This repository implements Lagrangian mechanics, simulates phase-space dynamics, explores damping and earthquake effects, and offers animations using open-source Python tools.

---

## 📁 Repository Structure
```bash
├── README.md                          # Project overview
├── LICENSE                            # License file (MIT or similar)
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Ignored files
├── data/
│   └── sample_output/                 # Optional visual/plot output
├── docs/
│   └── Deterministic_Chaos_in_the_Double_Pendulum.pdf  # Full PDF report
├── src/                               # Python source code
│   ├── main_simulation.py             # Core simulation
│   ├── animate_double_pendulum.py     # Animation
│   ├── phase_space_analysis.py        # Phase portraits
│   ├── damping_effects.py             # Damping and energy decay
│   └── earthquake_forcing.py          # Earthquake simulation
├── notebooks/
│   └── double_pendulum_analysis.ipynb # Jupyter notebook walkthrough
└── examples/
    └── sample_initial_conditions.json # Sample inputs
```

---

## 🚀 Getting Started

### 🛠 Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/double-pendulum-chaos.git
cd double-pendulum-chaos
pip install -r requirements.txt
```

### ▶️ Run a Simulation
```bash
python src/main_simulation.py
```

---

## 📷 Features
- Lagrangian formulation of the double pendulum
- Numerical integration using `scipy.integrate`
- Animations via `matplotlib.animation`
- Sensitive dependence on initial conditions
- Phase space analysis: θ–ω diagrams
- Damping simulation and energy decay visualization
- Earthquake forcing and response modeling

---
