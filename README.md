# Robot Ball-in-Hole with SAC + Imitation

## 📌 Overview
This project explores training a reinforcement learning (RL) agent to solve a **robot ball-in-hole task**. The environment simulates a robot gripper holding a club to strike a ball into a hole.  

The challenge:  
- Sparse/delayed rewards (success only when ball sinks).  
- High variance in trajectories (ball dynamics, noisy gripper).  
- Need for both **precision** (grasp & strike) and **robustness** (handle different start states).  

I combined **manual demonstrations**, **behavior cloning**, and **reinforcement learning fine-tuning** to build a policy that can consistently complete the task.

---

## 🛠️ Methods

### 1. Manual Demos
- Collected teleoperated demonstrations of “good” behaviors:  
  - Proper grasp of the club.  
  - Striking the ball.  
  - Driving the ball toward or into the hole.  

### 2. Behavior Cloning (BC)
- Supervised learning baseline.  
- Input: RGB observations + proprioception.  
- Output: continuous control actions (gripper, arm).  
- Loss: weighted MSE, with higher weight on gripper actions (critical for grasp).  
- Early stopping based on validation MSE.  

### 3. Reinforcement Learning (SAC)
- Soft Actor-Critic (SAC) initialized from BC weights.  
- Parallel environments for sample efficiency.  
- Reward shaping: ball-in-hole success, progress bonus (distance to hole).  
- Entropy regularization (`ent_coef="auto"`) for exploration.  
- Models selected by **mean − 0.5·std** of returns → favors stable policies.  

### 4. Evolutionary Fine-Tuning (Experimental)
- Population-based search with mutations in policy weights.  
- Elite selection across generations.  
- Tested but often led to instability and high variance compared to BC+SAC.  

---

## ✅ Validation
- Evaluated models visually and quantitatively:  
  - **Mean return** over 50–100 eval episodes.  
  - **Std of return** to measure robustness.  
- Best models showed stable grasp-and-strike behavior, though variance remains a challenge.  
- Self-play / auto-data collection pipeline was designed for future improvement (DAgger-style).  

---

## 📂 Project Structure
```
.
├── demos/                # Human + auto-collected demonstration data
├── models/               # Trained checkpoints (BC + SAC + Evo runs)
├── evolve_sac_seeded.py  # Evolutionary training script
├── train_bc.py           # Behavior cloning trainer
├── train_sac.py          # SAC fine-tuning script
├── eval.py               # Run policy visually or in batch eval
└── README.md             # Project description
```

---

## 🚀 How to Run

### Train Behavior Cloning
```bash
python train_bc.py --demos demos/human_demos.npz --out models/bc
```

### Fine-tune with SAC
```bash
python train_sac.py --init models/bc/best.pt --out models/sac_run1
```

### Evaluate a Trained Model
```bash
python eval.py --model models/sac_run1/best.zip --episodes 20 --render
```

### (Optional) Evolutionary Search
```bash
python evolve_sac_seeded.py --model_path models/sac_run1/best.zip --pop 16 --elites 4 --gens 10
```

---

## 📊 Lessons Learned
- **Manual demonstrations are key**: BC jumpstart is far more stable than pure RL.  
- **Variance matters**: Best policies balance high mean score with low std.  
- **Evolution was noisy**: SAC + BC outperformed population methods in reliability.  
- **Next step**: Self-improving loop (DAgger + SIL) could reduce the need for more human demos.  

---

## 📌 Skills Demonstrated
- Reinforcement learning (SAC, evolutionary search).  
- Imitation learning (behavior cloning, dataset curation).  
- Experimentation with stability/variance trade-offs.  
- Environment debugging, reward shaping, and evaluation metrics.  
