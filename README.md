# -smart-grid-ML-RL-[README (1).md](https://github.com/user-attachments/files/26428253/README.1.md)
# Smart Grid: ML Load Forecasting + RL Battery Control

## Stucture

```
smart_grid_project/
├── 1_prepare_data.py     # ML LSTM
├── 2_rl_environment.py   # Environment Gym 
├── 3_train_ppo.py        # RL -PPO 
├── 4_evaluate.py         # 
├── saved_models/         # 
└── results/              # 
```

## design

```
Historical Load (UCI Dataset)
        │
        ▼
  LSTM Forecaster          ← 5 INPUT
        │
        ▼
  predicted_load(t+1)
        │
        ▼
  PPO RL Agent             ← state = [predicted_load, battery_soc]
        │
        ▼
  action ∈ {-1, 0, 1}     ← Control [discharge / no opreat / charge]
        │
        ▼
  Battery Environment      ←  actual_load + tou_price
        │
        ▼
  reward = -cost
```

## install depand

```bash
pip install torch pandas numpy scikit-learn matplotlib
pip install stable-baselines3[extra] gymnasium
```

## Step

### 1 data download
1. web: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption
2.  `household_power_consumption.txt`
3. floder name: `smart_grid_project/` 

### 2： LSTM (ML training)
```bash
python 1_prepare_data.py
```
document output：
- `saved_models/lstm_model.pth`
- `saved_models/scalers.pkl`
- `saved_models/test_data.csv`

### RL: env bulid
```bash
python 2_rl_environment.py
```
out put：`Random policy — 48h total cost: $X.XXXX`

### 4： PPO training
```bash
python 3_train_ppo.py
```
`saved_models/ppo_battery.zip` + `saved_models/best_model.zip`

### evaluate
```bash
python 4_evaluate.py
```
输出：
- `results/policy_comparison.png`  
- `results/summary_table.csv`      

## RL variable

| V| num | state
|------|--------|------|
| `battery_cap` | 10.0 kWh | capacity
| `charge_rate` | 2.5 kW | max power
| `battery_eff` | 0.95 | efficinecy
| `initial_soc` | 0.5 |  50% |

## compare policy

1. **Always Idle** — never change
2. **Always Charge** — always charge
3. **Rule-Based** — low price charge high price discharge
4. **PPO (Ours)** — our way

## 

Multivariable time-series inputs (5 lags) are used to generate single-step load forecasts. The PPO agent makes decisions based only on the predicted load and the battery state (a 2-dimensional state space), implementing a Forecast-Driven Reinforcement Learning approach.

## data

Hebrail, G. & Berard, A. (2012). Individual Household Electric Power
Consumption [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C58K54
