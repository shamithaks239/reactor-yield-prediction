# reactor-yield-prediction
Physics-informed hybrid ML model for predicting product yield in a non-isothermal continuous-flow chemical reactor using reaction kinetics and Gaussian Process residual modeling.

# Reactor Yield Prediction

### Physics-Informed Hybrid Machine Learning for Chemical Reactor Yield Prediction

A hybrid predictive modeling approach for estimating the **overall yield of Product B** in a non-isothermal continuous-flow chemical reactor.

The project combines **chemical reaction kinetics, thermal modeling, feature engineering, and Gaussian Process regression** to create a data-driven surrogate capable of predicting reactor performance much faster than repeatedly solving the underlying differential equations.

---

## Problem Statement

The project was developed for the **ML Hackathon: The Predictive Modeling Optimization Challenge**.

Modern chemical plants often rely on computationally expensive physics-based simulations such as CFD and boundary value problem (BVP) solvers to understand reactor behavior. While these methods can accurately represent the underlying process, they can be too slow for real-time optimization.

The objective of this challenge was to build a machine learning surrogate that can rapidly predict the **overall yield of Product B** for new reactor operating conditions.

The reactor follows a series-parallel reaction network:

```text
A  ──k₁──>  B  ──k₂──>  C
```

The competing reactions create nonlinear relationships between operating conditions and the final yield of the desired product.

---

## Dataset

The training dataset contains **150 observations**, while the test dataset contains **50 unseen operating conditions**.

### Input Features

| Feature                | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| `flow_rate_L_min`      | Volumetric flow rate of the reactant mixture (L/min) |
| `concentration_mol_L`  | Inlet concentration of Reactant A (mol/L)            |
| `inlet_temperature_K`  | Feed inlet temperature (K)                           |
| `length_m`             | Reactor length (m)                                   |
| `jacket_temperature_K` | External heating jacket temperature (K)              |

### Target

`overall_yield` — final percentage yield of Product B at the reactor exit.

---

## Approach

Instead of relying exclusively on a black-box machine learning model, the project uses a **physics-guided hybrid architecture**.

### 1. Exploratory Data Analysis

The target distribution and relationships between the operating variables and product yield were investigated.

A notable characteristic of the dataset was the presence of **zero-yield observations**, motivating further investigation into the underlying reactor behavior.

---

### 2. Residence Time Feature

A physically meaningful feature was derived from the reactor geometry and flow rate:

$$
\tau = \frac{L}{Q}
$$

where:

* \(L\) = reactor length
* \(Q\) = volumetric flow rate
* \(\tau\) = effective residence time

This provides the model with information about how long the reactants remain inside the reactor.

---

### 3. Kinetic Model

The reaction network

$$
A \rightarrow B \rightarrow C
$$

was explicitly incorporated into the modeling process.

Temperature-dependent reaction rates were represented using the Arrhenius equation:

$$
k_i = A_i e^{-E_{a,i}/RT}
$$

The kinetic parameters were estimated from the available reactor data.

---

### 4. Thermal Relaxation

Because the reactor is non-isothermal, assuming a constant reactor temperature can be inaccurate.

A thermal relaxation model was introduced:

$$
T(t) =
T_{jacket} +
(T_{inlet}-T_{jacket})e^{-t/\tau_{th}}
$$

where \(\tau_{th}\) represents the thermal relaxation timescale.

A corresponding effective temperature was then incorporated into the kinetic yield model.

---

### 5. Exact ODE Formulation

The reaction system was also represented using ordinary differential equations:

$$
\frac{dC_A}{dt}=-k_1C_A
$$

$$
\frac{dC_B}{dt}=k_1C_A-k_2C_B
$$

The system was solved numerically using `solve_ivp` to investigate the exact kinetic behavior.

---

### 6. Hybrid Physics + Machine Learning Model

The final approach separates the prediction into two components:

$$
\boxed{
\hat{y} =
y_{kinetic} +
y_{residual}
}
$$

The physics-based kinetic model captures the major structure of the reactor behavior.

A **Gaussian Process Regressor (GPR)** is then trained on the residual:

$$
r = y_{actual} - y_{kinetic}
$$

The Gaussian Process learns the remaining nonlinear patterns that are not captured by the simplified kinetic model.

This gives the model both:

* **physical structure**
* **data-driven flexibility**

---

## Model Comparison

Several approaches were evaluated using repeated 5-fold cross-validation.

| Approach                                                 |   CV RMSE |
| -------------------------------------------------------- | --------: |
| Linear Regression                                        |     30.37 |
| Tree Ensembles – Round 1 Features                        |     20.04 |
| Tree Ensembles – Round 2 Features                        |     21.03 |
| Kinetic + GP Residual                                    |     17.26 |
| Kinetic + GP + LMTD                                      |     15.61 |
| Kinetic + GP + Thermal Relaxation                        |     11.65 |
| **Kinetic + GP + Thermal Relaxation + Trimmed Features** | **11.45** |

The results show that incorporating domain knowledge substantially improved predictive performance compared with purely statistical baselines.

---

## Cross-Validation

To reduce dependence on a single train-validation split, the model was evaluated using **Repeated 5-Fold Cross-Validation**.

The final workflow evaluates the model across multiple folds and reports the distribution of RMSE values.

The primary competition metric is:

$$
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

Lower RMSE indicates better predictive performance.

---

## Uncertainty Estimation

An additional advantage of the Gaussian Process model is that it provides an estimate of **prediction uncertainty**.

Along with the predicted yield, the model produces a predictive standard deviation for each test observation.

This makes it possible to identify operating conditions where the model is less confident.

---

## Final Prediction Pipeline

The final workflow is:

```text
Raw Reactor Data
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Physics-Based Feature Engineering
       │
       ├── Residence Time (τ)
       ├── Thermal Features
       └── Reactor Operating Conditions
       │
       ▼
Kinetic Model
       │
       ▼
Kinetic Yield Prediction
       │
       ▼
Residual Calculation
       │
       ▼
Gaussian Process Regression
       │
       ▼
Residual Prediction + Uncertainty
       │
       ▼
Hybrid Prediction
       │
       ▼
Final Product B Yield
```

---

## Tech Stack

* **Python**
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* SciPy
* Gaussian Process Regression
* Numerical ODE solving
* Chemical reaction kinetics
* Cross-validation

---

## Project Structure

```text
reactor-yield-prediction/
│
├── ML-hack-sourcecode.ipynb
├── README.md
├── requirements.txt
│
├── data/
│   ├── train_dataset.csv
│   └── test_dataset.csv
│
└── submission/
    └── submission.csv
```

> Dataset files may be excluded from the repository depending on distribution/competition restrictions.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/reactor-yield-prediction.git
cd reactor-yield-prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook ML-hack-sourcecode.ipynb
```

Update the dataset paths in the notebook according to your local directory structure.

---

## Key Takeaways

### Why hybrid modeling?

A purely data-driven model has to learn the entire reactor behavior from only a small dataset.

By incorporating known chemical relationships, the model receives a useful inductive bias about:

* reaction kinetics
* residence time
* temperature dependence
* thermal dynamics

The machine learning component then focuses on modeling the **unexplained residual behavior**.

### Main insight

The experiments showed that incorporating progressively more realistic thermal and kinetic assumptions significantly improved predictive performance.

The best documented approach achieved a cross-validation RMSE of approximately **11.45**, compared with **30.37** for the linear regression baseline.

---

## Future Improvements

Potential improvements include:

* Bayesian optimization of kinetic parameters
* More rigorous energy-balance modeling
* Physics-informed neural networks
* Neural ODE-based reactor modeling
* Ensemble models combining GPR with tree-based methods
* Better uncertainty calibration
* Automated hyperparameter optimization
* Real-time deployment as a lightweight reactor surrogate
* Integration with process optimization/control systems

---

## Acknowledgements

Developed as part of the **ML Hackathon: The Predictive Modeling Optimization Challenge**.

The project explores the intersection of **machine learning, chemical reaction engineering, numerical modeling, and surrogate modeling**.
