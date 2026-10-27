# Machine Learning Based Smart Optimization of Cooling Water Air Conditioning System for Energy and Water Efficiency

An end-to-end machine-learning project that classifies chiller **cooling-water operating
points** as `OPTIMAL` or `SUBOPTIMAL`, measures the model with a full classification
metric suite, and presents **overall project performance on a single dashboard**.

---

## 1. What is in this project

| File | Purpose |
|---|---|
| `ml_model.py` | Loads/generates the dataset, trains a RandomForest, computes every metric, exports artifacts |
| `app.py` | Dash dashboard showing **overall project performance** |
| `requirements.txt` | Python dependencies |
| `data/cooling_water_dataset.csv` | Dataset (created on first run, or drop your own CSV here) |
| `output/metrics.json` | All computed metrics |
| `output/model.joblib` | Serialized trained model |
| `output/*.png` | Metrics bar, confusion matrix, ROC curve, feature importance plots |

---

## 2. Quick start (VS Code)

**Prerequisite: Python 3.10+**. If `python --version` says *Python was not found*,
install it first with either:

```bash
winget install Python.Python.3.12      # Windows, no admin prompt needed
# or download the installer from https://www.python.org/downloads/
```

(check **Add python.exe to PATH** in the installer), then:

```bash
# 1. open the folder in VS Code, then open a terminal (Ctrl + `)

# 2. create + activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 3. install dependencies
pip install -r requirements.txt

# 4. train the model and generate all metrics
python ml_model.py

# 5. launch the performance dashboard
python app.py                  # -> http://127.0.0.1:8050
```

Use your own data instead of the synthetic benchmark:

```bash
python ml_model.py --csv path/to/your_dataset.csv
```

The CSV must contain the five sensor columns
`delta_T, flow_rate, pump_speed, outdoor_temp, occupancy` plus an optional binary
`efficiency_class` column.

---

## 3. Metrics reported

Every metric is written to `output/metrics.json` and rendered on the dashboard.

**Headline (requirement: >= 90%)**

* Accuracy
* Precision
* Recall (Sensitivity)
* F1-Score
* ROC-AUC

**Supporting**

* Specificity
* Balanced Accuracy
* Matthews Correlation Coefficient (MCC)
* Per-class precision / recall / F1 / support
* Confusion matrix (TN, FP, FN, TP)
* Stratified 5-fold cross-validation (mean + std + per-fold scores)
* ROC curve coordinates
* Gini feature importances

The training run prints an explicit `PASS` flag per metric against the 90% gate, and the
dashboard shows the same gate at the top.

---

## 4. Dashboard (`app.py`)

One page, eight panels, all reading from `output/metrics.json`:

1. 90% requirement gate banner
2. KPI scorecard - all eight headline metrics with PASS / BELOW TARGET flags
3. Headline metric bar chart vs. the 90% target line
4. Confusion matrix heatmap
5. ROC curve
6. Sensor feature importance
7. Per-class metric table + confusion-matrix detail
8. 5-fold cross-validation stability + run provenance (dataset, model, timestamp)

---

## 5. Novelty

1. **Cooling-water-centric efficiency labelling** - instead of predicting raw power draw,
   the target marks whether a condenser-water operating point sits inside the chiller's
   high-efficiency envelope, which directly couples *energy* (kW) and *water* (recirculated
   flow / cycles of concentration) performance in one decision.

2. **Physics-informed synthetic benchmark with a learnable boundary** - features are
   generated from chiller-side thermodynamic relations (load, wet-bulb, flow and pump
   speed drive the approach temperature), so the decision boundary is non-linear and
   class-balanced rather than trivially separable. Drop in a real CSV and the identical
   pipeline trains on it.

3. **Multi-metric acceptance gate, not accuracy alone** - the pipeline gates on accuracy,
   precision *and* recall simultaneously plus MCC and balanced accuracy, so a
   majority-class shortcut cannot pass. Cross-validation variance is reported next to the
   hold-out score to expose overfitting.

4. **Sensor attribution for retrofit decisions** - Gini importances rank which of the five
   instrumented sensors actually drive efficiency, turning the classifier into a
   prioritisation tool for where to install better instrumentation or add control logic.

5. **Single-page performance accountability** - the dashboard merges model quality,
   requirement compliance, class-level behaviour and run provenance into one screen, so a
   supervisor or reviewer can audit the whole project without opening a notebook.

6. **Edge-deployable artifact** - `output/model.joblib` is a compact tree ensemble that can
   be exported to ONNX/PMML for a building-management-system or PLC gateway, enabling
   online adaptive updates of the efficiency decision at the edge.

---

## 6. Project structure

```
ml-cooling-water-optimization/
├── app.py                # performance dashboard
├── ml_model.py           # data -> train -> metrics -> artifacts
├── requirements.txt
├── README.md
├── data/
│   └── cooling_water_dataset.csv   (generated on first run)
└── output/
    ├── metrics.json
    ├── model.joblib
    ├── metrics_bar.png
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── feature_importance.png
```
