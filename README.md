# Autonomous Green Hydrogen Power Plant Controller — Milestone 2

Milestone 2 implements the **core Mamdani fuzzy-logic controller** for an autonomous green hydrogen power plant. The implementation uses `scikit-fuzzy` to transform four plant measurements into a hydrogen production command.

## Controller Inputs and Output

| Signal | Role | Range | Linguistic terms |
|---|---|---:|---|
| Renewable power | Input | 0–100 kW | Low, Medium, High |
| Water flow rate | Input | 0–20 L/min | Low, Medium, High |
| Stack temperature | Input | 20–80 °C | Low, Normal, High |
| Hydrogen-tank pressure | Input | 0–100 bar | Low, Medium, High |
| Hydrogen production rate | Output | 0–10 kg/h | Off, Low, Medium, High |

The controller includes triangular membership functions, Mamdani inference rules, defuzzification, and demonstration cases. Hydrogen-tank pressure is safety-critical and the controller is designed to reduce production as pressure approaches the high-pressure region.

## Contents

- `Autonomous_Green_Hydrogen_Plant_Controller_Fuzzy.ipynb` — primary implementation notebook.
- `Autonomous_Green_Hydrogen_Plant_Controller_Fuzzy.py` — Python export of the primary notebook.
- `Autonomous_Green_Hydrogen_Plant_Controller_Fuzzy_validated.ipynb` and `.py` — validated implementation exports.
- `Phase_3_Core_Controller.ipynb` and `.py` — core controller phase notebook and Python export.
- `Phase_3_Core_Controller_executed.ipynb` and `.py` — executed notebook and Python export.
- `requirements.txt` — Python dependencies.

## Installation and Usage

Create a Python environment, install the dependencies, and run a notebook from top to bottom:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

The exported `.py` files can also be run directly with Python.

## Safety Disclaimer

This repository contains an educational simulation. It is **not** a certified process-safety system and must not be used to control real hydrogen-production equipment without qualified engineering validation, independent hardware safeguards, regulatory review, and professional oversight.

## Authors

Krupa Ashishkumar Rajput; Jaimin Sanghani; Harsh Shingala; Makwana Shlock.

## License

No license has been specified for this milestone.
