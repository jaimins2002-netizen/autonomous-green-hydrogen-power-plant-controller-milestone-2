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

The uploaded course implementation is included as the authoritative chemical-controller version:

- `ipynb/Milestone_2_Chemical_Controller.ipynb` — uploaded Milestone 2 notebook, preserved with its demonstrations and outputs.
- `python/Milestone_2_Chemical_Controller.py` — Python export generated from the uploaded notebook.

- `ipynb/Autonomous_Green_Hydrogen_Plant_Controller_Fuzzy.ipynb` — primary implementation notebook.
- `python/Autonomous_Green_Hydrogen_Plant_Controller_Fuzzy.py` — Python export of the primary notebook.
- `ipynb/Autonomous_Green_Hydrogen_Plant_Controller_Fuzzy_validated.ipynb` and `python/Autonomous_Green_Hydrogen_Plant_Controller_Fuzzy_validated.py` — validated implementation exports.
- `ipynb/Phase_3_Core_Controller.ipynb` and `python/Phase_3_Core_Controller.py` — core controller phase notebook and Python export.
- `ipynb/Phase_3_Core_Controller_executed.ipynb` and `python/Phase_3_Core_Controller_executed.py` — executed notebook and Python export.
- `requirements.txt` — Python dependencies.

## Installation and Usage

Create a Python environment, install the dependencies, and run a notebook from top to bottom:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

The exported files in `python/` can also be run directly with Python, for example `python python/Milestone_2_Chemical_Controller.py`. The uploaded notebook and generated export are the primary chemical-controller implementation for this Milestone 2 update.

## Safety Disclaimer

This repository contains an educational simulation. It is **not** a certified process-safety system and must not be used to control real hydrogen-production equipment without qualified engineering validation, independent hardware safeguards, regulatory review, and professional oversight.

## Authors

Krupa Ashishkumar Rajput; Jaimin Sanghani; Harsh Shingala; Makwana Shlock.

## Project Website

Visit the project landing page and milestone-wise website: [https://jaimins2002-netizen.github.io/](https://jaimins2002-netizen.github.io/). The Milestone 2 project page is [https://jaimins2002-netizen.github.io/milestone2/](https://jaimins2002-netizen.github.io/milestone2/).

## License

No license has been specified for this milestone.
