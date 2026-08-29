## Project Website

[Autonomous Green Hydrogen Power Plant Controller](https://jaimins2002-netizen.github.io/)

## Project Website


# Autonomous Green Hydrogen Power Plant Controller — Milestone 2

Milestone 2 contains the uploaded **chemical-controller Jupyter notebook** for the autonomous green hydrogen power plant project. The notebook implements the core Mamdani fuzzy-logic controller and demonstrates how four plant measurements are transformed into a hydrogen production command.

## Controller Inputs and Output

| Signal | Role | Range | Linguistic terms |
|---|---|---:|---|
| Renewable power | Input | 0–100 kW | Low, Medium, High |
| Water flow rate | Input | 0–20 L/min | Low, Medium, High |
| Stack temperature | Input | 20–80 °C | Low, Normal, High |
| Hydrogen-tank pressure | Input | 0–100 bar | Low, Medium, High |
| Hydrogen production rate | Output | 0–10 kg/h | Off, Low, Medium, High |

The controller uses triangular membership functions, an 11-rule expert knowledge base, Mamdani inference, and defuzzification. Hydrogen-tank pressure is safety-critical, so the demonstrations include reduced production behavior as pressure approaches the high-pressure region.

## Repository Contents

This Milestone 2 repository intentionally contains the notebook implementation only, as requested:

- `ipynb/Milestone_2_Chemical_Controller.ipynb` — uploaded Milestone 2 chemical-controller notebook with its explanations, demonstrations, and saved outputs.
- `requirements.txt` — dependencies required to run the notebook.
- `.github/workflows/ci.yml` — automated validation workflow.

Older notebook versions and generated Python exports have been removed so this repository has one clear Milestone 2 implementation source.

## Executed Phase Output

A fresh executed copy is stored at `ipynb/executed/Phase_3_Core_Controller/Milestone_2_Chemical_Controller_executed.ipynb`. It contains the completed controller demonstrations and final SCADA-style interface output.

## Installation and Usage

Create a Python environment, install the dependencies, and open the notebook:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ipynb/Milestone_2_Chemical_Controller.ipynb
```

## Created By

This project was created by **Jaimin Sanghani and  Krupa Ashishkumar Rajput,  Harsh Shingala, and Makwana Shlock.

## Thank You

Thank you for reviewing and supporting this autonomous green hydrogen power plant controller project.

## Safety Disclaimer

This repository contains an educational simulation. It is **not** a certified process-safety system and must not be used to control real hydrogen-production equipment without qualified engineering validation, independent hardware safeguards, regulatory review, and professional oversight.

## License

No license has been specified for this milestone.

## Final SCADA-Style Interface

The supplied notebook’s final implementation cell renders the SCADA-style fuzzy-electrolyzer interface. It includes live plant controls, hydrogen production output, membership weights, the 11-rule firing matrix, centroid calculation, daily yield estimate, fuzzy membership-function visualization, and a multi-trace telemetry monitor.

Launch the interface from the repository with:

```bash
cd autonomous-green-hydrogen-power-plant-controller-milestone-2
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook ipynb/Milestone_2_Chemical_Controller.ipynb
```

Open the notebook, select **Run → Run All Cells**, and scroll to the final cell. The final cell displays the SCADA interface in the notebook. JupyterLab can be used instead with `jupyter lab ipynb/Milestone_2_Chemical_Controller.ipynb`.
