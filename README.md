# TCN-FOG-Cueing-Benchtop
Real-time Temporal Convolutional Networks (TCN) for Freezing of Gait (FOG) detection and cueing control validation using ankle-mounted IMU data.

Master's Thesis: Real-Time Cueing for Freezing of Gait (FOG)

This repository contains the code base for the Master's thesis focusing on the benchtop validation of an open-source system for on-demand cueing of FOG in home settings, using a Temporal Convolutional Network (TCN).

🎯 Thesis Objectives

Train a real-time TCN model to detect FOG.

Integrate a single threshold cueing control.

Integrate a finite-state-machine cueing control.

Characterize performance (latency, FP correction) of the two controls.

Investigate predictive performance of the TCN model.

📁 Repository Structure

.
├── data/
│   └── PD0XX_WalkingTurning_FOG.npz (PLACE YOUR DATA HERE)
├── src/
│   ├── data_loader.py       # Data loading, reshaping (N, L, C -> N, C, L), and splitting
│   └── model_tcn.py         # TCN architecture definition and training/evaluation logic
├── .gitignore               # Files ignored by Git
└── requirements.txt         # Project dependencies


🚀 Setup and Installation

Clone the Repository:

git clone [YOUR_REPOSITORY_URL]
cd [YOUR_REPOSITORY_NAME]


Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`


Install Dependencies:

pip install -r requirements.txt


Data Placement (Crucial Step):

Create a data/ directory in the root of the repository.

Place the Walking-Turning and FOG dataset file (e.g., PD0XX_WalkingTurning_FOG.npz) into the data/ folder. The data_loader.py script expects this location.

▶️ Running the Model (Objective 1 - Initial Training)

To run the initial training, execute the main script:

python src/model_tcn.py


Note: The model_tcn.py contains placeholders for the full training loop, including LOSO-CV logic, which you and your partner can now fill in.
