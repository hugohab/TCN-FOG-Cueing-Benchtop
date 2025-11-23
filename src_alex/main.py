from src_alex.train import train
from src_alex.evaluate import evaluate

def main():
    # Path to your dataset file (.npz)
    # npz_file = "TCN-FOG-CUEING-BENCHTOP/Ankle_Sensor_Data_Semi_Free_Living_Conditions_FoG/PD001_AllActivities_FOG.npz"
    data_folder = "TCN-FOG-CUEING-BENCHTOP/Ankle_Sensor_Data_Semi_Free_Living_Conditions_FoG"

    # Train model
    model, test_loader = train(
        npz_path=data_folder,# npz_file if just 1 file
        epochs=100,
        batch_size=128,
        lr=1e-3,
        debug_single_file = False
    )

    # Evaluate model
    evaluate(
        model=model,
        dataloader=test_loader,
        show_plots=True,  # change to False for silent mode
        threshold= 0.5
    )

if __name__ == "__main__":
    main()