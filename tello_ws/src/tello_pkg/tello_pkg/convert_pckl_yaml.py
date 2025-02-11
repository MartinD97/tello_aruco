import pickle
import yaml

# Carica il file di calibrazione in formato Pickle
with open("calibration_camera_pc.pckl", "rb") as f:
    calib_data = pickle.load(f)

# Salva i dati in formato YAML
with open("calibration_camera_pc.yaml", "w") as f:
    yaml.dump(calib_data, f, default_flow_style=False)

print("Conversione completata: calibrazione.yaml salvato!")
