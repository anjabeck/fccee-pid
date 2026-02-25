import argparse
import numpy as np
from xgboost import XGBClassifier
from concurrent.futures import ThreadPoolExecutor
import awkward as ak
import uproot

def read_classifiers(detector, tof, dndx, dedx):
    model_cache = {}
    model_key = f'{tof:.1f}_{dndx:.1f}_{dedx}'
    for pdg in [211, 321, 2212]:
        model_path = f'models/{detector}/{tof:.1f}ps_{dndx:.1f}_{dedx}_{pdg}.ubj'
        model = XGBClassifier()
        model.load_model(model_path)
        model_cache[f"{model_key}_{pdg}"] = model
    return model_cache


def make_prediction(model_cache, data, tof, dndx, dedx, pdgs):

    model_key = f'{tof:.1f}_{dndx:.1f}_{dedx}'

    def predict_parallel(data, pdg):
        model = model_cache[f"{model_key}_{pdg}"]
        return model.predict_proba(data)[:, 1]

    predictions = np.zeros((len(pdgs), data.shape[0]))*np.nan
    with ThreadPoolExecutor() as executor:
        futures = []
        for i, pdg in enumerate(pdgs):
            futures.append(executor.submit(predict_parallel, data, pdg))
        for i, future in enumerate(futures):
            predictions[i, :] = future.result()
    return predictions


def get_features(data, speed_var, flight_var, tof_var, dndx_var, dedx_var, momentum_var, detector):
    assert (flight_var is None and tof_var is None) or (flight_var is not None and tof_var is not None), "Both flight_var and tof_var should be provided together or both should be None."
    features = []
    if tof_var is not None or speed_var is not None:
        if speed_var is None:
            speed_var = "speed"
            speed = (1e6 / 299792458) * (data[flight_var] / data[tof_var])
        else:
            speed = data[speed_var]
        speed = np.clip(speed, 1e-6, 1 - 1e-6)
        speed = -np.log(1 - speed)
        features.append(speed)
    if detector == 'IDEA':
        features.append(data[dndx_var])
    features.append(np.log(data[momentum_var]))
    if dedx_var is not None:
        features.append(np.log(data[dedx_var]))
    return np.asarray(features).T


parser = argparse.ArgumentParser()
parser.add_argument('--input',
                    type=str,
                    help='Input file')
parser.add_argument('--output',
                    type=str,
                    help='Output file')
parser.add_argument('--treename',
                    type=str,
                    help='Name of the ROOT tree in the input file')
parser.add_argument('--detector',
                    type=str, default='IDEA',
                    help='Which detector concept to use',
                    choices=['CLD', 'IDEA'])
parser.add_argument('--dndx_val',
                    type=float, default=0.8,
                    help='dN/dx efficiency',
                    choices=[0.5, 0.8, 1.0])
parser.add_argument('--tof_val',
                    type=float, default=30,
                    help='Timing resolution in ps',
                    choices=[-1, 0, 1, 3, 5, 10, 30, 50, 100, 300, 500])
parser.add_argument('--tof_var',
                    type=str, default=None,
                    help='Which tof variable to use in the input ROOT file')
parser.add_argument('--flight_var',
                    type=str, default=None,
                    help='Which flight variable to use in the input ROOT file')
parser.add_argument('--speed_var',
                    type=str, default=None,
                    help='Which speed variable to use in the input ROOT file')
parser.add_argument('--dndx_var',
                    type=str, default=None,
                    help='Which dN/dx variable to use in the input ROOT file')
parser.add_argument('--dedx_var',
                    type=str, default=None,
                    help='Which dE/dx variable to use in the input ROOT file')
parser.add_argument('--momentum_var',
                    type=str, default='momentum',
                    help='Which momentum variable to use in the input ROOT file')
args = parser.parse_args()

assert (args.speed_var is None) ^ (args.tof_var is None and args.flight_var is None), "If speed_var is provided, both tof_var and flight_var must also be provided."
assert (args.dndx_var is None) ^ (args.detector == 'IDEA'), "For IDEA detector, dndx_var must be provided."
assert (args.speed_var is None) ^ (args.detector == 'CLD'), "For CLD detector, speed_var must be provided."

usededx = "withdedx" if args.dedx_var is not None else "nodedx"

# Preload all models into a dictionary
model_cache = read_classifiers(args.detector, args.tof_val, args.dndx_val, usededx)

# Figure out which features we want
variables = []
for v in [args.speed_var, args.tof_var, args.flight_var, args.dndx_var, args.dedx_var, args.momentum_var]:
    if v is not None:
        variables.append(v)

# Read the data
tree = uproot.open(f"{args.input}:{args.treename}")


pdgs = [211, 321, 2212]
with uproot.recreate(args.output) as outfile:
    first = True
    for chunk in tree.iterate(variables, library="ak", step_size=100_000):

        # Each of these is jagged, shape: [n_events, var_length]
        feats = {feature: chunk[feature] for feature in variables}

        # Remember the structure (the "layout") before flattening
        counts = ak.num(feats[variables[0]])  # how many entries per row

        # Flatten everything to 1D
        flat = {feature: ak.to_numpy(ak.flatten(feats[feature])) for feature in variables}

        # Prepare features for BDT
        features = get_features(flat, args.speed_var, args.flight_var, args.tof_var, args.dndx_var, args.dedx_var, args.momentum_var, args.detector)

        # Predict in parallel for each particle type and store the results
        predictions = make_prediction(model_cache, features, args.tof_val, args.dndx_val, usededx, pdgs)

        # Unflatten and store the predictions in an output dictionary
        preds_and_probs = {}
        for i, pdg in enumerate(pdgs):
            preds_and_probs[f'probability_{pdg}'] = ak.unflatten(predictions[i, :], counts)
        prediction = np.asarray([
            pdgs[idx] for idx in np.argmax(predictions, axis=0)
        ])
        preds_and_probs['prediction'] = ak.unflatten(prediction, counts)
        # Save to file
        if first:
            outfile["pid"] = preds_and_probs
            first = False
        else:
            outfile["pid"].extend(preds_and_probs)
