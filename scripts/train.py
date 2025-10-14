from sklearn.model_selection import train_test_split
from qcnn.model import qcnn
from qcnn.datasets import generate_line_dataset
import argparse
from pathlib import Path
from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import sys
from datetime import datetime

def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="QCNN training (matches notebook).")
    parser.add_argument("--width", type=int, required=True, help="image width")
    parser.add_argument("--height", type=int, required=True, help="image height")
    parser.add_argument("--maxiter", type=int, default=200, help="COBYLA max iterations")
    parser.add_argument("--log", type=str, default="train.log", help="log file path")
    parser.add_argument("--min-l", type=int, default=2, help="min line length")
    parser.add_argument("--max-l", type=int, default=2, help="max line length")
    parser.add_argument("--outdir", type=str, default="runs", help="parent folder for runs")
    parser.add_argument("--n-samples", type=int, default=200, help="dataset size")
    parser.add_argument("--model", type=str, default=None, help="path to an existing classifier.model to resume training from")
    parser.add_argument("--rhobeg", type=float, default=1.0, help="COBYLA initial trust region")
    args = parser.parse_args()

    # make a new run folder every time
    run_dir = Path(args.outdir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # saved model file path
    model_path = Path(args.model) if args.model else None
    if model_path and not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # build QCNN 
    width, height = args.width, args.height
    _, qnn = qcnn(width * height)

    # generate dataset
    images, labels = generate_line_dataset(args.n_samples, width=width, height=height, min_l=args.min_l, max_l=args.max_l)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=246
    )

    # output logging
    log_path = run_dir / "train.log"
    log_f = log_path.open("a", encoding="utf-8")

    objective_func_vals = []

    def log_line(msg: str):
        print(msg, flush=True)
        print(msg, file=log_f, flush=True)

    # iteration and loss callback (logging and graph)
    def callback_graph(weights, obj_func_eval):
        objective_func_vals.append(obj_func_eval)
        it = len(objective_func_vals)
        log_line(f"[iter {it:04d}] loss={float(obj_func_eval):.8f}")    

    # Classifier
    if model_path:
        log_line(f"Loading classifier from: {model_path}")
        classifier = NeuralNetworkClassifier.load(str(model_path))
        classifier.warm_start = True
        classifier.optimizer.set_options(maxiter=args.maxiter, rhobeg=args.rhobeg)
        try:
            classifier.callback = callback_graph
        except Exception:
            log_line("[warn] Could not set callback on loaded classifier; continuing.")
    else:
        classifier = NeuralNetworkClassifier(
            qnn,
            optimizer=COBYLA(maxiter=args.maxiter),
            callback=callback_graph,
            warm_start=True,
            rhobeg=args.rhobeg,
        )

    # Training data accuracy
    x = np.asarray(train_images)
    y = np.asarray(train_labels)
    classifier.fit(x, y)

    acc = np.round(100 * classifier.score(x, y), 2)
    log_line(f"Accuracy from the train data : {acc}%")

    # Test Dataset
    x_test = np.asarray(test_images, dtype=float)
    y_test = np.asarray(test_labels, dtype=int)
    y_pred = classifier.predict(x_test)
    test_acc = np.round(100 * classifier.score(x_test, y_test), 2)
    log_line(f"Accuracy from the test data : {test_acc}%")
    
    # Save classifier in the run folder
    model_out = model_path if model_path else (run_dir / "classifier.model")

    try:
        classifier.save(str(model_out))
        log_line(f"Saved classifier to {model_out}")
    except Exception as e:
        log_line(f"[warn] Could not save classifier: {e}")

    # Save the objective curve plot to PNG in the run folder
    try:
        if len(objective_func_vals):
            fig_out = run_dir / "objective_curve.png"
            plt.figure()
            plt.title("Objective function value against iteration")
            plt.xlabel("Iteration")
            plt.ylabel("Objective function value")
            plt.plot(range(1, len(objective_func_vals) + 1), objective_func_vals)
            plt.tight_layout()
            plt.savefig(fig_out)
            plt.close()
            log_line(f"Saved objective curve to {fig_out}")
        else:
            log_line("No objective values recorded; plot not created.")
    finally:
        log_f.close()

if __name__ == "__main__":
    sys.exit(main())