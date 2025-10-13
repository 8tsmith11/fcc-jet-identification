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
    parser.add_argument("--max-l", type:int, default=2, help="max line length")
    args = parser.parse_args()

    # make a new run folder every time
    run_dir = Path(args.outdir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # build QCNN 
    width, height = args.width, args.height
    circuit, qnn = qcnn(width * height)

    # generate dataset
    images, labels = generate_line_dataset(100, width=width, height=height, min_l=args.min_l, max_l=args.max_l)
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
    classifier = NeuralNetworkClassifier(
        qnn,
        optimizer=COBYLA(maxiter=100),
        callback=callback_graph,
        warm_start=True,
    )

    x = np.asarray(train_images)
    y = np.asarray(train_labels)
    plt.rcParams["figure.figsize"] = (12, 6)
    classifier.optimizer = COBYLA(maxiter=args.maxiter)
    classifier.fit(x, y)

    acc = np.round(100 * classifier.score(x, y), 2)
    log_line(f"Accuracy from the train data : {acc}%")
    
    # Save classifier in the run folder
    model_out = run_dir / "classifier.model"
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