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

def main():
    # CLI argument parser
    parser = argparse.ArgumentParser(description="QCNN training (matches notebook).")
    parser.add_argument("--width", type=int, required=True, help="image width")
    parser.add_argument("--height", type=int, required=True, help="image height")
    parser.add_argument("--maxiter", type=int, default=200, help="COBYLA max iterations")
    parser.add_argument("--log", type=str, default="train.log", help="log file path")
    args = parser.parse_args()

    # build QCNN 
    width, height = args.width, args.height
    circuit, qnn = qcnn(width * height)

    # generate dataset
    images, labels = generate_line_dataset(100, width=width, height=height, min_l=2, max_l=2)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=246
    )

    # output logging
    log_path = Path(args.log)
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
    try:
        classifier.save("classifier.model")
        log_line("Saved classifier to classifier.model")
    finally:
        log_f.close()

if __name__ == "__main__":
    sys.exit(main())