""" Assignment 1 Experiment Runner

This script will run experiments for the following algorithms:
	1. Decision Tree
	2. ADABoost
	3. SVM
	4. Neural Networks
	5. KNN
"""

import argparse
import warnings
from experiments import run_experiments

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--all", help = "Run all experiments", action='store_true')
    g.add_argument("-e", help = "Run specific experiment: ada, dt, knn, nn, svm")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if args.all:
        print("Running all experiments")
        run_experiments(all=True)
    else: 
        if args.e.lower() not in ['ada', 'dt', 'knn', 'nn', 'svm', 'all']:
            raise ValueError("Invalid experiment, please select from following: ada, dt, knn, nn, svm")
        else:
           run_experiments(experiment=args.e)

if __name__ == "__main__":
    main()
