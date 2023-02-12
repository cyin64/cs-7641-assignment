# CS-7641 Supervised Learning (cyin64)

URL for code
https://github.com/cyin64/cs-7641-assignment

This project implements the following 5 different learning algorithms on two different datasets:
- Decision Tree Classifier
- Neural Networks
- k-nearest neighbors
- Boosting (ADABoost)
- SVM with a rbf kernel function
- SVM with an sigmoid kernel function (does not analyze in the report)
 

## Getting Started

### Installing dependencies and running experiments

Use python 3.11
install the following packages:
* Pandas
* Numpy
* Matplotlib
* sklearn
* yellowbrick


#### Run all algorithms at once:

```
python main.py --all
```

You can also individually run the experiments as shown below

```
python main.py -e dt
python main.py -e nn
python main.py -e knn
python main.py -e ada
python main.py -e svm
```

##### Authors

* **Changyong Yin** 