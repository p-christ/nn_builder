
+-----------------------------------------------------------+----------------------------------------------------------+
| **Building a simple NN with nn_builder**                  | **Building a simple NN without nn_builder**              |
+-----------------------------------------------------------+----------------------------------------------------------+
| .. code:: python                                          | .. code:: python                                         |
|  # With nn_builder                                        |  # Without nn_builder                                    |
|  from nn_builder.pytorch.NN import NN                     |  import torch.nn as nn                                   |
|  model = NN(input_dim=25,                                 |  class NN(nn.Module):                                    |
|             layers=[150, 100, 50, 50, 50, 50, 5],         |    def __init__(self):                                   |
|             output_activation="softmax", dropout=0.5,     |      nn.Module.__init__(self)                            |
|             initialiser="xavier", batch_norm=True)        |      self.fc1 = nn.Linear(25, 150)                       |
|                                                           |      self.fc2 = nn.Linear(150, 100)                      |
|                                                           |      self.fc3 = nn.Linear(100, 50)                       |
|                                                           |      self.fc4 = nn.Linear(50, 50)                        |
|                                                           |      self.fc5 = nn.Linear(50, 50)                        |
|                                                           |      self.fc6 = nn.Linear(50, 50)                        |
|                                                           |      self.fc7 = nn.Linear(50, 5)                         |
|                                                           |                                                          |
|                                                           |      self.bn1 = nn.BatchNorm1d(150)                      |
|                                                           |      self.bn2 = nn.BatchNorm1d(100)                      |
|                                                           |      self.bn3 = nn.BatchNorm1d(50)                       |
|                                                           |      self.bn4 = nn.BatchNorm1d(50)                       |
|                                                           |      self.bn5 = nn.BatchNorm1d(50)                       |
|                                                           |      self.bn6 = nn.BatchNorm1d(50)                       |
|                                                           |                                                          |
|                                                           |      self.dropout = nn.Dropout(p=0.5)                    |
|                                                           |      for params in [self.fc1, self.fc2, self.fc3,        |
|                                                           |                     self.fc4, self.fc5, self.fc6,        |
|                                                           |                     self.fc7]:                           |
|                                                           |        nn.init.xavier_uniform_(params.weight)            |
|                                                           |    def forward(self, x):                                 |
|                                                           |      x = self.dropout(self.bn1(nn.ReLU()(self.fc1(x))))  |
|                                                           |      x = self.dropout(self.bn2(nn.ReLU()(self.fc2(x))))  |
|                                                           |      x = self.dropout(self.bn3(nn.ReLU()(self.fc3(x))))  |
|                                                           |      x = self.dropout(self.bn4(nn.ReLU()(self.fc4(x))))  |
|                                                           |      x = self.dropout(self.bn5(nn.ReLU()(self.fc5(x))))  |
|                                                           |      x = self.dropout(self.bn6(nn.ReLU()(self.fc6(x))))  |
|                                                           |      x = self.fc7(x)                                     |
|                                                           |      x = nn.Softmax(dim=1)(x)                            |
|                                                           |      return x                                            |
|                                                           |                                                          |
|                                                           |  model = NN()                                            |
+-----------------------------------------------------------+----------------------------------------------------------+

















































+------------------------------------------------+--------------------------------------------+
| **Script to train an SVM on the iris dataset** | **The same script as a Sacred experiment** |
+------------------------------------------------+--------------------------------------------+
| .. code:: python                               | .. code:: python                           |
|  # With nn_builder                                              |                                            |
|  from nn_builder.pytorch.NN import NN          |   from numpy.random import permutation     |
|                                                |   from sklearn import svm, datasets        |
|                                                |   from sacred import Experiment            |
|                                                |   ex = Experiment('iris_rbf_svm')          |
|                                                |                                            |
|                                                |   @ex.config                               |
|                                                |   def cfg():                               |
|  C = 1.0                                       |     C = 1.0                                |
|  gamma = 0.7                                   |     gamma = 0.7                            |
|                                                |                                            |
|                                                |   @ex.automain                             |
|                                                |   def run(C, gamma):                       |
|  iris = datasets.load_iris()                   |     iris = datasets.load_iris()            |
|  perm = permutation(iris.target.size)          |     per = permutation(iris.target.size)    |
|  iris.data = iris.data[perm]                   |     iris.data = iris.data[per]             |
|  iris.target = iris.target[perm]               |     iris.target = iris.target[per]         |
|  clf = svm.SVC(C, 'rbf', gamma=gamma)          |     clf = svm.SVC(C, 'rbf', gamma=gamma)   |
|  clf.fit(iris.data[:90],                       |     clf.fit(iris.data[:90],                |
|          iris.target[:90])                     |             iris.target[:90])              |
|  print(clf.score(iris.data[90:],               |     return clf.score(iris.data[90:],       |
|                  iris.target[90:]))            |                      iris.target[90:])     |
+------------------------------------------------+--------------------------------------------+


# With nn_builder


model = NN(input_dim=25, layers=[150, 100, 50, 50, 50, 50, 5],
           output_activation="softmax", dropout=0.5, initialiser="xavier",
           batch_norm=True)

