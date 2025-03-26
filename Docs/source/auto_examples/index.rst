:orphan:

Examples
=============

Here are some examples on how to use the models in the library. These examples are meant 
to showcase each model and method defined in the library. Some utility methods, such as 
certain metrics, are omitted, as the author believes such examples have little to no practical 
use. However, most core features and functionalities are demonstrated through these examples.

Each example is designed to illustrate the model's usage, behavior, and performance in various 
scenarios. The provided scripts include data preprocessing, model training, evaluation, and 
visualization where applicable. By following these examples, users can gain a better 
understanding of how to effectively apply the models to different machine learning tasks.



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Deep learning
=============================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script implements a model to predict a dummy dataset using MultiHeadAttention. The model  has a similar structure to modern large language models, but with way less parameters.">

.. only:: html

  .. image:: /auto_examples/DeepLearning/images/thumb/sphx_glr_Attention_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_DeepLearning_Attention.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Deep learning with Attention</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script implements a model to predict values of a simple sine function. It uses recurrent layers  to handle the sequential nature of the sine function.">

.. only:: html

  .. image:: /auto_examples/DeepLearning/images/thumb/sphx_glr_TimeSeriesAnalysis_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_DeepLearning_TimeSeriesAnalysis.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recurrent networks for time series analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script implements a model to classify the iris dataset. This model uses LSTM and  RNN layers with a Bidirectional wrapper for the predictions.">

.. only:: html

  .. image:: /auto_examples/DeepLearning/images/thumb/sphx_glr_BidirectionalClassification_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_DeepLearning_BidirectionalClassification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bidirectional recurrent layers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script implements a model to predict values on a simple quadratic surface. It also  showcases some regularisation methods like Dropout and BatchNorm.">

.. only:: html

  .. image:: /auto_examples/DeepLearning/images/thumb/sphx_glr_Regression_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_DeepLearning_Regression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regression with neural networks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script implements a model using Kolmogorov-Arnold networks. It fits to a simple  quadratic surface using only a few parameters.">

.. only:: html

  .. image:: /auto_examples/DeepLearning/images/thumb/sphx_glr_KANs_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_DeepLearning_KANs.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kolmogorov-Arnold Networks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script implements a model to classify the MNIST dataset. The model mainly consists of  convolutonal layers and pooling layers with a few dense layers at the end. As the script is  only for demonstration purposes, only 100 first datapoints are used to make the training faster.  For a full example, change the parameter n to 60000. If n is increased, more epochs may need  to be added and other hyperparameters tuned.">

.. only:: html

  .. image:: /auto_examples/DeepLearning/images/thumb/sphx_glr_ImageClassification_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_DeepLearning_ImageClassification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MNIST Image classification</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Gaussian Processes
====================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of a Gaussian Process Regressor (GPR) with a  Radial Basis Function (RBF) kernel in a multidimensional setting. The example  involves training a GPR model on 2D input data and predicting the outputs on  a test set.">

.. only:: html

  .. image:: /auto_examples/GaussianProcesses/images/thumb/sphx_glr_MultidimensionalGPR_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_GaussianProcesses_MultidimensionalGPR.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multidimensional Gaussian Process Regression (GPR)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of a custom Gaussian Process Regressor (GPR)  model with a compound kernel on generated data. The model is trained using  a combination of a linear kernel and a periodic kernel, and the training  process optimizes the kernel parameters to fit the data. The script also  compares the custom GPR model with the GPR implementation from Scikit-learn  using a different kernel combination.">

.. only:: html

  .. image:: /auto_examples/GaussianProcesses/images/thumb/sphx_glr_GaussianProcessRegressor_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_GaussianProcesses_GaussianProcessRegressor.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gaussian Process Regressor (GPR)</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Linear Models
====================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of RANSAC (Random Sample Consensus) regression to fit a robust model in the presence of outliers. A quadratic relationship is used to generate inlier data, while a separate set of points acts as outliers. The performance of standard linear regression is compared with RANSAC regression to highlight the latter&#x27;s robustness to noisy data.">

.. only:: html

  .. image:: /auto_examples/LinearModels/images/thumb/sphx_glr_RANSAC_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_LinearModels_RANSAC.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Robust Regression with RANSAC</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates polynomial regression on a 2D dataset using total least squares (TLS). It generates a grid of input points, applies polynomial feature expansion, and fits a linear regression model. Predictions are visualized in 3D, comparing model output (blue) against actual values (red).">

.. only:: html

  .. image:: /auto_examples/LinearModels/images/thumb/sphx_glr_PolynomialFeatures_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_LinearModels_PolynomialFeatures.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polynomial Surface Regression with Total Least Squares</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script trains a logistic regression model on the Iris dataset using  gradient descent. It supports both binary and multi-class classification.">

.. only:: html

  .. image:: /auto_examples/LinearModels/images/thumb/sphx_glr_LogisticRegression_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_LinearModels_LogisticRegression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Logistic Regression on the Iris Dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates Locally Weighted Regression (LWR), using a Gaussian kernel  to assign weights to training samples based on their distance from a test  point.">

.. only:: html

  .. image:: /auto_examples/LinearModels/images/thumb/sphx_glr_LocallyWeightedRegression_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_LinearModels_LocallyWeightedRegression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Locally Weighted Regression on a Sine Function</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the effect of L1 (LASSO), L2 (Ridge), and  ElasticNet regularization on regression coefficients. It generates  a 3D synthetic dataset and fits different models with varying alpha  (regularization strength), tracking the weight paths.">

.. only:: html

  .. image:: /auto_examples/LinearModels/images/thumb/sphx_glr_LinearRegularisation_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_LinearModels_LinearRegularisation.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regularization Path for Ridge, LASSO, and ElasticNet Regression</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of linear regression models and their regularized counterparts (Ridge,  LASSO, and ElasticNet) on synthetic data. The models are fitted to 1D and 2D datasets, and performance  is evaluated through residual analysis, summary statistics, and visualizations.">

.. only:: html

  .. image:: /auto_examples/LinearModels/images/thumb/sphx_glr_LinearModels_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_LinearModels_LinearModels.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Linear and Regularized Regression Models on Synthetic Data</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Metrics
====================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of ROC Curve for binary classification on synthetic data.  The dataset is generated using make_blobs from scikit-learn to create a 2D feature space with two  centers. The script then splits the dataset into training and test sets, trains a logistic regression  model, and evaluates its performance using metrics such as accuracy, ROC curve, and AUC (Area Under the  Curve).">

.. only:: html

  .. image:: /auto_examples/Metrics/images/thumb/sphx_glr_ROC_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Metrics_ROC.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Logistic Regression on Synthetic Data with ROC Curve and AUC</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Naive Bayes
====================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of different Naive Bayes classifiers (Gaussian, Bernoulli, and  Multinomial) on multiple datasets: the Iris dataset and a synthetic dataset. The classifiers are  evaluated based on their accuracy in predicting the target values.">

.. only:: html

  .. image:: /auto_examples/NaiveBayes/images/thumb/sphx_glr_NaiveBayes_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_NaiveBayes_NaiveBayes.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Naive Bayes Classifiers on Iris and Synthetic Datasets</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Neighbours
====================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of K-Nearest Neighbors (KNN) for both classification and regression  tasks using the KNNClassifier and KNNRegressor models. It also showcases model serialization  with save_model.">

.. only:: html

  .. image:: /auto_examples/Neighbours/images/thumb/sphx_glr_KNN_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Neighbours_KNN.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">K-Nearest Neighbors (KNN) Classification and Regression</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Optimizers
====================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the performance of various optimization algorithms in minimizing the  Rosenbrock function, a well-known test problem in optimization. ">

.. only:: html

  .. image:: /auto_examples/Optimisers/images/thumb/sphx_glr_Optimisers_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Optimisers_Optimisers.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of Optimization Algorithms on the Rosenbrock Function</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Reinforcement learning
==========================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script implements a Deep Q-Network (DQN) by training an agent to  balance a pole in the CartPole-v1 environment from OpenAI Gymnasium. The script also  implements a custom training loop of a DLL.DeepLearning.Model.Model to train the model.">

.. only:: html

  .. image:: /auto_examples/ReinforcementLearning/images/thumb/sphx_glr_DeepReinforcementLearning_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_ReinforcementLearning_DeepReinforcementLearning.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Deep Q-Learning Agent for CartPole-v1</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Support vector machines
==========================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script demonstrates the use of Support Vector Regression (SVR) to model and predict a synthetic 3D surface. The objective is to train the model to approximate the surface defined by the equation:">

.. only:: html

  .. image:: /auto_examples/SVM/images/thumb/sphx_glr_SupportVectorRegression_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_SVM_SupportVectorRegression.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Support Vector Regression for 3D Surface Fitting</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script compares the performance of different Support Vector Machine (SVM) solvers on a synthetic 2D classification dataset. The solvers compared include:">

.. only:: html

  .. image:: /auto_examples/SVM/images/thumb/sphx_glr_SupportVectorClassification_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_SVM_SupportVectorClassification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Support Vector Classifier Solver Comparison</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Trees and boosting machines
=============================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script samples data points from a multivariate normal distribution, adds an outlier  and tries to detect it using DLL.MachineLearning.UnsupervisedLearning.OutlierDetection.IsolationForest.">

.. only:: html

  .. image:: /auto_examples/Trees/images/thumb/sphx_glr_IsolationForest_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Trees_IsolationForest.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Outlier detection using isolation forest</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script evaluates the performance of decision tree and random forest classifiers  on the Breast Cancer dataset using both DLL (`DLL.MachineLearning.SupervisedLearning.Trees`)  and scikit-learn.">

.. only:: html

  .. image:: /auto_examples/Trees/images/thumb/sphx_glr_DecisionTree_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Trees_DecisionTree.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Decision tree and random forest classifiers</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script compares the performance of various boosting classifiers, including  Gradient Boosting, AdaBoost, XGBoost, and LightGBM. Also sklearns version of AdaBoost  is compared.">

.. only:: html

  .. image:: /auto_examples/Trees/images/thumb/sphx_glr_BoostingClassifiers_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Trees_BoostingClassifiers.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Boosting Classifier Comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script evaluates and compares various regression models, including regression trees, random forest, gradient boosting, AdaBoost, XGBoost, and LGBM, using synthetic datasets.">

.. only:: html

  .. image:: /auto_examples/Trees/images/thumb/sphx_glr_RegressionTree_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_Trees_RegressionTree.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Regression using tree based models</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Unsupervised learning
=============================



.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script evaluates and visualizes various dimensionality reduction algorithms on the iris dataset.  For each algorithm, a visualization of the latent space, which is used for comparison of the algorithms.">

.. only:: html

  .. image:: /auto_examples/UnsupervisedLearning/images/thumb/sphx_glr_DimensionalityReduction_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_UnsupervisedLearning_DimensionalityReduction.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of dimensionality reduction algorithms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script classifies synthetic data using linear and quadratic disrciminant analysis.  In the visualisations, one is clearly able to see the difference  between the algorithms  using the decision boundaries.">

.. only:: html

  .. image:: /auto_examples/UnsupervisedLearning/images/thumb/sphx_glr_DiscriminantAnalysisClassification_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_UnsupervisedLearning_DiscriminantAnalysisClassification.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Classification with discriminant analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This script evaluates and visualizes various clustering algorithms on synthetic datasets. For each algorithm,  a silhouette plot is produced, which is used for comparison of the algorithms. If one wants to experiment with  differently shaped datasets, one should run the script locally and experiment with changing the &quot;dataset&quot; parameter.">

.. only:: html

  .. image:: /auto_examples/UnsupervisedLearning/images/thumb/sphx_glr_Clustering_thumb.png
    :alt:

  :ref:`sphx_glr_auto_examples_UnsupervisedLearning_Clustering.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of clustering algorithms using silhouette scores</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:
   :includehidden:


   /auto_examples/DeepLearning/index.rst
   /auto_examples/GaussianProcesses/index.rst
   /auto_examples/LinearModels/index.rst
   /auto_examples/Metrics/index.rst
   /auto_examples/NaiveBayes/index.rst
   /auto_examples/Neighbours/index.rst
   /auto_examples/Optimisers/index.rst
   /auto_examples/ReinforcementLearning/index.rst
   /auto_examples/SVM/index.rst
   /auto_examples/Trees/index.rst
   /auto_examples/UnsupervisedLearning/index.rst


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_examples_python.zip </auto_examples/auto_examples_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_examples_jupyter.zip </auto_examples/auto_examples_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
