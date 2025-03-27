.. Deep learning library documentation master file, created by
   sphinx-quickstart on Tue Oct 22 12:57:58 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deep learning library documentation
===================================

DLL is a deep learning library inspired by TensorFlow, PyTorch, and scikit-learn. It encompasses 
a wide range of deep learning and machine learning methods and includes numerous examples and tests 
to demonstrate their usage.

This library is intended as an educational project. While it offers a variety of functionalities, 
its performance and efficiency may not match those of the aforementioned libraries. Therefore, 
for production-level applications, it is recommended to use TensorFlow, PyTorch, or scikit-learn. 
However, DLL aims to provide greater clarity and ease of understanding compared to other libraries.

Library Structure
-----------------

The library is divided into three main packages: **Data**, **DeepLearning**, and **MachineLearning**.  
Additionally, there is a fourth package for internal exceptions. Below is a brief overview of these packages:

- **Data:** Contains utilities for data preprocessing, transformation, data loading, splitting, and assessing performance.
- **DeepLearning:** Implements various deep learning architectures, which can be combined in various ways.
- **MachineLearning:** Provides implementations of traditional machine learning algorithms. The package is divided into supervised and unsupervised learning.
- **Exceptions:** Defines internal error handling mechanisms for the library.

For detailed documentation or some example scripts, refer to the sections below.

Documentation
----------------

.. toctree::
   :maxdepth: 2

   api/DLL

Examples
----------------

.. toctree::
   :maxdepth: 1

   auto_examples/first
