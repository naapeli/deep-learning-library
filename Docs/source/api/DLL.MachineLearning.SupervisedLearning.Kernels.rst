.. _kernel_section_label:

Kernels
=====================

Available kernels
-----------------------

.. autoclass:: DLL.MachineLearning.SupervisedLearning.Kernels.RBF
   :members: __call__, parameters

.. autoclass:: DLL.MachineLearning.SupervisedLearning.Kernels.Linear
   :members: __call__, parameters

.. autoclass:: DLL.MachineLearning.SupervisedLearning.Kernels.Periodic
   :members: __call__, parameters

.. autoclass:: DLL.MachineLearning.SupervisedLearning.Kernels.WhiteGaussian
   :members: __call__, parameters

.. autoclass:: DLL.MachineLearning.SupervisedLearning.Kernels.RationalQuadratic
   :members: __call__, parameters

Combining kernels
-------------------------
Kernels can be combined in many ways. These ways are showcased in the example below:

    .. code-block:: python

        from DLL.MachineLearning.SupervisedLearning.Kernels import Linear, RBF

        linear_kernel = Linear()
        rbf_kernel = RBF()

        d = 2

        # new kernels include:
        sum_kernel = linear_kernel + rbf_kernel  # sums
        product_kernel = linear_kernel * rbf_kernel  # products
        exponent_kernel = linear_kernel ** d  # exponents
