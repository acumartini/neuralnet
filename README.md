# neuralnet.py

A vectorized implementation of an artificial neural network learning algorithm written in python using numpy.  The final implementation of this project will support a variable number of hidden layers (of variable size), multiclass classification, advanced optimized methods using scipy (BFGS, CG), code optimizations using LLVM interfaces, and options for unsupervised training of DBN's.

## Current Work Objectives
<ol>
<li>Advanced optimization techniques using Scipy implementations of L-BFGS-b and CG.
	<ul>
		<li>COMPLETE: L-BFGS-b outperforms CG in most cases.  Both methods work best using a large minibatch size.
		It is important to note that speeds are only slightly faster than standard gradient descent using minibatch with
		the advantage that the learning rate does not need to be set manually. <br>
		Both methods require very low tolerance setting to avoid early iteration stopping with deep networks.  Overall,
		standard gradient descent methods tend to have slightly more variation in prediction results with deeper networks. Note that it is very difficult to learn anything interested using networks with more than two hidden layers for any gradient descent method due to "the fading gradient" problem.</li>
	</ul>
</li>
<li>Break main() and minimize() methods into separate files to isolate NN class functionality.
  <ul>
	  <li>COMPLETE: main() will become a generic process() method that is called from the command line with flexible argument passing via getopt. <br>UPDATE: Decided to work toward sklearn style NNC class. All model architecture building functionality has been successfully pushed into the NeuralNetworkClassifier class. </li>
	  <li>COMPLETE: minimize() will take arguments that specify a method of optimization</li>
  </ul>
</li>
<li>Explore the LLVM interface through numba to increase speed.
	<ul><li>ON HOLD: An in depth look into performance improvements for vectorized implementation of numpy code
	using numba show that while small speed increases may be possible, large increases are only possible if you
	are able to avoid bottleneck array splitting procedures.  This would require a large code rewrite that may not
	not be very rewarding as most of the computation time is spent in np.dot operations that are already highly
	optimized.  LLVM methods and C extension (such as Cython) really shine when we require code with lots of for
	loops that do use in-place mutation of data structures and do not rely on array splicing.</li></ul>
</li>
<li>Apply MapReduce parallelization by splitting summations over multiple cores (Python Pools).
	<ul><li>HIGH PRIOIRTY: This is an important feature to add in terms of scalability.  There may will be measurable
	gains achieved if the implementation is written efficiently. <br>UPDATE: My implementation of parallel summation for
	logistic regression in Clojure indicates that this may only be beneficial for very large data structures where each
	component of the summation requires significant processing.  In other words, the overhead of map-reduce is
	a significant consideration.</li></ul>
</li>
<li>Add pre-training of parameters for DBN's:
  <ul>
    <li>HIGH PRIORITY: Implement or find reliable implementations of auto-encoders and RBM (sklearn has an implementation of RBM).  This item represents the best option for successfully training networks with more than two hidden layers.
    <br>UPDATE: The RBM implementation of deep learning is being tested in my dbn repository.  The RBM pre-training
    component will be extracted as a preprocessing step.</li>
  </ul>
</li>
