# neuralnet.py

A vectorized implementation of an artificial neural network learning algorithm written in python using numpy.  The final implementation of this project will support a variable number of hidden layers (of variable size), multiclass classification, advanced optimized methods using scipy (BFGS, CG), code optimizations using LLVM inferfaces, and options for unsupervised training of DBN's.

## Current Work Objectives
<ol>
<li>Advanced optimization techniques using Scipy implementations of BFGS and CG.</li>
<li>Break main() and minimize() methods into separate files to isolate NN class functionality.
  <ul>
  <li>main() will become a generic process() method that is called from the command line with flexible argument passing via getopt.</li>
  <li>minimize() will take arguments that specify a method of optimization</li>
</ul></li>
<li>Explore the LLVM interface through numba to increase speed.</li>
<li>As a last resort to speed-up: apply MapReduce by splitting summations over multiple cores (Python Pools).</li>
<li>Add pre-training of parameters for DBN's:
  <ul>
    <li>Implement or find reliable implementations of auto-encoders and RBM (sklearn has an implementation of RBM).</li>
  </ul></li>
