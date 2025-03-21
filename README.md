## Probability distributions and calculations for Hake's ratio statistics in measuring effect size
*a data storage for a research paper with supplementary materials - software, notebooks*

by **Jozef Hanč, Martina Hančová, Dominik Borovský**  
<jozef.hanc@upjs.sk>

### Abstract of the paper

Ratio statistics and distributions play a crucial role in various fields, including linear regression, metrology, nuclear physics, operations research, econometrics, biostatistics, genetics, and engineering. In this work, we examine the statistical properties and probability calculations of the Hake normalized gain as a measure of effect size and educational effectiveness in physics education. Leveraging existing knowledge about the Hake ratio as a ratio of normal variables and utilizing open data science tools, we developed two novel computational approaches for computing ratio distributions.

Our pilot numerical study demonstrates the speed, accuracy, and reliability of calculating ratio distributions through (1) DE quadrature with/without barycentric interpolation, a very quick and efficient quadrature method, and (2) a 2D vectorized numerical inversion of characteristic functions, which offers broader applicability by not requiring knowledge of PDFs or the independence of ratio constituents. 

These numerical explorations not only deepen the understanding of the Hake ratio's distribution but also showcase the efficiency, precision, and versatility of our proposed methods, making them highly suitable for fast data analysis based on exact probability ratio distributions. This capability has potential applications in multidimensional statistics and uncertainty analysis in metrology, where precise and reliable data handling is essential.

## Research paper 

The paper was presented as a PROBASTAT 2024 conference paper and was submitted to [Statistical Papers](https://link.springer.com/journal/362). The modified (improved) pre-print is available at <[https://arxiv.org/abs/2411.12938](https://arxiv.org/abs/2411.12938)>.

## Software [![render in nbviewer](figures/nbviewer_badge.svg)](https://nbviewer.org/github/JupyterPER/HakeRatio/tree/main/) 

The notebooks folders contain our open codes, Jupyter notebooks from the entire numerical study, which are detailed records of our computing 
with explaining narratives and illustrating explored concepts and methods. 

Notebooks can be studied and **viewed** statically in [Jupyter nbviewer](https://nbviewer.org/github/JupyterPER/HakeRatio/tree/main/) [![render in nbviewer](figures/nbviewer_badge.svg)](https://nbviewer.org/github/JupyterPER/HakeRatio/tree/main/) with full visualisation. If needed, they can also be viewed directly on GitHub as raw code. 

For interactive **executing** Jupyter notebooks as live documents without any need to install or compile the software, use [CoCalc](https://cocalc.com/) to provide interactive computing with our Jupyter notebooks.
 
All source code is distributed under [the MIT license](https://choosealicense.com/licenses/mit/).

## Overview of Jupyter Notebook Contents

### Analytical Exploration and Real Data Applications (6 Notebooks)

- [Jupyter/1.1.HypergeometricFunctions.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/1.1.HypergeometricFunctions.ipynb)
- [Jupyter/2.1.Sage-HakeRatio-plotsPdf.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/2.1.Sage-HakeRatio-PlotsPdf.ipynb)
- [Jupyter/2.1.Sage-HakeRatio-HinkleyMarsaglia.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/2.1.Sage-HakeRatio-HinkleyMarsaglia.ipynb)
- [Jupyter/2.2.Sage-HakeRatio-AnalyticMarsagliaPhamGia.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/2.2.Sage-HakeRatio-AnalyticMarsagliaPhamGia.ipynb)
- [Jupyter/2.3.Sage-HakeRatio-RealData.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/2.3.Sage-HakeRatio-RealData.ipynb)
- [Jupyter/3.1.AnalyticallyMellinlntegral.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.1.AnalyticallyMellinIntegral.ipynb)


The notebooks focus on interactive explorations of Hake ratio analytic forms and related hypergeometric functions. They also include a real data application and SageMath CAS verifications of presented analytic relations. 

---

### Algorithmic Explorations (8 Notebooks)

- [Jupyter/3.1.Python-HakeRatio-CharFun-Moments.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.1.Python-HakeRatio-CharFun-Moments.ipynb)
- [Jupyter/3.1.Python-Ratio-Mellin-DEChebyshev.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.1.Python-Ratio-Mellin-DEChebyshev.ipynb)
- [Jupyter/3.2.MTB-RatioCharFun-Algorithms-Builtin-Integrators.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.2.MTB-RatioCharFun-Algorithms-BuiltIn-Integrators.ipynb)
- [Jupyter/3.2.MTB-RatioCharFun-Algorithms-IntSum.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.2.MTB-RatioCharFun-Algorithms-IntSum.ipynb)
- [Jupyter/3.2.Python-RatioCharFun-Algorithms.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.2.Python-RatioCharFun-Algorithms.ipynb)
- [Jupyter/3.3.MTB-2DCharFun-Example.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.3.MTB-2DCharFun-Example%20.ipynb)
- [Jupyter/3.3.MTB-HakeRatio-Example.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.3.MTB-HakeRatio-Example%20.ipynb)
- [Jupyter/3.3.MTB-RatioCharFun-Examples.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/3.3.MTB-RatioCharFun-Examples.ipynb)

The notebooks present our explorations of proposed algorithms for ratio distributions, including the Hake ratio and other examples. Implementations span Python and MATLAB (MTB), showcasing built-in integrators, custom algorithms, and illustrative examples.

---

### Numerical Study (6 Notebooks)

- [Jupyter/4.1.NumStudy1-Sage-ExactValuesSageQP.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/4.1.NumStudy1-Sage-ExactValuesSageQP.ipynb)
- [Jupyter/4.1.NumStudy2A-MTB-BuiltCC-parallelCPU.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/4.1.NumStudy2A-MTB-BuiltCC-parallelCPU.ipynb)
- [Jupyter/4.1.NumStudy2B-MTB-BuiltCC.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/4.1.NumStudy2B-MTB-BuiltCC.ipynb)
- [Jupyter/4.1.NumStudy2C-MTB-Built_TRCC.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/4.1.NumStudy2C-MTB-Built_TRCC.ipynb)
- [Jupyter/4.1.NumStudy3-Py-DE_CC_Analytic.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/4.1.NumStudy3-Py-DE_CC_Analytic.ipynb)
- [Jupyter/4.2.NumStudy-Summary.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/4.2.NumStudy-Py-Summary.ipynb)

The notebooks present a numerical study of proposed built-in and custom algorithms for Hake ratio distribution using MATLAB and Python. The Hake ratio also provides closed analytical forms of distribution functions to assess method accuracy in the numerical study.

---

### Auxiliary Notebooks (3 notebooks)

- [Jupyter/HinkleyMarsaglia.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/HinkleyMarsaglia.ipynb)
- [Jupyter/MTB-MeasuringRunTimes.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/MTB-MeasuringRunTimes.ipynb)
- [Jupyter/Python-HakeRatio-CharFun-Moments.ipynb](https://nbviewer.org/github/JupyterPER/HakeRatio/blob/main/Jupyter/Python-HakeRatio-CharFun-Moments.ipynb)

These notebooks are auxiliary and are used in previous notebooks to streamline calculations.

## Acknowledgements

This work was supported by the Slovak Research and Development Agency under the Contract 
No. APVV-21-0369, No. APVV-21-0216 and by the Slovak Scientific Grant Agency VEGA under grant VEGA 1/0585/24.
We would also like to express our gratitude to Gabriel Semanišin from P.J. Šafárik University, 
for providing us with a one-year university license for MATLAB R2024b.
