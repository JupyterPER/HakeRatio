## Probability distributions and calculations for Hake's ratio statistics in measuring effect size
*a data storage for a research paper with supplementary materials - software, notebooks*

by **Jozef Hanč, Martina Hančová, Dominik Borovský**  
<jozef.hanc@upjs.sk>

### Abstract of the paper

Ratio statistics and distributions are essential in various fields, including linear regression, metrology, nuclear physics, operations research, econometrics, biostatistics, genetics, and engineering. They are also valuable in educational assessment, particularly in evaluating subjective, repeated measurements to indicate the magnitude of an intervention effect. 

This study examines the statistical properties and probability calculations of the Hake normalized gain, a key measure of effect size and educational effectiveness in physics education. Using open data science tools, we developed a computational approach involving the Mellin integral transform and its numerical integration through double exponential (DE) quadrature. Our numerical study demonstrates that DE quadrature, though still less commonly applied in statistics, provides high efficiency and accuracy for exact probability calculations. 

Recognizing that the Hake statistic is a ratio of normal variables, we compared our results with traditional analytic formulas for its distribution and other numerical methods. The findings not only enhance understanding of the Hake ratio's distribution but also show that our proposed approach, with its speed and precision, is highly suitable for fast data analysis based on exact probability distributions of quotients of random variables. This capability has potential applications in fields such as multidimensional statistics and uncertainty analysis in metrology, where precise data handling is essential.

## Research paper 

The paper was presented as a PROBASTAT 2024 conference paper and was submitted to [Statistical Papers](https://link.springer.com/journal/362). The pre-print will be available at <https://arxiv.org/>.

## Software [![render in nbviewer](figures/nbviewer_badge.svg)](https://nbviewer.org/github/JupyterPER/HakeRatio/tree/main/) 

The notebooks folders contain our open codes, Jupyter notebooks from the entire numerical study which are detailed records of our computing 
with explaining narratives ilustrating explored concepts and methods. 

Notebooks can be studied and **viewed** statically in [Jupyter nbviewer](https://nbviewer.org/github/JupyterPER/HakeRatio/tree/main/) [![render in nbviewer](figures/nbviewer_badge.svg)](https://nbviewer.org/github/JupyterPER/HakeRatio/tree/main/) with full visualisation. If there is a need, they can be also viewed directly on Github  also as a raw code. 

For interactive **executing** Jupyter notebooks as live documents without any need to install or compile the software use [CoCalc](https://cocalc.com/) providing interactive computing with our Jupyter notebooks.
 
All source code is distributed under [the MIT license](https://choosealicense.com/licenses/mit/).

## Acknowledgements

This work was supported by the Slovak Research and Development Agency under the Contract 
No. APVV-21-0369, No. APVV-21-0216 and by the Slovak Scientific Grant Agency VEGA under grant VEGA 1/0585/24.