# Adaptive-Stratified-Monte-Carlo
About Contains the code associated to the article "Adaptive stratified Monte Carlo using decision trees“, Chopin N, Wang H, Gerber M..
This is the code for the article "Adaptive stratified Monte Carlo using decision trees", by Chopin N, Wang H and Gerber M. (arxiv link: https://arxiv.org/pdf/2501.04842) 
## Citation
@misc{chopin2025adaptive,
      title={Adaptive stratified {M}onte {C}arlo using decision trees}, 
      author={Nicolas Chopin and Hejin Wang and Mathieu Gerber},
      year={2025},
      eprint={2501.04842},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
      url={https://arxiv.org/abs/2501.04842}, 
}
## Abstract
It has been known for a long time that stratification is one possible
  strategy to obtain higher convergence rates for the Monte Carlo estimation of
  integrals  over the hyper-cube $[0, 1]^s$ of dimension $s$. However,
  stratified estimators such as Haber's are not practical as $s$ grows, as they
  require $\OO(k^s)$ evaluations for some $k\geq 2$. We propose an adaptive
  stratification strategy, where  the strata are derived from a decision tree
  applied to a preliminary sample. We show that this strategy leads to higher
  convergence rates, that is the corresponding estimators converge at rate
  $\OO(N^{-1/2-r})$ for some $r>0$ for certain classes of functions.
  Empirically, we show through numerical experiments that the method may
  improve on standard Monte Carlo even when $s$ is large.
