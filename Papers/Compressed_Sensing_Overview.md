## Comprehensive Paper Recommendations for Compressed Sensing

Here are three key papers organized by your research interests:

### 1. **Foundational Establishing Paper** (Problem Formulation & Introduction)

**Title:** Robust Uncertainty Principles: Exact Signal Reconstruction from Highly Incomplete Frequency Information

**Authors:** Emmanuel Candès, Justin Romberg, Terence Tao

**Published:** IEEE Transactions on Information Theory, Vol. 52, No. 2, pp. 489-509 (2006)

**arXiv:** math/0409186

**Why This Paper:**
This is one of the seminal papers that established compressed sensing as a field. It provides:

- Clear problem formulation of signal reconstruction from incomplete measurements
- Mathematical foundations using uncertainty principles
- Proof that sparse signals can be exactly recovered from far fewer measurements than traditionally required by the Nyquist-Shannon theorem
- Introduction of the $$\ell_1$$ minimization approach (basis pursuit)
- Foundational theory including the Restricted Isometry Property (RIP)

This paper is essential for understanding the core compressed sensing problem and the sparsity-based recovery guarantees.

---

### 2. **Optimal Control & Variational Methods Approach**

**Title:** Information-Theoretically Optimal Compressed Sensing via Spatial Coupling and Message Passing

**Authors:** David L. Donoho, Adel Javanmard, Andrea Montanari

**Published:** IEEE Transactions on Information Theory (2011)

**arXiv:** 1112.0708

**Why This Paper:**
This paper connects to your optimal control interests by:

- Using **approximate message passing (AMP)** algorithms analyzed through state evolution methods
- Studying **band-diagonal sensing matrices** inspired by spatial coupling from coding theory
- Deriving fundamental limits using **information-theoretic methods** (Rényi information dimension)
- Providing variational characterization of optimal reconstruction thresholds
- Bridging between optimization theory and information theory perspectives

The state evolution analysis has connections to variational calculus and can be viewed through an optimal control lens where the algorithm dynamics are optimized for fastest convergence.

**Alternative for Variational Calculus Focus:**

**Title:** Compressed Modes for Variational Problems in Mathematical Physics

**Authors:** V. Ozoliņš et al.

**Published:** PNAS (2013)

**Why:** Directly applies sparsity-promoting techniques to constrained **variational problems** in physics, showing how to obtain sparse localized solutions by solving optimization problems with sparsity regularization—directly relevant to Liberzon's calculus of variations framework.

---

### 3. **Machine Learning & Deep Learning Approach**

**Title:** Compressed Sensing using Generative Models

**Authors:** Ashish Bora, Ajil Jalal, Eric Price, Alexandros G. Dimakis

**Published:** International Conference on Machine Learning (ICML, 2017)

**arXiv:** 1703.03208

**Why This Paper:**
This represents the modern ML paradigm for compressed sensing by:

- Replacing traditional sparsity assumptions with **learned generative models** (VAEs and GANs)
- Showing that $$O(k \log L)$$ random Gaussian measurements suffice for recovery through a generative model
- Achieving **5-10x fewer measurements** than traditional sparse methods (LASSO) for same accuracy
- Using gradient descent on a nonconvex objective without traditional convex relaxation

**Complementary Recent ML Paper:**

**Title:** Learning Fast Approximations of Sparse Coding (LISTA)

**Authors:** Karol Gregor, Yann LeCun

**Published:** International Conference on Machine Learning (ICML, 2010)

**Why:** Introduces **Learned ISTA (LISTA)**—unrolling iterative shrinkage-thresholding algorithms into neural networks:

- Bridges classical optimization and deep learning
- Learns optimal step sizes and thresholding parameters
- Achieves 20x speedup over FISTA for approximate solutions
- Foundation for many modern unrolled network approaches

**Recent Deep Unrolling Extension:**

**Title:** Self-supervised Deep Unrolled Reconstruction Using Differentiable Projected Gradient Descent

**Authors:** P. Huang et al.

**Published:** IEEE Transactions on Medical Imaging (2024)

**Why:** Shows the state-of-the-art in combining:

- Unrolled ADMM algorithms converted to learnable networks
- Self-supervised learning without paired training data
- Integration of physical imaging models with deep networks

---

### Quick Comparison Table

| Aspect                     | Foundational                                 | Optimal Control                                  | Machine Learning                                 |
| -------------------------- | -------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| **Main Contribution**      | Proves sparse recovery from few measurements | Optimal recovery with message passing            | Generative models replace sparsity               |
| **Mathematical Framework** | Convex optimization, RIP theory              | Variational analysis, information theory         | Gradient-based learning on neural networks       |
| **Recovery Guarantees**    | Exact recovery from $$\ell_1$$ minimization  | Information-theoretic limits via state evolution | Empirical success without formal guarantees      |
| **Computational Cost**     | Polynomial (linear programming)              | Polynomial (iterative message passing)           | Fast inference after training                    |
| **Practical Advantage**    | Universal; works for any sparse signal       | Provably optimal thresholds                      | Much fewer measurements needed; learns from data |

---

### How to Access These Papers

- **arXiv papers:** Direct link via arxiv.org (free, full preprints)
- **IEEE papers:** Access through your institution or IEEE Xplore
- **ICML/conference papers:** Often freely available at conference proceedings

These three papers will give you a complete journey from the classical compressed sensing problem through optimization-theoretic approaches to modern deep learning methods—perfect for your coursework applying Liberzon's optimal control theory to signal processing.

[1](https://en.wikipedia.org/wiki/Compressed_sensing)
[2](https://assets.cambridge.org/97811070/05587/excerpt/9781107005587_excerpt.pdf)
[3](https://www.uibk.ac.at/mathematik/personal/schnass/papers/17_krkrsa_tvcs_arxiv.pdf)
[4](https://onlinelibrary.wiley.com/doi/10.1155/2016/7616393)
[5](https://www.sciencedirect.com/topics/engineering/compressed-sensing)
[6](https://www.pnas.org/doi/10.1073/pnas.1318679110)
[7](https://dustingmixon.wordpress.com/2013/12/24/matheon-workshop-2013-compressed-sensing-and-its-applications/)
[8](https://www.math.hkbu.edu.hk/~ttang/UsefulCollections/compressed-sensing1.pdf)
[9](http://proceedings.mlr.press/v139/jalal21a/jalal21a.pdf)
[10](https://arxiv.org/abs/2306.04647)
[11](https://arxiv.org/abs/1206.0663)
[12](https://ijisae.org/index.php/IJISAE/article/view/4210)
[13](https://www.sciencedirect.com/science/article/abs/pii/S0950705123005555)
[14](https://www.measurement.sk/2019/msr-2019-0006.pdf)
[15](https://pmc.ncbi.nlm.nih.gov/articles/PMC7664163/)
[16](https://pmc.ncbi.nlm.nih.gov/articles/PMC8949959/)
[17](http://papers.neurips.cc/paper/2754-recovery-of-jointly-sparse-signals-from-few-random-projections.pdf)
[18](http://proceedings.mlr.press/v119/heckel20a/heckel20a.pdf)
[19](https://shaswot.com/pdfs/mcsoc-2023-paper.pdf)
[20](https://www.sciencedirect.com/science/article/pii/S0888327024001961)
[21](https://heungno.net/wp-content/uploads/2012/11/2013-01-15-Hyeongho-journal_club.pdf)
[22](https://pmc.ncbi.nlm.nih.gov/articles/PMC11056277/)
[23](http://pwp.gatech.edu/ece-jrom/wp-content/uploads/sites/436/2011/04/donoho06co.pdf)
[24](https://arxiv.org/abs/1112.0708)
[25](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09565.pdf)
[26](http://math.ucdavis.edu/~strohmer/courses/180BigData/180lecture_cs.pdf)
[27](https://web.stanford.edu/class/archive/cs/cs265/cs265.1212/Lectures/Lecture9/l9.pdf)
[28](https://math.gsu.edu/xye/papers/chen2020variational.pdf)
[29](https://ieeexplore.ieee.org/document/1580791)
[30](https://www.slideshare.net/slideshow/compressend-sensing-using-generative-model-122976230/122976230)
[31](https://pmc.ncbi.nlm.nih.gov/articles/PMC9330715/)
[32](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4666282)
[33](http://proceedings.mlr.press/v70/bora17a/bora17a.pdf)
[34](https://discovery.ucl.ac.uk/10150849/1/ICASSP-Robust.pdf)
[35](https://digitalcommons.mtu.edu/etdr/1864/)
[36](https://arxiv.org/abs/1703.03208)
[37](https://chaspari.engr.tamu.edu/wp-content/uploads/sites/147/2018/01/2_1-1.pdf)
[38](https://arxiv.org/pdf/2010.04112.pdf)
[39](http://www.gatsby.ucl.ac.uk/~balaji/udl-camera-ready/UDL-17.pdf)
[40](https://arxiv.org/abs/math/0409186)
[41](https://www.sciencedirect.com/science/article/pii/S0165168405002215)
[42](https://icml.cc/Conferences/2010/papers/449.pdf)
[43](https://onlinelibrary.wiley.com/doi/10.1002/cpa.20124)
[44](https://linnykos.github.io/papers/cs.pdf)
[45](http://staff.ustc.edu.cn/~lszhuang/Doc/2021-PRCV-MSN.pdf)
[46](https://www.scirp.org/reference/referencespapers)
[47](https://www.semanticscholar.org/paper/Learning-Fast-Approximations-of-Sparse-Coding-Gregor-LeCun/e8f811399746c059bf4d4c3d43334045e0222209)
