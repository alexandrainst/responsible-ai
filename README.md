# Responsible AI Knowledge-base

This repository is a knowledge-base of different areas of using and developing AI in a responsible way:heart:. Responsible AI includes both the field of explainable and interpretable machine learning, fairness and bias in machine learning, law regulations as well as the aspect of user experience and human centralized AI.  Hence, it is a cross-disciplinary field which includes both the field of computer science and social science. The aim is to achieve systems that are trustworthy, accountable and fair. Therefore, responsible AI should hopefully both interest researchers and practitioners, which includes both developers, system owners/buyers and users :family:.

This repo is a collection of links to **research papers, blog post, tools, tutorials, videos and books**. The references are divide into different areas as listed in the table of contents.

#### Table of contents :open_file_folder:

|        | | |
| ------------- |:-------------:| -----:|
| [Explainable AI](#explainable-ai)      | [Fairness](#fairness) | [Guidelines & principles](#guide-princip)
| [People & Tech](#people-tech)  | [Policy & Regulation](#pol-reg)      | [User Experience](#ux) |

<a name="explainable-ai"></a>

  ####  Contributions  :raising_hand:

We really welcome and appreciates :pray:contributions to make sure this knowledge-base stays relevant. So if you have a link or reference you think should be included then pleas create a pull request. You can also open an issue if you find it easier.



#### Who is behind :construction_worker:

The Responsible AI repository is maintained by the [Alexandra Institute](https://alexandra.dk/uk) which is a Danish non-profit company with a mission to create value, growth and welfare in society. The Alexandra Institute is a member of [GTS](https://gts-net.dk/), a network of independent Danish research and technology organisations.

The initial work on this repository is conducted under a performance contract allocated to the Alexandra Insitute by the [Danish Ministry of Higher Education and Science](https://ufm.dk/en?set_language=en&cl=en). The project ran in the two years in 2019 and 2020.``



# Explainable AI (XAI)
## Frameworks and Github repos
1. [InterpretML](https://interpret.ml/) - Open source Python framework that combines local and global explanation methods,
as well as, transparent models, like decision trees, rule based models, and GAMs (Generalized Additive Models), into
a common API and dashboard.
2. [AI Explainability 360](http://aix360.mybluemix.net/) - Open source Python XAI framework devloped by IBM researchers
combining different data, local and global explanation methods. Also see there [github page](https://github.com/Trusted-AI/AIX360).
3. [explainX.ai](https://github.com/explainX/explainx) - Open source Python framework that launches an
interactive dashboard for a model in a single line of code in which a model can be investigated using
different XAI methods.
4. [Alibi Explain](https://github.com/SeldonIO/alibi) - Open source Pyton XAI framework combining different methods.
Main focus on counterfactual explanations and SHAP for classification tasks on tabular data or images.
5. [SHAP](https://github.com/slundberg/shap) - THe open source Python framework for generating SHAP explanations. Focused
on tree based models, but contains the model agnostic KernelSHAP and an implementation for deep neural networks.
6. [Lucid](https://github.com/tensorflow/lucid) - Open source Python framework to explain deep convolutional
neural networks used on image data (currently only supports Tensforflow 1). Focuses on understanding the
representations the network has learned.
7. [DeepLIFT](https://github.com/kundajelab/deeplift) - Open source implementation of the DeepLIFT methods for generating
local feature attributions for deep neural networks.
8. [iNNvestigate](https://github.com/albermax/innvestigate) - Github repository collecting implementations of different
feature attribution and gradient based explanation methods for deep neural networks.
9. [Skope-rules](https://github.com/scikit-learn-contrib/skope-rules) - Open source Python framework for building rule
based models.
10. [Yellowbrick](https://www.scikit-yb.org/en/latest/) - Open source Python framework to create different visualizations
of data and ML models.
11. [Captum](https://captum.ai/) - Open source framework to explain deep learning models created with PyTorch. Includes
many known XAI algorithms for deep neural networks.
12. [What-If Tool](https://pair-code.github.io/what-if-tool/) - Open source framework from Google to probe the behaviour
of a trained model.
13. [AllenNLP Interpret](https://allennlp.org/interpret) - Python framework for explaining deep neural networks
for language processing developed by the Allen Institute for AI.
14. [Dalex](http://dalex.drwhy.ai/) - Part of the DrWhy.AI universe of packages for interpretable and responsible ML.
15. [RuleFit](https://github.com/christophM/rulefit) - Open source python implementation of an interpretable rule ensemble model.
16. [SkopeRules](https://github.com/scikit-learn-contrib/skope-rules) - Open source python package for fitting a rule based model.
17. [ELI5](https://eli5.readthedocs.io/en/latest/index.html) - Open source python package that implements LIME local explanations
    and permutation explanations.

## Reading material
1. [Ansvarlig AI](https://medium.com/ansvarlig-ai) - Cross-disciplinary medium blog about XAI,
fairness and responsible AI (in Danish)
2. [Introducing the Model Card Toolkit](https://ai.googleblog.com/2020/07/introducing-model-card-toolkit-for.html) -
Google blogpost about the Model Card Toolkit that is a framework for reporting about a ML model.
3. [Interpreting Decision Trees and Random Forests](https://engineering.pivotal.io/post/interpreting-decision-trees-and-random-forests/) -
Blog post about how to interpret and visualize tree based models.
4. [Introducing PDPbox](https://towardsdatascience.com/introducing-pdpbox-2aa820afd312) - Blog post about a python
package for generating partial dependence plots.
5. [Use SHAP loss values to debug/monitor your model](https://towardsdatascience.com/use-shap-loss-values-to-debug-monitor-your-model-83f7808af40f) -
 Blog post about how to use SHAP explanations to debug and monitoring.
6. [Be careful what you SHAP for…](https://medium.com/@pauldossantos/be-careful-what-you-shap-for-aeccabf3655c) - Blog
 post about the assumption for how and when to use SHAP explanations.
7. [Awesome Interpretable Machine Learning](https://github.com/lopusz/awesome-interpretable-machine-learning) - Collection
 of resources (articles, conferences, frameworks, software, etc.) about interpretable ML.
8. [http://heatmapping.org/](http://heatmapping.org/) - Homepage of the lab behind the LRP (layerwise propagation relevance)
 method with links to tutorials and research articles.
9. [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/) - E-book by Christoph Molnar
describing and explaining different XAI methods and ways to build intepretable models or methods to interpret them, including
examples on open available datasets.
10. [Can A.I. Be Taught to Explain Itself?](https://www.nytimes.com/2017/11/21/magazine/can-ai-be-taught-to-explain-itself.html) -
The New York Times Magazine article about the need of explainable models.
11. [Deconstructing BERT, Part 2: Visualizing the Inner Workings of Attention](https://towardsdatascience.com/deconstructing-bert-part-2-visualizing-the-inner-workings-of-attention-60a16d86b5c1) -
Blog post about how to interprete a BERT model.
12. [AI Explanations Whitepaper](https://storage.googleapis.com/cloud-ai-whitepapers/AI%20Explainability%20Whitepaper.pdf) -
Google's whitepaper about Explainable AI.
13. [Robust-and-Explainable-machine-learning](https://github.com/dongyp13/Robust-and-Explainable-Machine-Learning) -
    Collection of links and articles with respect to robust and explainable machine learning,
    containing mostly deep learning related resources.

## Videos and presentations
1. [ICML 2019 session - Robust statistics and interpretability](https://slideslive.com/38917641/robust-statistics-and-interpretability)

## Courses
1. [Kaggle - Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability) -
Kaggle course about the basics of XAI with example notebooks and exercises.

## Research articles
In this section we list research articles related to interpretable ML and explainable AI.

### Definitions of interpretability
1. A. Weller, "Transparency: Motivations and Challenges", [arXiv:1708.01870](https://arxiv.org/abs/1708.01870)
[cs.CY]
2. J. Chang et al., "[Reading Tea Leaves: How Humans Interpret Topic Models](http://papers.neurips.cc/paper/3700-reading-tea-leaves-how-humans-interpret-topic-models.pdf)",
NIPS 2009
3. Z. C. Lipton, "The Mythos of Model Interpretability", [arXiv:1606.03490](https://arxiv.org/abs/1606.03490)
[cs.LG]
4. F. Doshi-Velez and B. Kim, "Towards A Rigorous Science of Interpretable Machine Learning",
[arXiv:1702.08608](https://arxiv.org/abs/1702.08608) [stat.ML]

### Review, survey and overview papers
1. G. Vilone and L. Longo, "Explainable Artificial Intelligence: a Systematic Review",
[arXiv:2006.00093](https://arxiv.org/abs/2006.00093) [cs.AI]
2. U. Bhatt et al., "[Explainable Machine Learning in Deployment](https://dl.acm.org/doi/abs/10.1145/3351095.3375624)",
FAT*20 648-657, 2020 - Survey about how XAI is used in practice.  The key results are:
    1. XAI methods are mainly used by ML engineers / designers for debugging.
    2. Limitations of the methods are often unclear to those using it.
    3. The goal og why XAI is used in the first place is often unclear or not well defined, which could potentially lead to using the wrong method.
3. L. H. Gilpin, "[Explaining Explanations: An Overview of Interpretability of Machine Learning](https://doi.org/10.1109/DSAA.2018.00018)",
IEEE 5th DSAA 80-89, 2019
4. S. T. Mueller,
"Explanation in Human-AI Systems: A Literature Meta-Review, Synopsis of Key Ideas and Publications, and Bibliography for Explainable AI",
[arXiv:1902.01876](https://arxiv.org/abs/1902.01876) [cs.AI]
5. R. Guidotti et al., "[A Survey of Methods for Explaining Black Box Models](https://dl.acm.org/doi/abs/10.1145/3236009)",
ACM Computing Surveys, 2018 - Overview of different interpretability methods grouping them after type of method,
model they explain and type of explanation.
6. M. Du et al., "[Techniques for interpretable machine learning](https://dl.acm.org/doi/10.1145/3359786)",
Communications of the ACM, 2019
7. I. C. Covert et al., Explaining by Removing:A Unified Framework for Model Explanation,
[arXiv:2011.14878](https://arxiv.org/abs/2011.14878) [cs.LG] -
(Mathematical) framework that summarizes 25 feature influence methods.
8. A. Adadi and M. Berrada, "[Peeking Inside the Black-Box: A Survey on Explainable Artificial Intelligence (XAI)](https://doi.org/10.1109/ACCESS.2018.2870052)",
IEEE Access (6) 52138-52160, 2018
9. A. Abdul et al.,
"[Trends and Trajectories for Explainable, Accountable and Intelligible Systems: An HCI Research Agenda](https://dl.acm.org/doi/10.1145/3173574.3174156)",
CHI'18 582 1-18, 2018
10. A. Preece, "[Asking ‘Why’ in AI: Explainability of intelligent systems – perspectives and challenges](https://onlinelibrary.wiley.com/doi/abs/10.1002/isaf.1422)",
Intell Sys Acc Fin Mgmt (25) 63-72, 2018
11. Q. Zhang and S.-C. Zhu,
    "[Visual Interpretability for Deep Learning: a Survey](https://link.springer.com/article/10.1631/FITEE.1700808)",
    Technol. Electronic Eng. (19) 27–39, 2018
12. B. Mittelstadt et al.,
    "[Explaining Explanations in AI](https://dl.acm.org/doi/10.1145/3287560.3287574)",
    FAT*'19 279–288, 2019

### Evaluation of XAI
This section contains articles that describe ways to evaluate explanations and explainable models.
1. S. Mohseni et al., "A Human-Grounded Evaluation Benchmark for Local Explanations of Machine Learning",
[arXiv:1801.05075](https://arxiv.org/abs/1801.05075) [cs.HC]
2. J. Huysmans et al.,
"[An empirical evaluation of the comprehensibility of decision table, tree and rule based predictive models](https://www.sciencedirect.com/science/article/abs/pii/S0167923610002368)",
Decision Support Systems (51:1) 141-154, 2011
3. F. Poursabzi-Sangdeh et al., "Manipulating and Measuring Model Interpretability",
[arXiv:1802.07810](https://arxiv.org/abs/1802.07810) [cs.AI]
4. C. J. Cai et al.,
"[The Effects of Example-Based Explanations in a Machine Learning Interface](https://dl.acm.org/doi/abs/10.1145/3301275.3302289)",
 IUI'19 258-262, 2019
5. L. Sixt et al., "When Explanations Lie: Why Many Modified BP Attributions Fail",
[arXiv:1912.09818](https://arxiv.org/abs/1912.09818) [cs.LG]
6. Y. Zhang et al.,
"[Effect of confidence and explanation on accuracy and trust calibration in AI-assisted decision making](https://dl.acm.org/doi/abs/10.1145/3351095.3372852)",
FAT*'20 295-305, 2020 - Analyses the effect of LIME explanation and confidence score as explanation on trust and human decision performance.
7. K. Sokol and P. Flach,
"[Explainability fact sheets: a framework for systematic assessment of explainable approaches](https://dl.acm.org/doi/abs/10.1145/3351095.3372870)",
FAT*'20 56-67, 2020 - Framework (essentially a list of questions or checklist) to evaluate and document XAI methods.
Also includes question that are relevant to the context in which the XAI methods should be employed, i.e. changing the outcome of the assessment based on the context.
8. E. S. Jo and T. Gebru,
"[Lessons from archives: strategies for collecting sociocultural data in machine learning](https://dl.acm.org/doi/abs/10.1145/3351095.3372829)",
FAT*'20 306-316, 2020 - Use archives as inspiration of how to collect, curate and annotate data.

### Method to explain data
This section contains articles that explain datasets, for example by finding representative examples.
1. B. Kim et al.,
   "[Examples are not Enough, Learn to Criticize! Criticism for Interpretability](https://papers.nips.cc/paper/2016/hash/5680522b8e2bb01943234bce7bf84534-Abstract.html)",
   NIPS, 2016 - Code can we found on [github](https://github.com/BeenKim/MMD-critic).

### Explainable models
This section contains articles that describe models that are explainable or transparent by design.
1. X. Zhang et al.,
   "[Axiomatic Interpretability for Multiclass Additive Models](https://dl.acm.org/doi/abs/10.1145/3292500.3330898)",
   KDD'19 226–234, 2019
2. T. Kulesza et al.,
   "[Principles of Explanatory Debugging to Personalize Interactive Machine Learning](https://dl.acm.org/doi/10.1145/2678025.2701399)",
   IUI'15 126–137, 2015 - Framework showing how a Naive Bayes method can be trained with user interaction and
   how to generate explanations for these kinds of models.
3. M. Hind et al.,
   "[TED: Teaching AI to Explain its Decisions](https://dl.acm.org/doi/abs/10.1145/3306618.3314273)",
   AIES'19 123–129, 2019
4. Y. Lou et al.,
   "[Accurate Intelligible Models with Pairwise Interactions](https://dl.acm.org/doi/10.1145/2487575.2487579)",
   KDD'13 623–631, 2013
5. C. Chen et al., "An Interpretable Model with Globally Consistent Explanations for Credit Risk",
   [arXiv:1811.12615](https://arxiv.org/abs/1811.12615) [cs.LG]
6. C. Chen and C. Rudin,
   "[An Optimization Approach to Learning Falling Rule Lists](http://proceedings.mlr.press/v84/chen18a.html)",
   PMLR (84) 604-612, 2018
7. F. Wang and C. Rudin,  "Falling Rule Lists",
   [arXiv:1411.5899](https://arxiv.org/abs/1411.5899) [cs.AI]
8. B. Ustun and C. Rudin, "Supersparse Linear Integer Models for Optimized Medical Scoring Systems",
   [arXiv:1502.04269](https://arxiv.org/abs/1502.04269) [stat.ML]
8. E. Angelino et al.,
   "[Learning Certifiably Optimal Rule Lists for Categorical Data](https://dl.acm.org/doi/abs/10.5555/3122009.3290419)",
   JMLR (18:234) 1-78, 2018
9. H. Lakkaraju et al.,
   "[Interpretable Decision Sets: A Joint Framework for Description and Prediction](https://dl.acm.org/doi/10.1145/2939672.2939874)",
   KDD'16 1675–1684, 2016
10. K. Shu et al.,
      "[dEFEND: Explainable Fake News Detection](https://dl.acm.org/doi/10.1145/3292500.3330935)",
      KDD'19 395–405, 2019
11. J. Jung et al., "Simple Rules for Complex Decisions",
    [arXiv:1702.04690](https://arxiv.org/abs/1702.04690) [stat.AP]

### XAI methods to visualize / explain a model
This section contains articles that are describing methods to globally explain a model.
Typically, this is done by generating visualizations in one form or the other.
1. B. Ustun et al.,
   "[Actionable Recourse in Linear Classification](https://dl.acm.org/doi/10.1145/3287560.3287566)",
   FAT*'19 Pages 10–19, 2019 - Article describing a method to evaluate actionable variables,
   i.e. variables a person can impact to change the outcome af a model, of a linear
   classification model.
2. A Datta et al.,
   "[Algorithmic Transparency via Quantitative Input Influence: Theory and Experiments with Learning Systems](https://ieeexplore.ieee.org/document/7546525)",
    IEEE SP 598-617, 2016
3. P.Adler et al.,
   "[Auditing black-box models for indirect influence](https://link.springer.com/article/10.1007/s10115-017-1116-3)",
   Knowl. Inf. Syst. (54) 95–122, 2018
4. A. Lucic et al.,
   "[Why Does My Model Fail? Contrastive Local Explanations for Retail Forecasting](https://dl.acm.org/doi/abs/10.1145/3351095.3372824)",
   FAT*'20 90–98, 2020 - Presents an explanation to explain failure cases of an ML/AI model.
   The explanation is presented in form of a feasible range of feature values in which the model works and a trend
   for each feature. Code for the method is available on [github](https://github.com/a-lucic/mc-brp).
5. J. Krause et al.,
   "[Interacting with Predictions: Visual Inspection of Black-box Machine Learning Models](https://dl.acm.org/doi/10.1145/2858036.2858529)",
   CHI'16 5686–5697, 2016
6. B. Kim et al.,
   "[Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](http://proceedings.mlr.press/v80/kim18d.html)",
   ICML, PMLR (80) 2668-2677, 2018 - Code for the method can be found on [github](https://github.com/tensorflow/tcav).
7. A. Goldstein et al.,
   "[Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation](https://doi.org/10.1080/10618600.2014.907095)",
   Journal of Computational and Graphical Statistics (24:1) 44-65, 2015
8. J. Wang et al., "Shapley Flow: A Graph-based Approach to Interpreting Model Predictions",
   [arXiv:2010.14592](https://arxiv.org/abs/2010.14592) [cs.LG]

### XAI methods that explain a model through construction of mimicking models
This section contains articles that are describing methods to explain a model by constructing an inherent
transparent model that mimics the behaviour of the black-box model.
1. S. Tan et al.,  
   "[Distill-and-Compare: Auditing Black-Box Models Using Transparent Model Distillation](https://dl.acm.org/doi/abs/10.1145/3278721.3278725)",
   AIES'18 303–310, 2018
2. L. Chu et al., "Exact and Consistent Interpretation for Piecewise Linear Neural Networks: A Closed Form Solution",
   [arXiv:1802.06259](https://arxiv.org/abs/1802.06259) [cs.CV]
3. C. Yang et al., "Global Model Interpretation via Recursive Partitioning",
   [arXiv:1802.04253](https://arxiv.org/abs/1802.04253) [cs.LG]
4. H. Lakkaraju et al., "Interpretable & Explorable Approximations of Black Box Models",
   [arXiv:1707.01154](https://arxiv.org/abs/1707.01154) [cs.AI]
5. Y. Hayashi,
   "[Synergy effects between grafting and subdivision in Re-RX with J48graft for the diagnosis of thyroid disease](https://www.sciencedirect.com/science/article/abs/pii/S095070511730285X)",
   Knowledge-Based Systems (131) 170-182, 2017
6. H. F. Tan et al., "Tree Space Prototypes: Another Look at Making Tree Ensembles Interpretable",
   [arXiv:1611.07115](https://arxiv.org/abs/1611.07115) [stat.ML]

### Local XAI methods
This section contains articles that describe local explanation methods, i.e. methods that generate an explanation
for a specific outcome of a model.
1. M. T. Ribeiro et al.,
   "[Anchors: High-Precision Model-Agnostic Explanations](https://homes.cs.washington.edu/~marcotcr/aaai18.pdf)",
   AAAI Conference on Artificial Intelligence, 2018 -
   The implementation of the method can be found on [github](https://github.com/marcotcr/anchor).
2. A. Shrikumar et al.,
   "[Learning Important Features Through Propagating Activation Differences](https://dl.acm.org/doi/10.5555/3305890.3306006)",
   ICML'17 3145–3153, 2017 - DeepLIFT method for local explanations of deep neural networks.
3. S. M. Lundberg et al., "Explainable AI for Trees: From Local Explanations to Global Understanding",
   [arXiv:1905.04610](https://arxiv.org/abs/1905.04610) [stat.ML]
4. S. M. Lundberg et al.,
   "[From local explanations to global understanding with explainable AI for trees](https://www.nature.com/articles/s42256-019-0138-9)",
   Nat. Mach. Intell. (2) 56–67, 2020
5. M. T. Ribeiro et al.,
   [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://dl.acm.org/doi/10.1145/2939672.2939778),
   KDD'16 1135–1144, 2016
6. D. Slack et al., "How Much Should I Trust You? Modeling Uncertainty of Black Box Explanations",
   [arXiv:2008.05030](https://arxiv.org/abs/2008.05030) [cs.LG]
7. S. M. Lundberg and S.-I. Lee,
   "[A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)",
   NIPS, 2017
8. M. Sundararajan and A. Najmi,
   "[The Many Shapley Values for Model Explanation](http://proceedings.mlr.press/v119/sundararajan20b.html)",
   ICML (119) 9269-9278, 2020
9. I. E. Kumar et al., "Problems with Shapley-value-based explanations as feature importance measures",
   [arXiv:2002.11097](https://arxiv.org/abs/2002.11097) [cs.AI]
10. P. W. Koh and P. Liang, "Understanding Black-box Predictions via Influence Functions",
      [arXiv:1703.04730](https://arxiv.org/abs/1703.04730) [stat.ML]

### Counterfactual explanations
This section contains articles that describe methods for counterfactual explanations.
1. S. Sharma et al.,
   "[CERTIFAI: A Common Framework to Provide Explanations and Analyse the Fairness and Robustness of Black-box Models](https://dl.acm.org/doi/10.1145/3375627.3375812)",
   AIES'20 166–172, 2020
2. C. Russell, "[Efficient Search for Diverse Coherent Explanations](https://dl.acm.org/doi/10.1145/3287560.3287569)",
   FAT*'19  20–28, 2019
3. R. K. Mothilal et al.,
   "[Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations](https://dl.acm.org/doi/abs/10.1145/3351095.3372850)",
   FAT*'20 607–617, 2020 - Code for the method is available on [github](https://github.com/interpretml/DiCE).
4. S. Barocas et al.,
   "[The Hidden Assumptions Behind Counterfactual Explanations and Principal Reasons](https://dl.acm.org/doi/abs/10.1145/3351095.3372830)",
   FAT*'20  80–89, 2020 - Raises some questions with respect to the use of counterfactual examples as a form of explanation:
   * Are the changes proposed by the counterfactual example feasible (actionable) for a person to change their outcome?
   * If the changes are performed, what do they affect otherwise, i.e. they might not be favorable in other contexts?
   * Changing one factor might inherently change another factor that actually negatively affects the outcome
     (counterfactual examples can not describe complex relationships between variables)?

### XAI and user interaction
This section contains research articles that are looking at the interaction of users with explanations or
interpretable models.
1. B. Y. Lim and A. K. Dey,
   "[Assessing Demand for Intelligibility in Context-Aware Applications](https://dl.acm.org/doi/10.1145/1620545.1620576)",
   UbiComp'09 195–204, 2009
2. D. Wang et al.,
   "[Designing Theory-Driven User-Centric Explainable AI](https://dl.acm.org/doi/10.1145/3290605.3300831)",
   CHI'19 (601)  1–15, 2019
3. M. Narayanan et al.,
   "How do Humans Understand Explanations from Machine Learning Systems? An Evaluation of the Human-Interpretability of Explanation",
   [arXiv:1802.00682](https://arxiv.org/abs/1802.00682) [cs.AI]
4. U. Bhatt et al., "Machine Learning Explainability for External Stakeholders",
   [arXiv:2007.05408](https://arxiv.org/abs/2007.05408) [cs.CY]
5. V. Lai and C. Tan,
   "[On Human Predictions with Explanations and Predictions of Machine Learning Models: A Case Study on Deception Detection](https://dl.acm.org/doi/abs/10.1145/3287560.3287590)",
   FAT*'19 29–38, 2019
6. C. Molnar et al., "Pitfalls to Avoid when Interpreting Machine Learning Models",
   [arXiv:2007.04131](https://arxiv.org/abs/2007.04131) [stat.ML]
7. A. Preece et al., "Stakeholders in Explainable AI",
   [arXiv:1810.00184](https://arxiv.org/abs/1810.00184) [cs.AI]
8. M. Katell et al.,
   "[Toward Situated Interventions for Algorithmic Equity: Lessons from the Field](https://dl.acm.org/doi/abs/10.1145/3351095.3372874)",
   FAT*'20 45–55, 2020 - Presenting a framework for designing ML/AI solutions based on participatory design and co-design methods,
   which especially focuses on solutions that effect communities, i.e. models employed by municipalities. The framework is applied
   to an example case in which a surveillance tool with an automatic decision system is designed.
9. M. Eiband et al.,
   "[Bringing Transparency Design into Practice](https://dl.acm.org/doi/10.1145/3172944.3172961)",
   IUI'18 211–223, 2018

### XAI used in practice
This section contains research articles where XAI was used as part of an application or used for validation on a system
deployed in practice.
1. S. Coppers et al.,
   "[Intellingo: An Intelligible Translation Environment](https://dl.acm.org/doi/10.1145/3173574.3174098)",
   CHI'18 (524) 1–13, 2018
2. H. Tang and P. Eratuuli,
   "[Package and Classify Wireless Product Features to Their Sales Items and Categories Automatically](https://link.springer.com/chapter/10.1007/978-3-030-29726-8_20)",
   Machine Learning and Knowledge Extraction. CD-MAKE 2019. LNCS (11713), 2019

### XAI for deep neural networks
This section focuses on explainability with respect to deep neural networks (DNNs). This can be methods to explain
DNNs or methods to build DNNs that can explain themselves.
1. Y. Goyal et al.,
   "[Counterfactual Visual Explanations](http://proceedings.mlr.press/v97/goyal19a.html)",
   36th ICML, PMLR (97) 2376-2384, 2019 - Describing a method to construct a DNN for image classification that provides
   counterfactual explanations.
2. K. Simonyan et al., "Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps",
   [arXiv:1312.6034](https://arxiv.org/abs/1312.6034) [cs.CV]
3. A. Tavanaei, "Embedded Encoder-Decoder in Convolutional Net works Towards Explainable AI",
   [arXiv:2007.06712](https://arxiv.org/abs/2007.06712) [cs.CV] - DNN with a build in encoder-decoder that generates explanations.
7. S. Bach et al.,
   "[On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://doi.org/10.1371/journal.pone.0130140)",
   PLOS ONE (10:7) e0130140, 2015 - Description of the LRP method for DNNs. Code for playing around with the LRP method
   can be found on [github](https://github.com/sebastian-lapuschkin/lrp_toolbox).
4. W. Samek et al.,
   "[Evaluating the Visualization of What a Deep Neural Network Has Learned](https://ieeexplore.ieee.org/document/7552539)",
   IEEE Trans. Neural Netw. Learn. Syst. (28:11) 2660-2673, 2017
5. G. Montavon et al.,
   "[Explaining nonlinear classification decisions with deep Taylor decomposition](https://www.sciencedirect.com/science/article/pii/S0031320316303582)",
   Pattern Recognition (65) 211-222, 2017
6. G. Montavon et al.,
   "[Methods for Interpreting and Understanding Deep Neural Networks](https://www.sciencedirect.com/science/article/pii/S1051200417302385)",
   Digital Signal Processing (73) 1-15, 2018
7. S. Lapuschkin et al.,
   "[Unmasking Clever Hans predictors and assessing what machines really learn](https://www.nature.com/articles/s41467-019-08987-4)",
   Nat. Commun. 10 1096, 2019 - Using LRP the authors find "cheating" strategies of DNNs in varying tasks.
   I recommend to also check the supplementary which contains more experiments and insights.
6. M. Sundararajan et al.,
   "[Exploring Principled Visualizations for Deep NetworkAttributions](http://ceur-ws.org/Vol-2327/IUI19WS-ExSS2019-16.pdf)",
   IUI Workshops, 2019
7. R. R. Selvaraju,
   "[Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization](https://ieeexplore.ieee.org/abstract/document/8237336)",
   IEEE ICCV 618-626, 2017
8. Q. Zhang, "[Interpretable CNNs](https://ieeexplore.ieee.org/document/8579018)",
   IEEE/CVF CVPR 8827-8836, 2018
9. R. C. Fong and A. Vedaldi,
   "[Interpretable Explanations of Black Boxes by Meaningful Perturbation](https://ieeexplore.ieee.org/document/8237633)",
   IEEE ICCV 3449-3457, 2017 -
   A PyTorch implementation can be found on [github](https://github.com/ruthcfong/pytorch-explain-black-box).
10. R. Fong and A. Vedaldi,
    "[Net2Vec: Quantifying and Explaining how Concepts are Encoded by Filters in Deep Neural Networks](https://ieeexplore.ieee.org/abstract/document/8579008)",
    018 IEEE/CVF CVPR 8730-8738, 2018
11. R. Hu et al.,
    "[Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://ieeexplore.ieee.org/document/8237355)",
    IEEE ICCV 804-813, 2017
12. A. Nguyen, "Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks",
    [arXiv:1602.03616](https://arxiv.org/abs/1602.03616) [cs.CV]
13. S. O. Arik and T. Pfister, "ProtoAttend: Attention-Based Prototypical Learning",
    [arXiv:1902.06292](https://arxiv.org/abs/1902.06292) [cs.CV]
14. A. Ghorbani et al.,
    "[Towards Automatic Concept-based Explanations](https://papers.nips.cc/paper/2019/hash/77d2afcb31f6493e350fca61764efb9a-Abstract.html)",
    NeurIPS, 2019
15. M. Ancona et al., "Towards better understanding of gradient-based attribution methods for deep neural networks",
    [arXiv:1711.06104](https://arxiv.org/abs/1711.06104) [cs.LG]
16. A. Mahendran and A. Vedaldi,
    "[Understanding deep image representations by inverting them](https://ieeexplore.ieee.org/document/7299155)",
    IEEE CVPR 5188-5196, 2015
17. A. Kapishnikov et al.,
    "[XRAI: Better Attributions Through Regions](https://ieeexplore.ieee.org/document/9008576)",
    IEEE ICCV 4947-4956, 2019
18. B. Alsallakh et al., "Do Convolutional Neural Networks Learn Class Hierarchy?",
      [arXiv:1710.06501](https://arxiv.org/abs/1710.06501) [cs.CV]
19. S. Wang et al.,
      "[Bias Also Matters: Bias Attribution for Deep Neural Network Explanation](http://proceedings.mlr.press/v97/wang19p.html)",
      36th ICML, PMLR (97) 6659-6667, 2019 - Describing the effect of the bias parameter on XAI methods using the gradient.
20. N. Papernot and P. McDaniel, "Deep k-Nearest Neighbors: Towards Confident, Interpretable and Robust Deep Learning",
      [arXiv:1803.04765](https://arxiv.org/abs/1803.04765) [cs.LG] - A DNN using KNN in the representation space to ensure
      consistency in the predictions.
21. O. Li et al.,
    "Deep Learning for Case-Based Reasoning through Prototypes: A Neural Network that Explains Its Predictions",
    [arXiv:1710.04806](https://arxiv.org/abs/1710.04806) [cs.AI]
22. A. Wan et al., "NBDT: Neural-Backed Decision Trees",
    [arXiv:2004.00221](https://arxiv.org/abs/2004.00221) [cs.CV] - An approach that combines DNN with decision trees
    in cases where there is a "natural" hierarchy of classes.
    See also their [homepage](https://research.alvinwan.com/neural-backed-decision-trees/#ship).
23. K. Xu et al.,
    "[Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://proceedings.mlr.press/v37/xuc15.html)",
    PMLR (37) 2048-2057, 2015 - DNN that generates text explanation together with highlights within the image.
    Code can be found on [github](https://github.com/kelvinxu/arctic-captions).
24. C. Chen et al.,
    "[This Looks Like That: Deep Learning for Interpretable Image Recognition](https://papers.nips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html)",
    NeurIPS, 2019
25. V. Petsiuk et al., "RISE: Randomized Input Sampling for Explanation of Black-box Models",
    [arXiv:1806.07421](https://arxiv.org/abs/1806.07421) [cs.CV]
26. P. Sturmfels et al.,
    "[Visualizing the Impact of Feature Attribution Baselines](https://distill.pub/2020/attribution-baselines/)",
    Distill, 2020.    
27. D. Bau et al.,
    "[Understanding the role of individual units in a deep neural network](https://www.pnas.org/content/117/48/30071)",
    PNAS (117:48) 30071-30078, 2020 - All links and material regarding the article is summarized by the authors on their
    [website](https://dissect.csail.mit.edu).  

### XAI for natural language processing
This section contains papers in which XAI methods are used or developed for NLP tasks and models.
1. S. Jain and B. C. Wallace, "Attention is not Explanation",
   [arXiv:1902.10186](https://arxiv.org/abs/1902.10186) [cs.CL]
2. W. J. Murdoch and A. Szlam, "Automatic Rule Extraction from Long Short Term Memory Networks",
   [arXiv:1702.02540](https://arxiv.org/abs/1702.02540) [cs.CL]
3. W. J. Murdoch et al., "Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs",
   [arXiv:1801.05453](https://arxiv.org/abs/1801.05453) [cs.CL]
4. L. Arras et al., "Explaining Recurrent Neural Network Predictions in Sentiment Analysis",
   [arXiv:1706.07206](https://arxiv.org/abs/1706.07206) [cs.CL]
5. T. Guo et al.,
   "[Exploring Interpretable LSTM Neural Networks over Multi-Variable Data](http://proceedings.mlr.press/v97/guo19b.html)",
   36th ICML (97) 2494-2504, 2019
6. F. Liu and B. Avci,
   "[Incorporating Priors with Feature Attribution on Text Classification](https://www.aclweb.org/anthology/P19-1631/)",
   57th ACL (P19-1631) 6274–6283, 2019
7. A. Radford et al., "Learning to Generate Reviews and Discovering Sentiment",
   [arXiv:1704.01444](https://arxiv.org/abs/1704.01444) [cs.LG]
8. H. Strobelt et al.,
   "[LSTMVis: A Tool for Visual Analysis of Hidden State Dynamics in Recurrent Neural Networks](https://ieeexplore.ieee.org/document/8017583)",
   IEEE Trans. Vis. Comput. Graph (24:1) 667-676, 2018
9. T. Lei et al.,
   "[Rationalizing Neural Predictions](https://www.aclweb.org/anthology/D16-1011/)",
   EMNLP (D16-1011) 107–117, 2016
10. M. T. Ribeiro et al.,
    "[Semantically Equivalent Adversarial Rules for Debugging NLP Models](https://www.aclweb.org/anthology/P18-1079/)",
    56th ACL (P18-1079) 856–865, 2018
11. C. Guan et al.,
    "[Towards a Deep and Unified Understanding of Deep Neural Models in NLP](http://proceedings.mlr.press/v97/guan19a.html)",
    36th ICML (97) 2454-2463, 2019
12. J. Li et al.,
    "[Visualizing and Understanding Neural Models in NLP](https://www.aclweb.org/anthology/N16-1082/)",
    NAACL (N16-1082) 681–691, 2106
13. A. Karpathy et al., "Visualizing and Understanding Recurrent Networks",
      [arXiv:1506.02078](https://arxiv.org/abs/1506.02078) [cs.LG]
14. L. Arras et al., "What is Relevant in a Text Document?": An Interpretable Machine Learning Approach,
    [arXiv:1612.07843](https://arxiv.org/abs/1612.07843) [cs.CL]

### XAI for recommender systems
This section contains papers describing explainability with respect to recommender systems.
1. I. Nunes and D. Jannach,
   "[A systematic review and taxonomy of explanations in decision support and recommender systems](https://link.springer.com/article/10.1007/s11257-017-9195-0)",
   User Model User-Adap. Inter. (27) 393–444, 2017
2. J. L. Herlocker et al.,
   "[Explaining Collaborative Filtering Recommendations](https://dl.acm.org/doi/10.1145/358916.358995)",
   CSCW'00 241–250, 2000
3. D. Mcsherry,
   "[Explanation in Recommender Systems](https://link.springer.com/article/10.1007/s10462-005-4612-x)",
    Artif. Intell. Rev. 24 179–197, 2005

### XAI with and for reinforcement learning
This section contains papers describing explainability with respect to reinforcement learning.
1. L. She and J. Y. Chai,
   "[Interactive Learning of Grounded Verb Semantics towards Human-Robot Communication](https://www.aclweb.org/anthology/P17-1150/)",
   55th ACL (P17-1150) 1634–1644, 2017
2. Samantha Krening et al.,
   "[Learning From Explanations Using Sentiment and Advice in RL](https://ieeexplore.ieee.org/document/7742965)",
   TCDS (9:1) 44-55, 2017

### XAI in the medical domain
This section contains papers in which XAI models or methods were used on medical data.
1. S. Meyer Lauritsen et al.,
   "[Explainable artificial intelligence model to predict acute critical illness from electronic health records](https://www.nature.com/articles/s41467-020-17431-x)",
   Nat. Commun. 11 3852, 2020
2. S. M. Lundberg et al.,
   "[Explainable machine-learning predictions for the prevention of hypoxaemia during surgery](https://www.nature.com/articles/s41551-018-0304-0)"
    Nat. Biomed. Eng. (2:10) 749-760, 2018
3. Z. Che et al.,
   "[Interpretable Deep Models for ICU Outcome Prediction](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5333206/)",
   AMIA Annu. Symp. Proc. (2016) 371-380, 2017
4. R. Sayres et al.,
   "[Using a Deep Learning Algorithm and Integrated Gradients Explanation to Assist Grading for Diabetic Retinopathy](https://www.sciencedirect.com/science/article/pii/S0161642018315756)",
   Ophthalmology  (126:4), 2019s
5. J. Ma et al.,
   "[Using deep learning to model the hierarchical structure and function of a cell](https://www.nature.com/articles/nmeth.4627)",
   Nat. Methods (15) 290–298, 2018
6. R. Caruana et al.,
   "[Intelligible Models for HealthCare: Predicting Pneumonia Risk and Hospital 30-day Readmission](https://dl.acm.org/doi/10.1145/2783258.2788613)",
   KDD'15 1721–1730, 2015
7. B. Letham et al.,
   "Interpretable classifiers using rules and Bayesian analysis: Building a better stroke prediction model",
   [arXiv:1511.01644](https://arxiv.org/abs/1511.01644) [stat.AP]
8. E. Choi et al.,
   "[RETAIN: An Interpretable Predictive Model for Healthcare using Reverse Time Attention Mechanism](https://papers.nips.cc/paper/2016/hash/231141b34c82aa95e48810a9d1b33a79-Abstract.html)",
   NIPS, 2016

## Books
1. [Explainable AI: Interpreting, Explaining and Visualizing Deep Learning](https://doi.org/10.1007/978-3-030-28954-6) -
Explainability with respect to deep learning with a focus on convolutional neural networks used for image data.
The editor of the book are also behind the layerwise relvance propagation (LRP) method.
2. [Explainable and Interpretable Models in Computer Vision and Machine Learning](https://www.springer.com/gp/book/9783319981307) -
More general book about explainability in machine learning, but also with a focus on deep learning in computer vison.

<a name="fairness"></a>
# Fairness

## Frameworks and Github repos

1.  [AI Fairness 360](https://aif360.mybluemix.net/) Toolkit from IBM both in Python and R to examine, report and mitigate bias and discriminations in data and machine learning models.
2. [What-if-tool](https://pair-code.github.io/what-if-tool/ai-fairness.html) from Google's PAIR (People and AI Research)  allowed to play around with different fairness metrics.
3. [FAT Forensics](https://github.com/fat-forensics/fat-forensics) is a python toolbox for evaluating fairness, accountability and transparency of predictive systems.
4. [Fairlearn](https://github.com/fairlearn/fairlearn) is a python package for accessing and mitigating bias in machine leaning system. The repo both contain implemented algorithm and Jupyter Notebook with examples of use.
5. [LiFT](https://github.com/linkedin/LiFT) - The LinkedIn Fairness Toolkit (LiFT)
6. [Aequitas](http://aequitas.dssg.io/) - Bias and Fairness Audit Toolkit

## Reading material
1. [What is bias?](https://towardsdatascience.com/what-is-ai-bias-6606a3bcb814) - Towards data science
   blogpost about bias.
2. [Explaining Measures of Fairness](https://towardsdatascience.com/explaining-measures-of-fairness-f0e419d4e0d7), Scott Lundberg, 2020, Medium, Towards Data Science - Blogpost describing how to use XAI methods to explain features' contributions to fairness metrics.
3. [Algorithmic Solutions to Algorithmic Bias: A Technical Guide](https://towardsdatascience.com/algorithmic-solutions-to-algorithmic-bias-aef59eaf6565) - Towards data science blogpost describing different methods and techniques to avoid or correct for bias.
4. [Fairness Metrics Won’t Save You from Stereotyping](https://towardsdatascience.com/fairness-metrics-wont-save-you-from-stereotyping-27127e220cac), Valerie Carey, 2020, Medium, Towards Data Science - Blogpost pointing out that different models with different "bias" can have the same performance on fairness metrics.
5. [A Tutorial on Fairness in Machine Learning](https://towardsdatascience.com/a-tutorial-on-fairness-in-machine-learning-3ff8ba1040cb), Ziyuan Zhong, 2018, Medium, Towards Data Science
6. [Racial Bias in BERT](https://towardsdatascience.com/racial-bias-in-bert-c1c77da6b25a), Gergely D. Németh, 2020, Medium, Towards Data Science

## Videos and presentations
1. [The Trouble with Bias](https://www.youtube.com/watch?v=fMym_BKWQzk), Kate Crawford, NIPS 2017 Keynote 

## Courses
1. [Google's Fairness course](https://developers.google.com/machine-learning/crash-course/fairness/video-lecture)
2. [A Course on Fairness, Accountability and Transparency in Machine Learning](https://geomblog.github.io/fairness/)

## Research articles

### Review, survey and overview papers

1. Mitchell, Shira, et al. "[Prediction-based decisions and fairness: A catalogue of choices, assumptions, and definitions](https://arxiv.org/pdf/1811.07867.pdf)" *arXiv preprint arXiv:1811.07867* (2018).
2. Gajane, Pratik, and Mykola Pechenizkiy. "[On formalizing fairness in prediction with machine learning](https://arxiv.org/pdf/1710.03184.pdf)." *arXiv preprint arXiv:1710.03184* (2017).
3. Mehrabi, Ninareh, et al. "[A survey on bias and fairness in machine learning](https://arxiv.org/pdf/1908.09635.pdf)." *arXiv preprint arXiv:1908.09635* (2019).
4. Chouldechova, Alexandra, and Aaron Roth. "[A snapshot of the frontiers of fairness in machine learning](https://arxiv.org/pdf/1908.09635.pdf)." *Communications of the ACM* 63.5 (2020): 82-89.
5. Holstein, Kenneth, et al. "[Improving fairness in machine learning systems: What do industry practitioners need?](https://dl.acm.org/doi/pdf/10.1145/3290605.3300830?casa_token=IRwVVzPP6gsAAAAA:W3WXxYO_9MxGulXAM842EYss-eStMH7L_x9UWrWRQNFA79fDb7EiudKjp-ujCa-6Rfz7Ogm7zOrGsnk)." *Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems*. 2019.
6. Corbett-Davies, Sam, and Sharad Goel. "[The measure and mismeasure of fairness: A critical review of fair machine learning](https://arxiv.org/pdf/1808.00023.pdf)." *arXiv preprint arXiv:1808.00023* (2018).
7. Selbst, Andrew D., et al. "[Fairness and abstraction in sociotechnical systems](https://dl.acm.org/doi/pdf/10.1145/3287560.3287598?casa_token=STFTSq_Y_vMAAAAA:67Tz0ZexSMGft-FsRbifOK0P-YnJn-lRiv_-p3tW4I3FYju2H8pdfPewgnCJivHu9ztxXszBUGf3-7M)." *Proceedings of the Conference on Fairness, Accountability, and Transparency*. 2019.

### Ways to define fairness, critics and challenges with definitions

**Static fairness metrics**

1. Hardt, Moritz, Eric Price, and Nathan Srebro. "[Equality of opportunity in supervised learning](https://arxiv.org/pdf/1808.00023.pdf)." *arXiv preprint arXiv:1610.02413* (2016). -- The paper defines the fariness metric Equalized Odds and makes critics of the fariness metric Demographic Parity
2. Verma, Sahil, and Julia Rubin. "[Fairness definitions explained](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8452913&casa_token=uBqt4_J3m6wAAAAA:eB9a9ySO28rjfQeogitspJ5JaL1JpNo57i6RORflKa3lz0ulRlWEw7P_jgq56TNW3y4XEiKdRe0l&tag=1)." *2018 IEEE/ACM International Workshop on Software Fairness (FairWare)*. IEEE, 2018. -- This paper explains and demonstrates different statistical fairness metrics which requires to achieve parity for a metric between groups.
3. Berk, Richard, et al. "[Fairness in criminal justice risk assessments: The state of the art](https://journals.sagepub.com/doi/pdf/10.1177/0049124118782533?casa_token=MoCx596UMOMAAAAA%3AXFo-EGEe9eUWPJDLF3UR-0XDtiRgcgLfjIlnoKPE3nYat50rhpsWYNNTJ7bw0zNuTro2xa4ZLfjGWqg&)." *Sociological Methods & Research* (2018): 0049124118782533. -- The paper discuss trades-offs between different fairness metrics and accuracy for criminal assessment, and shows that some metrics and accuracy is incompatible.
4. Kleinberg, Jon, Sendhil Mullainathan, and Manish Raghavan. "[Inherent trade-offs in the fair determination of risk scores](https://arxiv.org/pdf/1609.05807.pdf)." *arXiv preprint arXiv:1609.05807* (2016). -- They examine three definitions of fairness metrics and shows that except in special cases the metrics and incompatible and can be achieved simultaneous.
5. Kearns, Michael, et al. "[Preventing fairness gerrymandering: Auditing and learning for subgroup fairness](http://proceedings.mlr.press/v80/kearns18a/kearns18a.pdf)." *International Conference on Machine Learning*. PMLR, 2018. -- The paper highlight that using statistical fairness metrics for ensuring parity between groups does not give any guaranty for subgroups. They formulated function to test in smaller subgroups.
6. Liu, Lydia T., et al. "[Delayed impact of fair machine learning](https://arxiv.org/pdf/1803.04383.pdf)." *arXiv preprint arXiv:1803.04383* (2018). -- Demonstrates trough one step simulation that achieving fairness metric such as Demographic Parity and Equalized Odds can leave the protected group worse off one step in the "future"
7.  D'Amour, Alexander, et al. "[Fairness is not static: deeper understanding of long term fairness via simulation studies](https://dl.acm.org/doi/pdf/10.1145/3351095.3372878?casa_token=hyV5SnA1kzsAAAAA:w-k6fkR2lmBkNfBX5BFz0AClUb-49KUe4EpHWQi8T5rqON29n_1auMLxoSoXrAriToVm3tbpw2prJwM)." *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*. 2020.

**Individual and preference fairness**

1. Dwork, Cynthia, et al. "[Fairness through awareness](https://dl.acm.org/doi/pdf/10.1145/2090236.2090255?casa_token=yvxsQJAHa5gAAAAA:KMnAWHsunCNSu03BdKhuzHwih11VMkRmIDcqVivJrXjJBINS2CB2n_qMPbXnx7OGo8hfIlxP9tpp3Xs)." *Proceedings of the 3rd innovations in theoretical computer science conference*. 2012. -- The paper formulates the ideas behind individual fairness (similar individuals should be treated similar)

2. Kim, Michael, Omer Reingold, and Guy Rothblum. "[Fairness through computationally-bounded awareness]()." *Advances in Neural Information Processing Systems* 31 (2018): 4842-4852.

3. Zafar, Muhammad Bilal, et al. "[From parity to preference-based notions of fairness in classification](https://papers.nips.cc/paper/2017/file/82161242827b703e6acf9c726942a1e4-Paper.pdf)." *Advances in Neural Information Processing Systems*. 2017. -- Defines *preference* based fairness which carries the idea that each individual should have preference for receiving the outcome from ones own group deepened classifier. This should leave room for optimizing the classifiers within each group.  

4. Kim, Michael P., et al. "[Preference-informed fairness](https://arxiv.org/pdf/1904.01793.pdf)." *arXiv preprint arXiv:1904.01793* (2019). -- Combine the ideas between individual and preference fairness.  



**Causal Reasoning  based fairness**

1. Kusner, Matt J., et al. "[Counterfactual fairness](https://papers.nips.cc/paper/2017/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)." *Advances in neural information processing systems*. 2017. -- Definition on counterfactatual fairness. The idea is that fairness is achieved if a individual will receive the same outcome both in the actual world and in the counterfactual.
2. Chiappa, Silvia. "[Path-specific counterfactual fairness](https://ojs.aaai.org//index.php/AAAI/article/view/4777)." *Proceedings of the AAAI Conference on Artificial Intelligence*. Vol. 33. 2019. -- Formulates a counterfactual fairness  that follows different path of sensitive attributes. Applicable to complex non-linear models.
3. Garg, Sahaj, et al. "[Counterfactual fairness in text classification through robustness](https://dl.acm.org/doi/abs/10.1145/3306618.3317950?casa_token=yJx29t7-lC4AAAAA:4bs5iTr2Z4piEcHIpW5m1AANTGoajFCV1SlT83BaDbpjs5RMJ1iUxIs5_EuNIj3Bv91LwF58650ZJ7A)." *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society*. 2019.  -- Counterfactual method to look at text classification e.g. for finding toxic comments where the aim is that the reference to the sensitive attribution should not affect the classification.

**Fairness trough explanations**

1. Cesaro, Juliana, and Fabio Gagliardi Cozman. "[Measuring Unfairness Through Game-Theoretic Interpretability](https://arxiv.org/pdf/1910.05591.pdf)." *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*. Springer, Cham, 2019. -- Present the idea that fairness can be assessed by looking a the "global" feature attribution on a test set for different protected group using e.g the SHAP framework  
2. Hickey, James M., Pietro G. Di Stefano, and Vlasios Vasileiou. "[Fairness by Explicability and Adversarial SHAP Learning](https://arxiv.org/pdf/2003.05330.pdf)." *arXiv preprint arXiv:2003.05330* (2020).  -- Asses fairness trough explanations (SHAP) and compare to other statistic measures and propose in-process algorithm for mitigating bias.

#### Mitigating algorithm

1. Kamiran, Faisal, and Toon Calders. "[Data preprocessing techniques for classification without discrimination](https://link.springer.com/article/10.1007/s10115-011-0463-8)." *Knowledge and Information Systems* 33.1 (2012): 1-33.
2. Zemel, Rich, et al. "[Learning fair representations](http://proceedings.mlr.press/v28/zemel13.pdf)." *International Conference on Machine Learning*. 2013.
3. Calmon, Flavio, et al. "[Optimized pre-processing for discrimination prevention](https://papers.nips.cc/paper/2017/file/9a49a25d845a483fae4be7e341368e36-Paper.pdf)." *Advances in Neural Information Processing Systems* 30 (2017): 3992-4001.
4. Feldman, Michael, et al. "[Certifying and removing disparate impact](https://dl.acm.org/doi/pdf/10.1145/2783258.2783311?casa_token=mSeGWlsvE1wAAAAA:U_v45paImwWGerBh_JQO_6eEM30-DDRUblBW83iuTX14dCoAZ7GC8URbcYewA1D5YEzVjUn9D2EHJl8)." *proceedings of the 21th ACM SIGKDD international conference on knowledge discovery and data mining*. 2015.
5. Zhang, Brian Hu, Blake Lemoine, and Margaret Mitchell. "[Mitigating unwanted biases with adversarial learning](https://dl.acm.org/doi/pdf/10.1145/3278721.3278779?casa_token=6P7Ba2BKRQsAAAAA:4VXAig2OO5-GtQU4fiYs06HnYyLftC_QLga9-cd5dknu703521R7JZjzylSLHWdsSv799wN-qgSBrG0)." *Proceedings of the 2018 AAAI/ACM Conference on AI, Ethics, and Society*. 2018.
6. Kamishima, Toshihiro, et al. "[Fairness-aware classifier with prejudice remover regularizer](https://link.springer.com/chapter/10.1007/978-3-642-33486-3_3)." *Joint European Conference on Machine Learning and Knowledge Discovery in Databases*. Springer, Berlin, Heidelberg, 2012.
7. Pleiss, Geoff, et al. "[On fairness and calibration](https://papers.nips.cc/paper/2017/file/b8b9c74ac526fffbeb2d39ab038d1cd7-Paper.pdf)." *Advances in Neural Information Processing Systems*. 2017.

### Perceived algorithmic fairness

1. Grgic-Hlaca, Nina, et al. "[Beyond Distributive Fairness in Algorithmic Decision Making: Feature Selection for Procedurally Fair Learning](https://people.mpi-sws.org/~nghlaca/papers/fair_feature_selection.pdf)." *AAAI*. Vol. 18. 2018. -- Propose to shift the focus for outcome fairness to procedurally fairness where there instead should be a focus of how the outcome is concluded instead of what it actually is. The paper makes a survey to examine peoples perception of using different input features in different settings.
2. Srivastava, Megha, Hoda Heidari, and Andreas Krause. "[Mathematical notions vs. human perception of fairness: A descriptive approach to fairness for machine learning](https://dl.acm.org/doi/pdf/10.1145/3292500.3330664?casa_token=ZimWr0aISCYAAAAA:VREKO6HnQz9ikz3thWkby1ZuS26qCCzzCPF-rWCrncUGNahAWimeUyiguF_XZX4zkWLUpya_23p0hTs)." *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining*. 2019. -- Attempt to measure peoples perception of different statistical fairness metrics trough an Amazon Turk survey.
3.  Harrison, Galen, et al. "[An empirical study on the perceived fairness of realistic, imperfect machine learning models](https://dl.acm.org/doi/pdf/10.1145/3351095.3372831?casa_token=o-9Ab5o5C4cAAAAA:0aK2JEpnySjLfkYp7uv0ErQmdSD6saQknXDxK381H0vsVRKPBDFGqQuunDWNI6fkSQa1iUOkK-vT_II)." *Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency*. 2020. -- Examine peoples perception of trade-offs between models which satisfies different statistical fairness measure or accuracy trough an Amazon Turk Survey
4. Saha, Debjani, et al. "[Measuring non-expert comprehension of machine learning fairness metrics](http://proceedings.mlr.press/v119/saha20c/saha20c.pdf)." *International Conference on Machine Learning*. PMLR, 2020. -- Examine lay-peoples comprehension of statistical fairness metrics and show comprehension can be measured trough a multiple-choice survey, and find that comprehension is correlated with education, and the fact that higher comprehension is correlated with a more negative perception of the metrics.  
5. Dodge, Jonathan, et al. "[Explaining models: an empirical study of how explanations impact fairness judgment](https://dl.acm.org/doi/pdf/10.1145/3301275.3302310?casa_token=JRLzBvoey5sAAAAA:Bcnlr6-vuphUDOMOT4lkW8fXep5qrio1z49UxQhFwUppFMnbCjiGORDC8ALUr_uuttGSQvb88lrBW6s)." *Proceedings of the 24th International Conference on Intelligent User Interfaces*. 2019.

### Fairness issue in real cases or areas

**Fairness in NLP**

1. Bolukbasi, Tolga, et al. "[Man is to computer programmer as woman is to homemaker? debiasing word embeddings](https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)." *Advances in neural information processing systems* 29 (2016): 4349-4357. -- The paper examine gender stereotypes in occupations in word embeddings which they identify  though a survey conducted on peoples perception on gender stereotype. The paper propose a technic to mitigate such identified bias in word embeddings.
2. Gonen, Hila, and Yoav Goldberg. "[Lipstick on a pig: Debiasing methods cover up systematic gender biases in word embeddings but do not remove them](https://papers.nips.cc/paper/2016/file/a486cd07e4ac3d270571622f4f316ec5-Paper.pdf)." *arXiv preprint arXiv:1903.03862* (2019). -- This paper makes a critic on the paper mentioned above's method on mitigating bias in wordembedding.
3. Nissim, Malvina, Rik van Noord, and Rob van der Goot. "[Fair is better than sensational: Man is to doctor as woman is to doctor](https://www.mitpressjournals.org/doi/full/10.1162/coli_a_00379)." (2020): 1-11. -- This paper  makes a critic of using word analogies for concluding bias in word embeddings.
4.  Basta, Christine, Marta R. Costa-Jussà, and Noe Casas. "[Evaluating the underlying gender bias in contextualized word embeddings](https://arxiv.org/pdf/1904.08783.pdf)." *arXiv preprint arXiv:1904.08783* (2019).
5. Zhao, Jieyu, et al. "[Learning gender-neutral word embeddings](https://arxiv.org/pdf/1809.01496.pdf)." *arXiv preprint arXiv:1809.01496* (2018).
6. Kiritchenko, Svetlana, and Saif M. Mohammad. "[Examining gender and race bias in two hundred sentiment analysis systems](https://arxiv.org/pdf/1805.04508.pdf)." *arXiv preprint arXiv:1805.04508* (2018).

**Different cases**

1. Dressel, Julia, and Hany Farid. "[The accuracy, fairness, and limits of predicting recidivism](https://advances.sciencemag.org/content/advances/4/1/eaao5580.full.pdf)." *Science advances* 4.1 (2018): eaao5580.

2. Mukerjee, Amitabha, et al. "[Multi–objective evolutionary algorithms for the risk–return trade–off in bank loan management](https://onlinelibrary.wiley.com/doi/pdf/10.1111/1475-3995.00375?casa_token=LFzwEBUQeVIAAAAA:YKLJSUqrUyshyDCU-VsvT9vvruhyMrbH3EzJtoeP7M92Bdn62jyX1aMLmB7zcfQFlPh4pC1Krz73P7-lFQ)." *International Transactions in operational research* 9.5 (2002): 583-597.
3. Raji, Inioluwa Deborah, and Joy Buolamwini. "[Actionable auditing: Investigating the impact of publicly naming biased performance results of commercial ai products](https://dl.acm.org/doi/pdf/10.1145/3306618.3314244?casa_token=lkiw-D-BXjYAAAAA:sZ6e1I-z9zW-1Qy7KJC5uWDjPk_ZMEtWV65vmA_eOm9skQ39a450_4CECBDpEGTietWbmtuEDGZLDNE)." *Proceedings of the 2019 AAAI/ACM Conference on AI, Ethics, and Society*. 2019.

## Books

1. Weapons of math destruction: How big data increases inequality and threatens democracy
   O'Neil, C., 2016. Broadway Books.
2. Invisible Women - Exposing Data Bias in a World Designed for Men,
   Caroline Criado Perez, 2020, Vintage Publishing
3. Data Feminism, Lauren F. Klein & Catherine D'Ignazio, 2020, Mit Press Ltd
4. Fairness and machine learning - Limitations and Opportunities, Solon Barocas, Moritz Hardt, Arvind Narayanan, in process, https://fairmlbook.org/
5. Practical Fairness, Aileen Nielsen, 2020, O'Reilly Media



<a name="guide-princip"></a>

# Guidelines & principles

## Research articles
In this section we list research articles related to guidelines and principles regarding responsible AI.
1. A. Jobin et al.,
   "[Artificial Intelligence: the global landscape of ethics guidelines](https://www.nature.com/articles/s42256-019-0088-2)",
   Nat. Mach. Intell. (1) 389–399, 2019
2. T. Miller,
   "[Explanation in Artificial Intelligence: Insights from the Social Sciences](https://www.sciencedirect.com/science/article/abs/pii/S0004370218305988)",
   Artificial Intelligence (267) 1-38, 2019
3. C. Rudin,
   "[Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead](https://www.nature.com/articles/s42256-019-0048-x)",
   Nat. Mach. Intell. (1) 206–215, 2019

### Documentation frameworks
1. F. Pinto et al., "Automatic Model Monitoring for Data Streams", [arXiv:1908.04240](https://arxiv.org/abs/1908.04240)
[cs.LG] - Describes a method to monitor models that predict on data streams for detecting model drift.
2. T. Gebru et al., "Datasheets for Datasets", [arXiv:1803.09010](https://arxiv.org/abs/1803.09010)
[cs.DB] - Describes a framework for how to document datasets used for building machine learning models.
3. E. M. Bender and B. Friedman, "[Data Statements for Natural Language Processing: Toward Mitigating System Bias and Enabling Better Science](https://www.mitpressjournals.org/doi/abs/10.1162/tacl_a_00041)",
Transactions of ACL (6), 2018 - Describes a framework for how to document datasets used for NLP tasks.
4. M. Mitchell, "[Model Cards for Model Reporting](https://dl.acm.org/doi/10.1145/3287560.3287596)",
FAT*'19 220-229, 2019 - Describes a framework for how to document ML models.
The [model card toolkit](https://github.com/tensorflow/model-card-toolkit) can be found on github released under the tensorflow repository.
5. I. D. Raji et al.,
"[Closing the AI Accountability Gap: Defining an End-to-End Framework for Internal Algorithmic Auditing](https://dl.acm.org/doi/abs/10.1145/3351095.3372873)",
FAT*'20 33-44, 2020 - Presents a framework for auditing AI/ML based systems. The idea is to use auditing concepts (risk assesment and documentation)
known from other industries, like aerospace or finance, and adjust them to AI/ML. One example is the "Failure Modes and Effect Analysis" (FMEA).

<a name="people-tech"></a>
# People & Tech
1. [Google PAIR: People + AI guidebook for UX professionals and product managers to follow a human centered approach to AI](https://pair.withgoogle.com/guidebook/)
2. [Google’s medical AI was super accurate in a lab. Real life was a different story](https://www.technologyreview.com/2020/04/27/1000658/google-medical-ai-accurate-lab-real-life-clinic-covid-diabetes-retina-disease/)
3. [AI Now Institute reports](https://ainowinstitute.org/reports.html) - Publications of the AI Now Institute

<a name="pol-reg"></a>
# Policy & regulation

## Research articles
1. F. Doshi-Velez and M. Kortz,"Accountability of AI Under the Law: The Role of Explanation",
   [arXiv:1711.01134](https://arxiv.org/abs/1711.01134) [cs.AI]
2. B. Goodman and S. Flaxman,
   "[European Union regulations on algorithmic decision-making and a “right to explanation”](https://ojs.aaai.org/index.php/aimagazine/article/view/2741)",
   AI Magazine (38:3) 50-57, 2017    
3. A. D. Selbst and J. Powles,
   "[Meaningful information and the right to explanation](http://proceedings.mlr.press/v81/selbst18a.html)",
   Proceedings of the 1st FAT (81) 48-48, 2018
4. M. E. Kaminski and G. Malgieri,
   "[Multi-layered Explanations from Algorithmic Impact Assessments in the GDPR](https://dl.acm.org/doi/abs/10.1145/3351095.3372875)",
   FAT*'20 68–79, 2020
5. L. Edwards and M. Veale,
   "[Slave to the Algorithm? Why a 'Right to an Explanation' Is Probably Not the Remedy You Are Looking For](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2972855)",
   16 Duke Law & Technology Review (18), 2017
6. S. Wachter et al.,
   "[Why a Right to Explanation of Automated Decision-Making Does Not Exist in the General Data Protection Regulation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2903469)",
   International Data Privacy Law, 2017
7.

<a name="ux"></a>
# User Experience
