# responsible-ai-knowledge-base



# About the repo
...

# Table of contents

|        | | |
| ------------- |:-------------:| -----:|
| [Explainable AI](#explainable-ai)      | [Fairness](#fairness) | [Guidelines & principles](#guide-princip)
| [People & Tech](#people-tech)  | [Policy & Regulation](#pol-reg)      | [User Experience](#ux) |


<a name="explainable-ai"></a> 
# Explainable AI
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
12. [What-If Tool](https://pair-code.github.io/what-if-tool/) - Open source frmaework from Google to probe the behaviour
of a trained model.
13. [AllenNLP Interpret](https://allennlp.org/interpret) - Python framework for explaining deep neural networks 
for language processing developed by the Allen Institute for AI.

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

### Evaluation of XAI
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
FAT*'20 56-67, 2020 - Framework (essentialy a list of questions or checklist) to evaluate and document XAI methods. 
Also includes question that are relevant to the context in which the XAI methods should be employed, i.e. changing the outcome of the assessment based on the context.
8. E. S. Jo and T. Gebru, 
"[Lessons from archives: strategies for collecting sociocultural data in machine learning](https://dl.acm.org/doi/abs/10.1145/3351095.3372829)", 
FAT*'20 306-316, 2020 - Use archives as inspiration of how to collect, curate and annotate data.

## Books
1. [Explainable AI: Interpreting, Explaining and Visualizing Deep Learning](https://doi.org/10.1007/978-3-030-28954-6) - 
Explainability with respect to deep learning with a focus on convolutional neural networks used for image data.
The editor of the book are also behind the layerwise relvance propagation (LRP) method.
2. [Explainable and Interpretable Models in Computer Vision and Machine Learning](https://www.springer.com/gp/book/9783319981307) -
More general book about explainability in machine learning, but also with a focus on deep learning in computer vison.

<a name="fairness"></a> 
# Fairness

<a name="guide-princip"></a>
# Guidelines & principles

## Research articles
In this section we list research articles related to guidelines and principles regarding responsible AI.

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

<a name="pol-reg"></a>
# Policy & regulation

<a name="ux"></a>
# User Experience



