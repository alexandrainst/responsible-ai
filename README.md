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
11. Q. Zhang and S.-C. Zhu, 
    "[Visual Interpretability for Deep Learning: a Survey](https://link.springer.com/article/10.1631/FITEE.1700808)",
    Technol. Electronic Eng. (19) 27–39, 2018

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
   IEEE ICCV 3449-3457, 2017
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
24. C. Chen et al., 
    "[This Looks Like That: Deep Learning for Interpretable Image Recognition](https://papers.nips.cc/paper/2019/hash/adf7ee2dcf142b0e11888e72b43fcb75-Abstract.html)",
    NeurIPS, 2019

### XAI in the medical domain
This section contains paper in which XAI models or methods were used on medical data.
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



