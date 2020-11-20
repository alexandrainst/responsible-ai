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

## Courses
1. [Kaggle - Machine Learning Explainability](https://www.kaggle.com/learn/machine-learning-explainability) - 
Kaggle course about the basics of XAI with example notebooks and exercises.

## Research articles

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

<a name="people-tech"></a>
# People & Tech
1. [Google PAIR: People + AI guidebook for UX professionals and product managers to follow a human centered approach to AI](https://pair.withgoogle.com/guidebook/)

<a name="pol-reg"></a>
# Policy & regulation

<a name="ux"></a>
# User Experience



