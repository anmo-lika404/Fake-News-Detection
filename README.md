# Fake-News-Detection

#ABSTRACT#
It is true that with great power comes great responsibility.The internet and social media hold immense power to spread information like a wildfire. But is the information spread always credible? Fake news i.e. a subject including news, data, report and information that is wholly or partly false, is one of the biggest scandals in the the digitally connected world. Its impact on the society is humungous. It creates a threat to national security, economy, prosperity and individual.Billions of articles created every day on the web, people might be spreading without knowing this news is real or fake. A simple action has become a serious issue if there is no control gate to prevent fake news stories being spread aggressively.Deep learning models are widely used for linguistic can detect complex patterns in textual data. Long Short-Term Memory (LSTM) is a tree-structured recurrent neural network used modeling. Typical deep learning models such as Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to analyze variable-length sequential data. Bi-directional LSTM allows looking at particular sequence both from front-to-back as can detect complex patterns in textual data. Long Short-Term Memory (LSTM) is a tree-structured recurrent neural network used well as from back-to-front. The paper presents a fake news detection model based on Bi-directional LSTM-recurrent neural to analyze variable-length sequential data. Bi-directional LSTM allows looking at particular sequence both from front-to-back as network. Two publicly available unstructured news articles datasets are used to assess the performance of the model. The result well as from back-to-front. The paper presents a fake news detection model based on Bi-directional LSTM-recurrent neural shows the superiority in terms of accuracy of Bi-directional LSTM model over other methods namely CNN, vanilla RNN and network. The aim of this project is to apply natural language processing(NLP)techniques for text analytics and train a deep learning model to detect fake news based on news content or title.This model can be applied to real-world social media and can be used to eliminate bad experience for users.NLP techniques like regular expression, tokenisation are used before vectorising using Word2Vec library.

**PROBLEM STATEMENT**

Fake news is “a news article that is intentionally and verifiably false” . A news article is a sequence of words. Hence in past, many authors propose the use of text mining techniques and machine learning techniques to analyze news textual data to predict the news credibility. With more computational capabilities and to handle massive datasets, deep learning models present a finer performance over traditional text mining techniques and machine learning techniques. Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN), are widely explored Deep Neural Network (DNN) architectures to handle various NLP tasks .The current work is related to number of research areas such as text classification, rumour detection, spammer detection, and sentiment analysis.
Fake news can be identified using different machine learning methods. Authors proposed a simple approach for fake news detection using naive Bayes classifier is tested against a data set of Facebook news posts. Performance evaluation of multiple classification algorithms namely Support Vector Machines, Gradient Boosting, Bounded Decision Trees and Random Forests on a corpus of about 11,000 news articles are presented in .

**METHODOLOGY**

In this section I have described my LSTM based fake news detection system.

Data Acquisition and Exploratory Data Analysis:
I have worked on a dataset from Kaggle imported in my code by uploading it on my Github repository only. Then the data is converted into lists form .csv files. I have plotted a histogram as to get a fair idea of what genre of news are present in the data. The plotting of Wordcloud gives an idea of the words frequently occurring in the datasets.

Data Cleaning:
Data without publishers contributes towards fake news and so do some tweets. Data without a valid publication source or with characters <120 is regarded as fake. The news content is also appended with the titles.All the data is also converted into lower case to reduce the vocabulary size.

Data Preprocessing:
Real News is classified as ‘1’ and Fake News is classified as ‘0’.Special characters which do not contribute any special meaning to the data are also removed. Our data consisting of words is now to be converted to vectors for the further steps, so we take help of Word2Vec Library to create our word embeddings.
Word embeddings is a technique where individual words are transformed into a numerical representation of the word (a vector). Where each word is mapped to one vector, this vector is then learned in a way which resembles a neural network. The vectors try to capture various characteristics of that word with regard to the overall text. These characteristics can include the semantic relationship of the word, definitions, context, etc. With these numerical representations, you can do many things like identify similarity or dissimilarity between words.Clearly, these are integral as inputs to various aspects of machine learning. A machine cannot process text in their raw form, thus converting the text into an embedding will allow users to feed the embedding to classic machine learning models. The simplest embedding would be a one hot encoding of text data where each vector would be mapped to a category.Clearly, these are integral as inputs to various aspects of machine learning. A machine cannot process text in their raw form, thus converting the text into an embedding will allow users to feed the embedding to classic machine learning models. The simplest embedding would be a one hot encoding of text data where each vector would be mapped to a category. 
Word2vec is a technique for natural language processing published in 2013. The word2vec algorithm uses a neural network model to learn word associations from a large corpus of text. Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence. As the name implies, word2vec represents each distinct word with a particular list of numbers called a vector. The vectors are chosen carefully such that a simple mathematical function (the cosine similarity between the vectors) indicates the level of semantic similarity between the words represented by those vectors.The effectiveness of Word2Vec comes from its ability to group together vectors of similar words. Given a large enough dataset, Word2Vec can make strong estimates about a words meaning based on their occurrences in the text. 

Then we further Tokenize our text. Tokenization is a method to segregate a particular text into small chunks or tokens. Here the tokens or chunks can be anything from words to characters, even subwords.The Tokenizer class of Keras is used for vectorizing a text corpus. For this either, each text input is converted into integer sequence or a vector that has a coefficient for each token in the form of binary values.texts_to_sequences method helps in converting tokens of text corpus into a sequence of integers
In order to reassure all our embeddings are of the same length we add padding to the embeddings.

Then we create our model:

We use the LSTM model on our data. The activation function used is ‘Sigmoid’ as we only have to classify into 2 types.Sigmoid specifically, is used as the gating function for the three gates (in, out, and forget) in LSTM, since it outputs a value between 0 and 1, and it can either let no flow or complete flow of information throughout the gates.
Mathematical expression: sigmoid(z) = 1 / (1 + exp(-z))
First-order derivative: sigmoid'(z) = -exp(-z) / 1 + exp(-z)^2

The optimiser we use is ‘Adam’ optimiser.Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent. The method is really efficient when working with large problem involving a lot of data or parameters. It requires less memory and is efficient. Intuitively, it is a combination of the ‘gradient descent with momentum’ algorithm and the ‘RMSP’ algorithm.Adam optimizer involves a combination of two gradient descent methodologies: 
Momentum: 
This algorithm is used to accelerate the gradient descent algorithm by taking into consideration the ‘exponentially weighted average’ of the gradients. Using averages makes the algorithm converge towards the minima in a faster pace. 

Root Mean Square Propagation (RMSP): 
Root mean square prop or RMSprop is an adaptive learning algorithm that tries to improve AdaGrad. Instead of taking the cumulative sum of squared gradients like in AdaGrad, it takes the ‘exponential moving average’.

Adam Optimizer inherits the strengths or the positive attributes of the above two methods and builds upon them to give a more optimized gradient descent.
Loss function used is ‘Binary Cross Entropy’.Binary Cross Entropy is the negative average of the log of corrected predicted probabilities.


The evaluation metrics used is ‘accuracy’.Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition:
Accuracy =Number of correct predictions\Total number of predictions


![Methodology of the Fake News Detection Model](https://user-images.githubusercontent.com/67863699/169004756-306a2f82-195b-433b-874a-7a6b799b62e0.png)
