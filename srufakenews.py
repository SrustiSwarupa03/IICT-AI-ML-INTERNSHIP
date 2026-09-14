#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import re
import string
from wordcloud import WordCloud
get_ipython().system('pip install scikit-learn')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

#loading the data
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

#fake_dataset
display(fake.head(10))
print(fake.subject.value_counts())

#true_dataset
display(true.head(10))
print(true.subject.value_counts())

#category for whether fake or true
fake['category']=1
true['category']=0

#joining the datasets
data = pd.concat([fake,true]).reset_index(drop = True)

#setting the figure
plt.figure(figsize=(6,7))
sns.countplot(x = data['category'])
plt.legend()

data= data[['text','category']]

#DATA CLEANING
print(data.isna().sum()*100/len(data))
data= data.dropna(subset=['text'])

#checking for empty strings in TEXT
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_words = [word for word in words if word.isalnum() and word not in stop_words]
    return " ".join(clean_words)
data['text'] = data['text'].apply(clean_text)
data.head(10)
data.to_csv('cleaned_articles_dataset.csv',index=False)
print("Data cleaning complete!!! file saved as news.csv")

#WORD CLOUD
#fake news
plt.figure(figsize=(20,10))
word_cloud = WordCloud(max_words = 200, width = 800, height = 400).generate(" ".join(data[data.category==0].text))
plt.axis("off")
plt.imshow(word_cloud,interpolation='bilinear')
plt.show()

#true news
plt.figure(figsize=(20,10))
word_cloud = WordCloud(max_words = 200, width = 800, height = 400, background_color= 'skyblue').generate(" ".join(data[data.category==1].text))
plt.axis("off")
plt.imshow(word_cloud,interpolation='bilinear')
plt.show()

#feature engineering
X= data["text"]
Y= data["category"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#feature extraction configuration
vectorizer = TfidfVectorizer(max_features = 5000)
X_Vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

models = {
          "KNN": KNeighborsClassifier(n_neighbors =5),
          "LogReg": LogisticRegression(max_iter =1000),#train parametric model
          "Naive Bayes" : MultinomialNB(),#train non-parametric model
          "RandomForest":RandomForestClassifier(n_estimators =100),
          "NeuralNet": MLPClassifier(hidden_layer_sizes=(100, ), max_iter=300),
          "PassiveAggressive": PassiveAggressiveClassifier(max_iter = 50, random_state = 42)
         }
name_list =[]
score =[]

for name, model in models.items():
    model.fit(X_Vec,Y_train)
    predictions = model.predict(X_test_vec)
    accuracy = accuracy_score(Y_test,predictions)
    name_list.append(name)
    score.append(accuracy)
    print(f"{name} Accurancy:",accuracy)
    print(classification_report(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))
    confuse_mat = confusion_matrix(Y_test, predictions)
    display = ConfusionMatrixDisplay(confusion_matrix = confuse_mat, display_labels = ['fake = 1', 'true = 0'])
    display.plot(cmap ='pink', values_format = 'd')
    plt.title(f"Confusion Matrix:{name}")
    plt.show()

#bar chart
plt.figure(figsize=(8, 5))
plt.bar(name_list, score, color = 'skyblue')
plt.title('Fake News Detection')
plt.ylabel('accuracy')
plt.show()

#selecting the best model
best_model_name = name_list[score.index(max(score))]
print("best model:", best_model_name)
best_model = models[best_model_name]
predict_model = best_model.predict(X_test_vec)
for i in range(10):
        print("Actual:", Y_test.iloc[i], "prediction:", predict_model[i])

