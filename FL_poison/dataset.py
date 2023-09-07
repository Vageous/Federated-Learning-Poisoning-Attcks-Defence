import torch
import pandas as pd
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,TensorDataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# 特殊的符号
from string import punctuation
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split




transform_train = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914,), (0.2023,)),
		])

transform_test = transforms.Compose([
			transforms.RandomCrop(32,padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914,), (0.2023,)),
		])

# dataset
def remove_spaces(data):
    clean_text = data.replace("\\n"," ").replace("\t"," ").replace("\\"," ")
    return clean_text

# defining the function for removing stopwords
stopword = stopwords.words("english") # gives a list of stopwords

def clean_text(data):
    token = word_tokenize(data)
    clean_text = [i.lower() for i in token if (i not in punctuation) 
                  and (i.lower() not in stopword) and (i.isalpha()) and (len(i) > 2)]
    return clean_text

def tokenize_text(vocab,text):
	tokens=word_tokenize(text)
	return [vocab[token] for token in tokens if token in vocab]

# defining the function for getting root words
def lemmatization(data):
    lem = WordNetLemmatizer()
    lst1 = []
    for i in data:
        lem_words = lem.lemmatize(i)
        lst1.append(lem_words)
    return " ".join(lst1)

def data_processing():
	df=pd.read_csv("./data/cancer_text_data/alldata_1_for_kaggle.csv",encoding='latin-1')
	# print(df.head())
	df=df.drop('Unnamed: 0',axis=1)
	new_names = {'0': 'Cancer_Type', 'a':'Research_Paper_Text'}
	df = df.rename(columns=new_names)
	df["Research_Paper_Text"]=df["Research_Paper_Text"].apply(remove_spaces)
	df["Research_Paper_Text"]=df["Research_Paper_Text"].apply(clean_text)
	df["Research_Paper_Text"]=df["Research_Paper_Text"].apply(lemmatization)
	# print(df.head())
	print(df["Research_Paper_Text"].head(1))
	df.Cancer_Type.replace({"Thyroid_Cancer": 0, "Colon_Cancer": 1, "Lung_Cancer": 2}, inplace=True)
	text=df["Research_Paper_Text"]
	label=df["Cancer_Type"]
	
	# vectorizer=TfidfVectorizer(max_df=0.95,max_features=1000,min_df = 10, stop_words="english",lowercase=True)
	vectorizer=CountVectorizer()
	vectorizer.fit(text)
	# text_train=torch.tensor(vectorizer.fit_transform(text_train).toarray())
	vocab=vectorizer.vocabulary_
	text_train,text_test,label_train,label_test=train_test_split(tokenize_text(vocab,text),label,test_size=0.2,random_state=10,stratify=label)
	# print(text_train.shape)
	# print(len(vectorizer.vocabulary_))
	text_train=torch.tensor(text_train)
	# text_test=torch.tensor(vectorizer.transform(text_test).toarray())
	text_test=torch.tensor(test_set)
	label_train=torch.tensor(np.array(label_train))
	label_test=torch.tensor(np.array(label_test))
	train_set=TensorDataset(text_train,label_train)
	test_set=TensorDataset(text_test,label_test)
	return train_set,test_set
	# print(df.head())


def get_data(dataset):

	if dataset=="mnist":
		train_set=datasets.MNIST(root="./data",train=True,download=True,transform=transform_train)
		test_set=datasets.MNIST(root="./data",train=False,download=True,transform=transform_test)
		return train_set,test_set

	elif dataset=="cifar10":
		train_set=datasets.CIFAR10(root="./data",train=True,download=True,transform=transform_train)
		test_set=datasets.CIFAR10(root="./data",train=False,download=True,transform=transform_test)
		return train_set,test_set

	elif dataset=="cancer":
		train_set,test_set=data_processing()
		return train_set,test_set



if __name__=='__main__':
	# data_processing()
	train_set,test_set=get_data("cancer")
	for id,batch in enumerate(DataLoader(train_set,batch_size=16,shuffle=True)):
		print(batch[0])
		print(batch[1])
		print(batch[0].shape)
		print(batch[1].shape)
		break

