import json
import random
import string
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmer = WordNetLemmatizer()

with open('chatbot.txt', 'r', encoding='utf8', errors='ignore') as fin:
    raw = fin.read().lower()

sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)


def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_RESPONSES = {
    "hello": ["Hello, I am glad! You are talking to me"],
    "hi": ["Hi, I am glad! You are talking to me"],
    "greetings": ["Hi, I am glad! You are talking to me", "Hello, I am glad! You are talking to me"],
    "sup": ["Sup? How can I assist you today?"],
    "hey": ["Hey there! How can I help you today?"],
    "hola": ["Hola! How can I help you?"],
    "bonjour": ["Bonjour! How can I assist you today?"],
    "goodmorning": ["Good morning. I hope you had a good night's sleep. How are you feeling today?"],
    "goodafternoon": ["Good afternoon. How is your day going?"],
    "goodevening": ["Good evening. How has your day been?"],
    "goodnight": ["Good night. Get some proper sleep.", "Good night. Sweet dreams."],
    "whoareyou": ["I'm Jamila, your Personal Therapeutic AI Assistant. How are you feeling today?", 
                    "I'm Pandora, a Therapeutic AI Assistant designed to assist you. Tell me about yourself.",
                    "You can call me Pandora.", "I'm Pandora!", "Call me Pandora"],
    "whatareyou": ["I am a chatbot, designed to help you with mental health and wellness. How are you feeling today?"],
    "whoyouare": ["I am Pandora, your personal therapeutic assistant. How can I help you today?"],
    "tell me more about yourself": ["I am Pandora, your assistant designed to help you work through your emotions. How can I assist you today?"],
    "whatisyourname": ["My name is Pandora. How can I assist you today?"],
    "what'syourname": ["I'm Pandora!", "You can call me Pandora!"]
}

def greeting(sentence):
    """If user's input matches a greeting, return the corresponding response"""
    for word in sentence.split():
        word = word.lower()
        if word in GREETING_RESPONSES:
            return random.choice(GREETING_RESPONSES[word])
    return None

def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        robo_response = "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = sent_tokens[idx]
        return robo_responses

flag = True
print("ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!")

while flag:
    user_response = input()
    user_response = user_response.lower()

    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ROBO: You are welcome..")
        else:
            greeting_response = greeting(user_response)
            if greeting_response:
                print("ROBO: " + greeting_response)
            else:
                print("ROBO: ", end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag = False
        print("ROBO: Bye! Take care..")
