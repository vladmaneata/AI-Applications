import re     # regular expressions library
import pandas as pd
import nltk  # standard library for text preprocessing tasks (Natural Elementary Toolkit)
import matplotlib.pyplot as plt
import seaborn as sns  # powerful visualization library for Python
from keras.utils.data_utils import pad_sequences
from keras.layers import Dense, Embedding, LSTM, Dropout, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report, confusion_matrix

# include data
data = pd.read_excel(r'https://www.math.uaic.ro/~stoleriu/HS_dataset.xlsx')
print(data.shape)
print(data.head(7))     # prints first 7 labelled uncleaned messages
print(data['label'].value_counts(normalize=True))
sns.countplot(x=data['label'])   # distribution of labels {HAM, SPAM}
plt.show()

# data preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_data(text):
    # Convert to string and lowercase
    text = str(text).lower()

    # Remove HTML, URLs and non-alphanumeric characters
    text = re.sub('<[^<]+?>', '', text)
    text = re.sub('https://.*', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Split, remove stopwords and Lemmatize
    words = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]

    return " ".join(cleaned)

data['message'] = data['message'].apply(clean_data)
X = data['message']
print(X.head(7))    # prints first 7 labelled cleaned messages

# convert the labels to numeric form
class_label = data.label.factorize()  # Ensures that the labels in the Excel file are binary (0 and 1)
print(class_label)

# tokenize words (convert text into an array of vector embeddings)
tweet = data.message.values
t = Tokenizer(num_words=50000)    # initializing the tokenizer
t.fit_on_texts(tweet)             # fitting on the text data
vocab_size = len(t.word_index) + 1
encoded_mess = t.texts_to_sequences(tweet)   # creating the numerical sequence

# look into some text and corresponding numerical sequence
for i in range(5):
    print("Text               : ", X[i])
    print("Numerical Sequence : ", encoded_mess[i])

# use padding to pad the sentences to have equal length
max_len_seq = max([len(i) for i in encoded_mess])  # find the length of the largest sequence
# ensures every message has max_len_seq by adding zeros to shorter messages
padded_sequence = pad_sequences(encoded_mess, maxlen=max_len_seq)

# build the text classifier
embedding_vector_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_length, input_length=max_len_seq)) # Maps each word to a 32-dimensional vector
model.add(SpatialDropout1D(0.5))  # Drops the entire feature channel across all time steps
model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.25))  # This is the model's brain. It uses 50 units
model.add(Dropout(0.4))    # This is standard dropout, randomly dropping individual elements
model.add(Dense(20, activation="relu"))
model.add(Dropout(0.3))    # Also standard dropout
model.add(Dense(1, activation='sigmoid'))  # Sigmoid activation returns a value between 0 and 1
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

# train the model
history = model.fit(padded_sequence, class_label[0], validation_split=0.2, epochs=5, batch_size=16)

# plot metrics
plt.plot(history.history['accuracy'], label='acc')
plt.legend()
plt.show()
plt.savefig("Accuracy plot.jpg")

plt.plot(history.history['loss'], label='loss')
plt.legend()
plt.show()
plt.savefig("Loss plt.jpg")

# do some predictions
def detect_spam(text):
    cleaned_text = clean_data(text)  # Clean the input first
    tw = t.texts_to_sequences([cleaned_text])
    tw = pad_sequences(tw, maxlen=max_len_seq)  # zero padding to reach max_len_seq size
    prediction = int(model.predict(tw).round().item())
    print("Predicted label: ", class_label[1][prediction])


# Get predictions (0 or 1)
y_pred = (model.predict(padded_sequence) > 0.5).astype("int32")

# Print the confusion matrix and the scores
print(confusion_matrix(class_label[0], y_pred))
print(classification_report(class_label[0], y_pred, target_names=['Ham', 'Spam']))
# Precision: Of all labelled Spams, how many were actually Spam?
# (avoids false alarms)
# Recall: Of all the actual Spams, how many did the model catch?
# (avoids missing scams)
# F1-Score: If low, the model is biased toward one class

# Testing for unseen texts
test_sentences = [
    'Congratulations!!! If you are 18+, retrieve your prize http://jump123.com/cTuWX3',
    'I like Python very much, but I would rather prefer chicken :)',
    'URGENT: Your April 1st bonus is waiting! Click here to claim your $500 gift card before the holiday ends: http://bit.ly/April1Promo'
]

for i, text in enumerate(test_sentences, 1):
    ordinal = "First" if i == 1 else "Second" if i == 2 else "Third" if i == 3 else f"{i}th"
    print(f"--- {ordinal} text: ---")
    detect_spam(text)
    print("\n") # Adds a newline for spacing


