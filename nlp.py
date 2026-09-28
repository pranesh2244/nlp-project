import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")

file_path = r"cars_in_india_500_lines.txt"

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()


print("\n================================")
print("FILE INFORMATION")
print("================================")

print("File loaded successfully!")


sentences = sent_tokenize(text)

print("Total sentences:", len(sentences))


stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess(sentence):

    sentence = sentence.lower()

    words = word_tokenize(sentence)

    words = [
        word
        for word in words
        if word.isalpha() and word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)

clean_sentences = []
original_sentences = []

for sentence in sentences:

    clean_text = preprocess(sentence)

    if clean_text.strip():

        clean_sentences.append(clean_text)
        original_sentences.append(sentence)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(
    clean_sentences
)

while True:

    question = input(
        "\nAsk a question (type 'exit' to stop): "
    )

    if question.lower() == "exit":

        print("Program stopped.")
        break



    clean_question = preprocess(question)

    question_vector = vectorizer.transform(
        [clean_question]
    )

    similarity = cosine_similarity(
        question_vector,
        tfidf_matrix
    )
    best_index = similarity.argmax()

    best_score = similarity[0][best_index]

    answer = original_sentences[best_index]

    print("\n================================")
    print("ANSWER")
    print("================================")

    if best_score < 0.10:

        print(
            "Sorry, I could not find a "
            "relevant answer in the file."
        )

    else:

        print(answer.strip())

        print(
            "\nSimilarity Score:",
            round(best_score, 3)
        )