import os
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import openai

openai.api_key = os.environ.get('API_KEY')
nltk.download('vader_lexicon')
nltk.download('punkt')

emotions = {
    'neutral': {'rate': 0.5, 'volume': 1.0},
    'happy': {'rate': 0.5, 'volume': 0.6},
    'sad': {'rate': 0.4, 'volume': 0.4},
    'angry': {'rate': 0.6, 'volume': 0.75},
    'excited': {'rate': 0.6, 'volume': 0.6},
    'fearful': {'rate': 0.5, 'volume': 0.5},
    'calm': {'rate': 0.45, 'volume': 0.4},
    'surprised': {'rate': 0.6, 'volume': 0.8},
    'tender': {'rate': 0.4, 'volume': 0.45}
}

def map_to_emotion(sentiment_score):
    compound_score = sentiment_score['compound']
    if compound_score >= 0.2:
        return 'excited'
    elif compound_score <= -0.2:
        return 'fearful'
    elif compound_score >= 0.1:
        return 'happy'
    elif compound_score <= -0.1:
        return 'sad'
    elif -0.1 < compound_score < 0.1:
        return 'neutral'
    elif compound_score >= 0.05:
        return 'surprised'
    elif compound_score <= -0.05:
        return 'angry'
    else:
        return 'calm'


def generate_story(selected_likes):
    prompt = "Create a prompt for generating a short story with themes including: " + ", ".join(selected_likes)
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=1000
    )

    story_prompt = "You are a short story writer don't exceed 200 words " + response.choices[0].text.strip()
    response = openai.Completion.create(
        model="gpt-3.5-turbo-instruct",
        prompt=story_prompt,
        max_tokens=1000
    )

    story = response.choices[0].text.strip()
    sentences = nltk.sent_tokenize(story)

    sid = SentimentIntensityAnalyzer()
    sentences_data = []

    for sentence in sentences:
        sentiment_score = sid.polarity_scores(sentence)
        emotion = map_to_emotion(sentiment_score)
        rate = emotions[emotion]['rate']
        volume = emotions[emotion]['volume']
        sentences_data.append({
            'sentence': sentence,
            'emotion': emotion,
            'rate': rate,
            'volume': volume
        })

    return sentences_data

app = Flask(__name__)

@app.route('/generate_story', methods=['POST'])
def generate_story_endpoint():
    request_json = request.get_json()
    selected_likes = request_json['selected_likes']
    result = generate_story(selected_likes)
    return jsonify({'result': result})

# if __name__ == '__main__':
#     app.run(debug=True)
