import os
from flask import Flask, request, jsonify
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import openai
from openai.error import RateLimitError

openai.api_key = os.environ.get('API_KEY')
nltk.download('vader_lexicon')
nltk.download('punkt')

emotions = {
    'neutral': {'rate': 0.5, 'volume': 0.6, 'pitch': 1.0},
    'excited': {'rate': 0.55, 'volume': 0.85, 'pitch': 1.125},
    'happy': {'rate': 0.5, 'volume': 0.75, 'pitch': 1.05},
    'sad': {'rate': 0.45, 'volume': 0.45, 'pitch': 0.775},
    'surprised': {'rate': 0.55, 'volume': 0.75, 'pitch': 1.4},
    'angry': {'rate': 0.525, 'volume': 0.8, 'pitch': 0.95}
}

def map_to_emotion(sentiment_score):
    compound_score = sentiment_score['compound']
    magnitude = abs(compound_score)
    
    if compound_score >= 0.2:
        if magnitude >= 0.5:
            return 'excited'
        else:
            return 'happy'
    elif compound_score >= 0.05:
        if magnitude >= 0.5:
            return 'surprised'
        else:
            return 'happy'
    elif compound_score <= -0.2:
        if magnitude >= 0.5:
            return 'sad'
        else:
            return 'angry'
    elif compound_score <= -0.1:
        return 'angry'
    else:
        return 'neutral'



def generate_story(selected_likes):
    prompt = "Create a prompt for generating a short story with themes including: " + ", ".join(selected_likes) + ". Ensure that the generated prompt includes clear instructions for crafting an engaging narrative. Emphasize the importance of vivid imagery leaving the reader eager for more. Explore the selected themes and create a story that resonates with the audience."
    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=prompt,
    max_tokens=1000
    )

    story_prompt = "Craft an engaging, meaningful short story, not exceeding two minutes in narration, based on the following " + response.choices[0].text.strip()+ ". additionaly  the story should have a title. give the title in the first line without any prefix title."
    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=story_prompt,
    max_tokens=1000
    )
    title, story =response.choices[0].text.strip().split('\n\n', 1)
    sentences = nltk.sent_tokenize(story)

    sid = SentimentIntensityAnalyzer()
    sentences_data = []
    sentences_data.append({
        'title': title
    })
    for sentence in sentences:
        sentiment_score = sid.polarity_scores(sentence)
        emotion = map_to_emotion(sentiment_score)
        rate = emotions[emotion]['rate']
        volume = emotions[emotion]['volume']
        pitch = emotions[emotion]['pitch']
        sentences_data.append({
            'sentence': sentence,
            'emotion': emotion,
            'rate': rate,
            'volume': volume,
            'pitch': pitch
        })

    return sentences_data

app = Flask(__name__)

@app.route('/generate_story', methods=['POST'])
def generate_story_endpoint():
    request_json = request.get_json()
    selected_likes = request_json['selected_likes']
    try:
        result = generate_story(selected_likes)
        return jsonify({'result': result})
    except RateLimitError as e:
        error_message = str(e)
        return jsonify({'error': error_message}), 429  # Return HTTP status code 429 for rate limit exceeded

if __name__ == "__main__":
    app.run(debug=True)
