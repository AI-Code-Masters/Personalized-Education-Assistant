from flask import Flask, render_template, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import nltk
from nltk.chat.util import Chat, reflections

app = Flask(__name__)

# Sample pairs of patterns and responses for the chatbot
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, How are you today ?",]
    ],
    [
        r"hi|hey|hello",
        ["Hello", "Hey there",]
    ],
    [
        r"what is your name?",
        ["I am a bot created for personalized education assistance.",]
    ],
    [
        r"how are you ?",
        ["I'm doing good. How about you?",]
    ],
    [
        r"sorry (.*)",
        ["It's alright", "It's OK, never mind",]
    ],
    [
        r"i'm (.*) doing good",
        ["Nice to hear that", "Alright, great!",]
    ],
    [
        r"quit",
        ["Bye for now. See you soon!", "It was nice talking to you. Goodbye!"]
    ],
]

chat = Chat(pairs, reflections)

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat_response():
    user_message = request.json.get("message")
    bot_response = chat.respond(user_message)
    
    if bot_response is None:  
        input_ids = tokenizer.encode(user_message, return_tensors='pt')
        output = model.generate(input_ids, max_length=50, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id,temperature=0.9,top_k=50)
        bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(debug=True)
