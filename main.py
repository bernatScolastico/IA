import random
import json
import torch
import pywebio

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


# A simple script to calculate BMI
from pywebio.input import input, FLOAT
from pywebio.output import put_text

def bmi():
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('intents.json', 'r') as f:
        intents=json.load(f)

    FILE="data.pth"
    data=torch.load(FILE)

    input_size= data["input_size"]
    hidden_size= data["hidden_size"]
    output_size= data["output_size"]
    all_words=data["all_words"]
    tags=data["tags"]
    model_state=data["model_state"]


    model= NeuralNet(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    bot_name= "Sam"
    print("Parliamo! Scrivere 'quit' per uscire")
    while True:
        sentence=input('Tu:')
        if sentence =="quit":
            
            break

        stringsentence=sentence
        
        sentence=tokenize(sentence)
        X=bag_of_words(sentence, all_words)
        X= X.reshape(1, X.shape[0])
        X=torch.from_numpy(X)
        
        output=model(X)
        _, predicted= torch.max(output, dim=1)
        tag=tags[predicted.item()]

        probs=torch.softmax(output, dim=1)
        prob= probs[0][predicted.item()]

        if prob.item()>0.75:
            for intent in intents["intents"]:
                if tag==intent["tag"]:
                    #print(f"{bot_name}: {random.choice(intent['responses'])}")
                   response = random.choice(intent['responses'])
                    chat_history.append({"user": stringsentence,"bot":response})  # Aggiungi la risposta del bot alla chat
                    put_text(f"Tu: {stringsentence}")
                    put_text(f"{bot_name}: {response}")

                    # Invia la chat al server come file JSON
                    data = {"chat_history": chat_history}
                    headers = {'Content-type': 'application/json'}
                    response = requests.post('http://localhost/backend_bernat/API.php', json=data, headers=headers)
                    if response.status_code == 200:
                        print("Chat inviata con successo al server.")
                    else:
                        print("Errore nell'invio della chat al server.")


        else:
            #print(f"{bot_name}: Non capisco...")
            put_text(f"{bot_name}: Non capisco...")

    
pywebio.start_server(bmi, port=8990)



#def bmi():
#    height = input("Input your height(cm)：", type=FLOAT)
#    weight = input("Input your weight(kg)：", type=FLOAT)
#
#    BMI = weight / (height / 100) ** 2
#
#    top_status = [(16, 'Severely underweight'), (18.5, 'Underweight'),
#                  (25, 'Normal'), (30, 'Overweight'),
#                  (35, 'Moderately obese'), (float('inf'), 'Severely obese')]
#
#    for top, status in top_status:
#        if BMI <= top:
#            put_text('Your BMI: %.1f. Category: %s' % (BMI, status))
#            break
#
