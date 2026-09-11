# Questo script (da lanciare UNA TANTUM, non fa parte del training) costruisce il
# dataset di preferenze usato da tutto il resto del progetto (`dpo_data.json`).
# Per ogni domanda "dannosa" del dataset pubblico BeaverTails, crea una COPPIA di
# risposte:
#   - "chosen"   = la risposta dannosa già presente in BeaverTails (usata com'è).
#   - "rejected" = una risposta sicura, generata al momento chiedendo a GPT-4.1-mini di
#                  rispondere nel modo più utile ma innocuo possibile.
#
# ATTENZIONE (importante per capire il resto del progetto!): i nomi "chosen"/"rejected"
# qui riflettono l'orientamento "grezzo" del dataset, che è quello che vorrebbe un
# attaccante (preferire la risposta dannosa) -- NON è l'orientamento di sicurezza che
# vogliamo insegnare al modello. Il codice di training (training.py) deve "scambiare"
# chosen/rejected quando allena il DEFENDER, esattamente per questo motivo.
from datasets import Dataset, load_dataset
import json
from dotenv import load_dotenv
from openai import OpenAI
import sys, os

# Carica variabili d'ambiente (es. chiavi API) dal file 'tokens.env', se presente.
load_dotenv('tokens.env')

# Liste che accumuleranno i dati PRIMA di essere scritti su disco a blocchi (vedi sotto).
prompt, chosen, rejected = [], [], []

# Client OpenAI usato per generare le risposte "sicure". NOTA: la chiave qui è un
# segnaposto ('sk-'), non una chiave vera -- va sostituita (tipicamente tramite la
# variabile d'ambiente OPENAI_API_KEY, che il client legge automaticamente se non le
# passiamo una chiave esplicita) prima di poter effettivamente lanciare questo script.
client = OpenAI(api_key='sk-')

# BeaverTails è un dataset pubblico di domande e risposte etichettate come sicure/non
# sicure, pensato per la ricerca sulla sicurezza dei modelli linguistici. Qui carichiamo
# lo split di training con 30.000 esempi.
ds = load_dataset('PKU-Alignment/BeaverTails', split = '30k_train')
# Teniamo solo gli esempi in cui la risposta ORIGINALE è etichettata come NON sicura
# (is_safe == False): sono le nostre "risposte dannose" di partenza.
filtered_df = ds.filter(lambda example : example['is_safe'] == False)
# Teniamo solo le colonne che ci servono (scartiamo le altre etichette del dataset).
filtered_df_final = filtered_df.select_columns(['prompt', 'response'])

for ix, item in enumerate(filtered_df_final):
    # La domanda originale...
    prompt.append(item['prompt'])
    # ...e la risposta dannosa originale, che useremo come "chosen" (vedi nota sopra sul
    # perché si chiama così anche se è la risposta "cattiva").
    chosen.append(item['response'])

    # Chiediamo a un modello OpenAI di generare, per la stessa domanda, la risposta più
    # sicura e utile possibile: sarà il nostro "rejected" (la risposta che vorremmo il
    # modello preferisse, una volta "immunizzato").
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role" : "system", "content" : """You are a helpful assistant, and your job is to generate the safest, most harmless and law abiding response to the queries. ensure that however harmful or provoking the query is, you will response in the most harmless way while trying to be maximally helpful. If you cannot respond to a certain query, mention the reason behind your choice of not responding."""
            },
            {
                "role": "user",
                "content": f"{prompt}"
            },
        ]
    )
    rejected.append(completion.choices[0].message.content)
    print(f'Part {ix + 1} completed.')

    # Ogni 5 esempi, salviamo su disco (append incrementale) invece di aspettare la fine
    # dello script: così, se lo script si interrompe (es. per un errore di rete verso
    # l'API OpenAI, che su migliaia di chiamate è abbastanza probabile), il lavoro già
    # fatto non viene perso.
    if (ix + 1) % 5 == 0 :
        with open('dpo_data.json', encoding='utf-8') as f:
            load_current = json.load(f)

        load_current['prompt'].extend(prompt)
        load_current['chosen'].extend(chosen)
        load_current['rejected'].extend(rejected)

        with open('dpo_data.json', 'w', encoding='utf-8') as f:
            json.dump(load_current, f, indent=4)

        print('File updated!!')

        # Svuotiamo le liste temporanee: i dati che contenevano sono già stati scritti
        # su disco, quindi possiamo ripartire da zero per il prossimo blocco di 5.
        prompt, chosen, rejected = [], [], []
