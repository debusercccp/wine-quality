# Wine Quality Analysis and Machine Learning

Applicazione sviluppata con Streamlit per l'analisi esplorativa e la modellazione della qualità del vino.

## Descrizione

Questo progetto permette di:
- Analizzare un dataset di vini (rosso e bianco)
- Applicare filtri interattivi sui dati
- Visualizzare distribuzioni e statistiche
- Addestrare un modello di Machine Learning (Random Forest)
- Esplorare l'importanza delle feature

## Tecnologie utilizzate

- Python
- Streamlit
- Pandas
- Plotly
- Matplotlib
- Scikit-learn

## Struttura del progetto
```bash
wine-quality/
│
├── app.py
├── scr/
│ ├── data_loader.py
│ ├── features.py
│ └── model.py
├── data/
│ ├── winequality-red.csv
│ └── winequality-white.csv
├── requirements.txt
└── README.md
```
---
## Installazione

### Clona il repository:

```bash
git clone https://github.com/RockyT98/wine-quality.git
cd wine-quality
```
### Crea un ambiente virtuale:
```bash
python -m venv .venv
```
### Attiva l'ambiente:

Windows
```bash
.venv\Scripts\activate
```
Mac/Linux
```
source .venv/bin/activate
```
### Installa le dipendenze:
```
pip install -r requirements.txt
```
### Esecuzione

Avvia l'app Streamlit:
```
streamlit run app.py
```
### Demo

https://wine-quality-rwrzfnvs2wwssqbsbq9qrs.streamlit.app/

---
## Funzionalità principali:
- Filtri dinamici (tipo di vino e qualità)
- Visualizzazione dataset
- Istogrammi interattivi
- Distribuzione per tipo di vino
- Addestramento modello Random Forest
- Report di classificazione
- Analisi feature importance
---
## Note:

- Il dataset deve essere presente nella cartella data.
- Il modello viene addestrato dinamicamente dall'app
- Le feature vengono generate nel modulo features.py
- Possibili miglioramenti:
   - Deploy su Streamlit Cloud
   - Aggiunta di altri modelli ML
   - Dashboard più avanzata
   - Ottimizzazione iperparametri
 ---
## Autore
### Rocco Tarantino
