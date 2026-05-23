# SERA: Evaluating LLM-based SEmantic RAnking over Structured Tables
Per definire una nuova famiglia di test (ovvero un nuovo dataset),
è necessario aggiungerne la relativa cartella in `queries`. All'interno di tale
cartella, si aggiungono poi tanti file quanti sono i test che si vogliono
implementare per quella famiglia. Il nome del file identificherà il nome del test.

Ogni file di test deve definire una (sola) classe che estenda la classe `Test`, definita nel file `test.py`.
La classe dovrà definire necessariamente alcuni metodi, come descritto nella documentazione della classe `Test`.

Per eseguire uno (o più) test, si genera un file `.yaml` all'interno della cartella
`experiments`. Il file ha la seguente struttura:

```yaml
queries: # test da eseguire
models: # modelli con cui eseguire le query
run_type: # tipo di esecuzione
disabled: # opzionale, se true disabilita l'esecuzione del file
seed: # opzionale, setta parametro 'seed' all'interno di ciascun Test
```
Sia il campo `queries` che il campo `models` possono essere:
- singoli nomi
- liste di nomi e/o dizionari. In questo modo è possibile sovrascrivere i parametri di default
    definiti nelle classi `TestParameters` all'interno dei singoli Test o nei file `.yaml` dei modelli.

Sia query che modelli si identificano con la loro cartella di appartenenza + il nome 
del file (senza estensione): `movies/movie2similars` o `together/deepseek-V3-together`.
Esempio completo:
```yaml
models: 'together/deepseek-V3-together'
queries:
  - 'movies/movie2similars'
  - name: 'purchases/customer_segmentation' # nome del test
    top_k: 5 # un parametro del test che verrà sovrascritto
```

Il file ha un terzo parametro `run_type` che può essere:
- `direct`: esegue i test in modalità direct prompting
- `lotus`: esegue i test tramite lotus

Ciascuna modalità richiede un'implementazione all'interno della classe di test (come
da documentazione della classe `Test`).

In caso vengano specificati più modelli, più modalità di esecuzioni e/o più query,
verranno generati tutti i test possibili combinando i parametri specificati.

I dati di esecuzione vengono salvati nella cartella `runs`, all'interno di una cartella
con lo stesso nome del file `.yaml`.

Per eseguire i testi, lanciare il file `experimenter.py`: verranno eseguiti tutti i 
file `.yaml` (non disabilitati) presenti nella cartella `experiments`.
