import json
from mitmproxy import http

class OllamaLogger:
    def __init__(self):
        # Apriamo un file in modalità append per salvare i log in formato JSONL
        self.logfile = open("lotus_ollama_logs.jsonl", "a", encoding="utf-8")

    def response(self, flow: http.HTTPFlow):
        # Filtriamo solo le chiamate dirette a Ollama (API generate o chat)
        if "/api/generate" in flow.request.path or "/api/chat" in flow.request.path:
            try:
                # 1. Estraiamo il JSON della richiesta (ciò che LOTUS ha inviato)
                req_data = json.loads(flow.request.content.decode("utf-8"))

                # 2. Estraiamo il JSON della risposta (ciò che Ollama ha restituito)
                res_data = json.loads(flow.response.content.decode("utf-8"))

                # 3. Creiamo un pacchetto ordinato con i dati utili
                # Se LOTUS usa /api/chat, i prompt sono in 'messages'. Se usa /api/generate, è in 'prompt'.
                prompt_inviato = req_data.get("messages") or req_data.get("prompt")
                risposta_ricevuta = res_data.get("message") or res_data.get("response")

                log_entry = {
                    "modello_richiesto": req_data.get("model", "sconosciuto"),
                    "prompt": prompt_inviato,
                    "risposta": risposta_ricevuta,
                    "token_usati": res_data.get("prompt_eval_count", 0) + res_data.get("eval_count", 0)
                }

                # 4. Scriviamo la riga sul nostro file di log e forziamo il salvataggio
                self.logfile.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                self.logfile.flush()

            except json.JSONDecodeError:
                # Ignoriamo le chiamate che non contengono JSON validi
                pass
            except Exception as e:
                print(f"Errore durante il parsing: {e}")

    def done(self):
        # Assicuriamoci di chiudere il file quando premiamo Ctrl+C per fermare mitmdump
        if self.logfile:
            self.logfile.close()

# Questa riga è necessaria per dire a mitmproxy di caricare la nostra classe
addons = [OllamaLogger()]