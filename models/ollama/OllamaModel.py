import inspect
import json
import os
import sys
import time

import httpx
import psutil
import requests
from pydantic import BaseModel
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import PurePosixPath, PureWindowsPath, Path
from typing import Iterable

import paramiko
from ollama import Client

from experiments.run_type import RunType
from models.model import Model, SubmissionError
from queries.test import Query, DirectQuery

import subprocess
import threading
from ollama import RequestError as OllamaRequestError
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

ERRORI_DI_RETE = (
    ConnectionError,                  # Quello nativo di Python (non serve import)
    httpx.ConnectError,               # Quello di 'httpx' (quando non trova il server)
    httpx.RemoteProtocolError,        # Quando il tunnel cade a metà conversazione
    httpx.ReadTimeout,                # Quando il server ci mette troppo a rispondere
    httpx.ReadError,
    OllamaRequestError                # Gli errori sollevati direttamente dal client Ollama
)

class OllamaTunnelManager:
    def __init__(self, ssh_host: str, ssh_user: str, ssh_key_path: str, local_port: int = 11434, remote_port: int = 11434):
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.ssh_key_path = ssh_key_path
        self.local_port = local_port
        self.remote_port = remote_port

        self.is_running = False
        self.process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        """Avvia il demone che tiene in vita il tunnel."""
        self._uccidi_processi_orfani()

        self.is_running = True
        self._thread = threading.Thread(target=self._keep_tunnel_alive, daemon=True)
        self._thread.start()

        # Aspettiamo un paio di secondi per dare tempo al tunnel di stabilirsi la prima volta
        time.sleep(2)
        print("Tunnel manager avviato in background.")

    def _keep_tunnel_alive(self):
        """Il ciclo infinito che controlla e riavvia SSH."""
        comando_ssh = [
            "ssh",
            "-N", # Non aprire una shell interattiva
            "-i", self.ssh_key_path,
            "-L", f"{self.local_port}:localhost:{self.remote_port}",
            "-o", "ServerAliveInterval=10", # Ping ogni 30 secondi
            "-o", "ServerAliveCountMax=3",  # Se fallisce 3 volte, chiudi la connessione
            "-o", "ExitOnForwardFailure=yes", # Esci se la porta locale è già occupata
            "-o", "StrictHostKeyChecking=no", # Evita blocchi se cambia qualcosa nell'host
            f"{self.ssh_user}@{self.ssh_host}"
        ]

        while self.is_running:
            print("[Tunnel] Avvio connessione SSH...")
            # Popen lancia il processo in background senza bloccare il thread
            self.process = subprocess.Popen(comando_ssh)

            # wait() mette in pausa QUESTO thread finché il processo ssh non muore
            self.process.wait()

            if self.is_running:
                print("[Tunnel] Tunnel caduto! Riconnessione tra 5 secondi...")
                time.sleep(5)

    def stop(self):
        """Ferma il tunnel in modo pulito alla fine dei test."""
        self.is_running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
        print("Tunnel manager fermato.")

    def _uccidi_processi_orfani(self):
        """Cerca e distrugge qualsiasi processo che sta occupando la nostra porta locale."""

        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Controlla tutte le connessioni di rete di questo processo
                for conn in proc.connections(kind='tcp'):
                    # Se il processo sta ascoltando o usando la nostra porta locale
                    if conn.laddr.port == self.local_port:
                        print(f"[Tunnel] ⚠️ Trovato processo zombie ({proc.info['name']}, PID: {proc.info['pid']}) sulla porta {self.local_port}. Terminazione in corso...")

                        proc.kill()   # Uccide il processo
                        proc.wait()   # Aspetta che sia effettivamente morto
                        time.sleep(1) # Pausa di sicurezza per far rilasciare la porta al sistema operativo
                        print("[Tunnel] Porta liberata con successo!")
                        return # Esci, lavoro finito

            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                # Ignora i processi di sistema a cui non abbiamo accesso o che stanno già morendo
                pass


def monitor_remote_running(host, user, password, file_name: str = "run.log"):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=host, username=user, password=password)

    stdin, stdout, stderr = client.exec_command(f"tail -f {file_name}")
    channel = stdout.channel

    try:
        while not channel.closed or channel.recv_ready():
            if channel.recv_ready():
                data = channel.recv(1024)
                sys.stdout.write(data.decode('utf-8'))
                sys.stdout.flush()
            else:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


def launch_remote_job(ssh_client: paramiko.SSHClient, run_file_path: str) -> None:
    venv_name = "venv_similarity"
    requirements = ["ollama", "tqdm", "requests"]
    remote_command = f"""
# Crea il venv solo se non esiste la cartella
if [ ! -d "{venv_name}" ]; then
python3 -m venv {venv_name}
fi

# Attiva il venv
source {venv_name}/bin/activate

# Installa silenziosamente (-q) i moduli necessari
pip install -q {" ".join(requirements)}

# Avvia lo script usando l'eseguibile python del venv
nohup python3 {run_file_path} > run.log 2>&1 &"""

    ssh_client.exec_command(remote_command)


def build_remote_path(os_type: str, *parts: Iterable[str]) -> str:
    if os_type == "posix":
        return str(PurePosixPath(*parts))
    elif os_type == "windows":
        return str(PureWindowsPath(*parts))
    else:
        raise ValueError("os_type must be 'posix' or 'windows'")


def send_files_ssh(client: paramiko.SSHClient, files: list[tuple[Path, str]]) -> None:
    sftp = client.open_sftp()
    for local, remote in files:
        remote_folder = os.path.dirname(remote)
        if remote_folder:
            client.exec_command(f"mkdir -p {remote_folder}")
        sftp.put(localpath=local, remotepath=remote)
    sftp.close()


def get_files_ssh(client: paramiko.SSHClient, files: list[tuple[str, Path]]) -> None:
    sftp = client.open_sftp()
    for remote, local in files:
        sftp.get(remotepath=remote, localpath=local)
    sftp.close()


def spawn_client_ssh(hostname: str, user: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(hostname=hostname, username=user, password=password)
    return client


@dataclass
class OllamaModel(Model):
    client: Client = None
    tunnel: OllamaTunnelManager = None
    ollama_address: str = "localhost:11434".replace("11434","8080")
    hostname = "***REMOVED***"
    user = "***REMOVED***"
    password = "***REMOVED***"
    supports_thinking: bool = False
    questions_file_name: str = 'questions.jsonl'
    remote_run_file_name: str = 'remote_run.py'
    @dataclass
    class Params(Model.Params):
        remote_job: bool = False
    params: Params = field(default_factory=Params)

    def _get_lotus_params(self) -> tuple[str,str]:
        return f"ollama/{self.name_api}", f"http://{self.ollama_address}"#.replace("11434","8080")

    def _init_model(self) -> None:
        if self.run_type == RunType.LOTUS and self.params.remote_job:
            raise SubmissionError("Remote job not supported for LOTUS run type.")
        if not self.params.remote_job:
            self.tunnel = OllamaTunnelManager(
                ssh_host=self.hostname,
                ssh_user=self.user,
                ssh_key_path="***REMOVED***"
            )
            self.tunnel.start()

        timeout_personalizzato = httpx.Timeout(
            connect=5.0,  # Se non riesce a connettersi al tunnel entro 5 secondi, esplode
            read=180.0,   # Dà a Ollama fino a x minuti per generare e inviare la risposta
            write=10.0,   # Tempo massimo per inviare il tuo prompt al server
            pool=10.0
        )

        self.client = Client(host="http://" + self.ollama_address, timeout=timeout_personalizzato)

    def _finish_model(self) -> None:
        self.tunnel.stop()

    def __write_questions_file(self, queries: list[DirectQuery]) -> None:
        questions_file_path = self.run_folder / self.questions_file_name

        with open(questions_file_path, 'w', encoding='utf-8') as f:
            for query in queries:
                record = {
                    "id": query.id,
                    "prompt": query.prompt,
                    "response_json_schema": query.response_json_schema
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()

    def __remote_run_folder(self, os_type: str) -> str:
        return build_remote_path(os_type, "/home/***REMOVED***", *self.run_folder.parts[self.run_folder.parts.index('runs'):])

    def __can_launch_remote(self, ssh_client: paramiko.SSHClient) -> bool:
        command = f"pgrep -f 'python.*{self.remote_run_file_name}'"
        stdin, stdout, stderr = ssh_client.exec_command(command)
        # Se stdout contiene qualcosa, significa che ha trovato il PID del processo
        pid = stdout.read().decode('utf-8').strip()
        return len(pid) == 0

    def __build_remote_python_file(self, os_type: str = "posix") -> None:
        remote_file_path = self.run_folder / self.remote_run_file_name

        curr_params = self.params.__dict__
        curr_params.update({'remote_job': True})

        ollama_model_code = inspect.getsource(OllamaModel)
        model_code = inspect.getsource(Model)
        run_type_code = inspect.getsource(self.run_type.__class__)
        #query_code = inspect.getsource(Query)
        #query_code = query_code.replace("class Query(ABC, Generic[T_QueryParameters, T_Evaluations]):", "class Query():")
        imports_code = """
import os
from abc import ABC
from enum import StrEnum, auto
from pathlib import Path
from typing import Generic, Any
import json
from io import TextIOWrapper
from dataclasses import dataclass, field
from ollama import Client
from tqdm import tqdm
import requests
"""
        run_code = f"""
if __name__ == "__main__":
    model = OllamaModel(
        name={self.name!r},
        name_path={self.name_path!r},
        name_api={self.name_api!r},
        family={self.family!r},
        max_tokens={self.max_tokens},
        run_type="{self.run_type.value}",
        run_folder=Path({self.__remote_run_folder(os_type)!r}),
        params={curr_params!r},
    )
    model._init_model()
    model._submit_direct_queries_remote()
        """

        remote_code = f"{imports_code}\n\n{run_type_code}\n\n{model_code}\n\n{ollama_model_code}\n\n{run_code}"
        # per evitare problemi di import nei file remoti, visto che non servono per l'esecuzione
        remote_code = (remote_code
                       .replace("PartitionedQuery", "Any")
                       .replace("LotusQuery", "Any")
                       .replace("DirectQuery", "Any")
                       .replace("Query", "Any")
                       .replace("DataFrame", "Any")
                       .replace("paramiko.SSHClient", "Any"))
        with open(remote_file_path, 'w', encoding='utf-8') as f:
            f.write(remote_code)
            f.flush()

    def _prepare_remote_job(self, remote_os_type: str, ssh_client: paramiko.SSHClient, queries: list[DirectQuery]) -> None:
        self.__write_questions_file(queries)
        self.__build_remote_python_file(remote_os_type)
        remote_question_path = build_remote_path(remote_os_type, "/home/***REMOVED***", self.__remote_run_folder(remote_os_type), self.questions_file_name)
        remote_run_file_path = build_remote_path(remote_os_type, "/home/***REMOVED***", self.__remote_run_folder(remote_os_type), self.remote_run_file_name)
        send_files_ssh(ssh_client, [
            (self.run_folder / self.questions_file_name, remote_question_path),
            (self.run_folder / self.remote_run_file_name, remote_run_file_path)
        ])

    @retry(
        wait=wait_fixed(10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(ERRORI_DI_RETE),
        before_sleep=lambda retry_state: print(f"Errore di connessione a Ollama. Ritento tra 10s... (Tentativo {retry_state.attempt_number})")
    )
    def __submit_prompt(self, prompt: str, schema: BaseModel|None) -> tuple[str|None,int]:
        #! con format settato, thinking è sempre False. Per solvere bisogna usare "self.client.chat", oppure "chat" (ma sembra che in quel caso non sia comunque affidabile il parametro think)
        response = self.client.generate(
            model=self.name_api,
            prompt=prompt,
            stream=False,
            think=self.supports_thinking,
            format=schema.model_json_schema() if schema else None,
            options={'temperature': self.params.temperature} if self.params.temperature != -1 else None,
        )

        resp = response.response if response.response != '' else response.thinking
        token_count = (response.prompt_eval_count or 0) + (response.eval_count or 0)

        return resp, token_count

    def _submit_direct_query_inline(self, query: DirectQuery) -> None:
        query.response, query.tokens = self.__submit_prompt(query.prompt, query.response_json_schema)

    def _submit_direct_queries_remote(self):
        questions_file_path: Path = self.run_folder / self.questions_file_name

        processed_queries = self._retrieve_queries_inline()

        queries_to_process = []
        with open(questions_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    qid = data['id']
                    prompt = data['prompt']
                    schema = data.get('response_json_schema', None)

                    if processed_queries and qid in processed_queries:
                        continue
                    queries_to_process.append((qid, prompt, schema))

        if not queries_to_process:
            print("All queries already processed.")
            return

        for qid, prompt, schema in tqdm(queries_to_process, desc=f"Submitting {self.name}", unit="query", colour='yellow'):
            response, tokens = self.__submit_prompt(prompt, schema)
            self._store_query_inline(qid, response, tokens)

    def _submit_direct_queries(self, queries: list[DirectQuery]) -> None:
        if self.params.remote_job:
            # check if already completed
            processed_queries = self._retrieve_queries_inline()
            if len(processed_queries) == len(queries):
                for q in queries:
                    if q.prompt != processed_queries[q.id]:
                        SubmissionError("Remote error: prompt mismatch for query id " + str(q.id))
                    q.response = processed_queries[q.id]["response"]
                    q.tokens = processed_queries[q.id]["tokens"]
                return
            else:
                print("Submitting remotely...")
                os_type = 'posix'
                ssh_client = spawn_client_ssh(self.hostname, self.user, self.password)
                needs_run = True

                try:
                    remote_response_file_path = build_remote_path(os_type,self.__remote_run_folder(os_type), self.responses_file_name)
                    get_files_ssh(ssh_client, [(remote_response_file_path, self.run_folder / self.responses_file_name)])
                    processed_queries = self._retrieve_queries_inline()
                    if len(processed_queries) == len(queries):
                        needs_run = False
                        print("All queries completed.")
                        for q in queries:
                            if q.prompt != processed_queries[q.id]:
                                SubmissionError("Remote error: prompt mismatch for query id " + str(q.id))
                            q.response = processed_queries[q.id]["response"]
                            q.tokens = processed_queries[q.id]["tokens"]
                        return
                except FileNotFoundError:
                    pass
                if needs_run:
                    if not self.__can_launch_remote(ssh_client):
                        ssh_client.close()
                        print("A remote job already running. Not launching this one.")
                        return
                    self._prepare_remote_job(os_type, ssh_client, queries)
                    remote_run_file_path = build_remote_path(os_type,self.__remote_run_folder(os_type), self.remote_run_file_name)
                    launch_remote_job(ssh_client, remote_run_file_path)
                    if self.params.no_waiting:
                        print("Remote job launched. Not waiting.")
                        return
                    else:
                        print("Remote job launched. Attaching to stdout...")
                        monitor_remote_running(self.hostname, self.user, self.password)
        else:
            self._submit_direct_queries_inline(queries)

    def _query_fits_limit(self, query: Query) -> bool:
        url = f"http://{self.ollama_address}/api/tokenize"
        payload = {
            "model": self.name_api,
            "prompt": query.prompt
        }
        response = requests.post(url, json=payload).json()

        n_tokens = len(response.get("tokens", []))
        return self.max_tokens >= n_tokens