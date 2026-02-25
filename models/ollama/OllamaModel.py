import inspect
import json
import os
import sys
import time

from tqdm import tqdm
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import PurePosixPath, PureWindowsPath, Path
from typing import Iterable

import paramiko
from ollama import Client

from models.model import Model, SubmissionError
from queries.test import Query

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
    requirements = ["ollama", "tqdm"]
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
    ollama_address: str = "localhost:11434"
    supports_thinking: bool = False
    responses_file_name: str = 'responses.jsonl'
    questions_file_name: str = 'questions.jsonl'
    remote_run_file_name: str = 'remote_run.py'
    @dataclass
    class Params(Model.Params):
        remote_job: bool = False
        pass
    params: Params = field(default_factory=Params)

    def _init_model(self) -> None:
        self.client = Client(host="http://" + self.ollama_address)

    def __write_questions_file(self, queries: list[Query]) -> None:
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
    )
    model._init_model()
    model._submit_direct_remote()
        """

        remote_code = f"{imports_code}\n\n{run_type_code}\n\n{model_code}\n\n{ollama_model_code}\n\n{run_code}"
        # per evitare problemi di import nei file remoti, visto che non servono per l'esecuzione
        remote_code = (remote_code.replace("Query", "Any")
                       .replace("DataFrame", "Any")
                       .replace("paramiko.SSHClient", "Any"))
        with open(remote_file_path, 'w', encoding='utf-8') as f:
            f.write(remote_code)
            f.flush()

    def _prepare_remote_job(self, remote_os_type: str, ssh_client: paramiko.SSHClient, queries: list[Query]) -> None:
        self.__write_questions_file(queries)
        self.__build_remote_python_file(remote_os_type)
        remote_question_path = build_remote_path(remote_os_type, "/home/***REMOVED***", self.__remote_run_folder(remote_os_type), self.questions_file_name)
        remote_run_file_path = build_remote_path(remote_os_type, "/home/***REMOVED***", self.__remote_run_folder(remote_os_type), self.remote_run_file_name)
        send_files_ssh(ssh_client, [
            (self.run_folder / self.questions_file_name, remote_question_path),
            (self.run_folder / self.remote_run_file_name, remote_run_file_path)
        ])

    def __submit_prompt(self, prompt: str, schema: dict) -> str:
        response = self.client.generate(
            model=self.name_api,
            prompt=prompt,
            stream=False,
            think=self.supports_thinking,
            format=schema,
        )

        return response.response

    def __store_response(self, f: TextIOWrapper, qid: str, response: str) -> None:
        record = {
            "id": qid,
            "response": response
        }

        f.write(json.dumps(record, ensure_ascii=False) + '\n')
        f.flush()

    def __retrieve_responses(self) -> dict[int, str]:
        response_file_path = self.run_folder / self.responses_file_name

        processed_queries = {}
        if response_file_path.exists():
            with open(response_file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        processed_queries[data['id']] = data['response']
            if len(processed_queries) > 0:
                print(f"Retrieved {len(processed_queries)} already processed answer{"s" if len(processed_queries) > 1 else ""}.")

        return processed_queries

    def _submit_direct_remote(self):
        questions_file_path: Path = self.run_folder / self.questions_file_name
        response_file_path: Path = self.run_folder / self.responses_file_name

        processed_queries = self.__retrieve_responses()

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

        with open(response_file_path, 'a', encoding='utf-8') as f:
            for qid, prompt, schema in tqdm(queries_to_process, desc=f"Submitting {self.name}", unit="query", colour='yellow'):
                response = self.__submit_prompt(prompt, schema)
                self.__store_response(f, qid, response)

    def _submit_direct_local(self, queries: list[Query]) -> None:
        response_file_path = self.run_folder / self.responses_file_name

        processed_queries: dict[int, str] = self.__retrieve_responses()

        queries_to_process = []
        for query in queries:
            if query.id in processed_queries:
                query.response = processed_queries[query.id]
            else:
                queries_to_process.append(query)

        if not queries_to_process:
            return

        with open(response_file_path, 'a', encoding='utf-8') as f:
            for query in tqdm(queries_to_process, desc=f"Querying {self.name} via Ollama", unit="query", colour='yellow'):
                query.response = self.__submit_prompt(query.prompt, query.response_json_schema)

                self.__store_response(f, query.id, query.response)


    def _submit_direct(self, queries: list[Query]) -> None:
        if self.params.remote_job:
            # check if already completed
            processed_queries = self.__retrieve_responses()
            if len(processed_queries) == len(queries):
                for q in queries:
                    if q.prompt != processed_queries[q.id]:
                        SubmissionError("Remote error: prompt mismatch for query id " + str(q.id))
                    q.response = processed_queries[q.id]
                return
            else:
                print("Submitting remotely...")
                os_type = 'posix'
                hostname = "***REMOVED***"
                user = "***REMOVED***"
                password = "***REMOVED***"
                ssh_client = spawn_client_ssh(hostname, user, password)

                try:
                    remote_response_file_path = build_remote_path(os_type,self.__remote_run_folder(os_type), self.responses_file_name)
                    get_files_ssh(ssh_client, [(remote_response_file_path, self.run_folder / self.responses_file_name)])
                    processed_queries = self.__retrieve_responses()
                    if len(processed_queries) == len(queries):
                        print("All queries completed.")
                        for q in queries:
                            if q.prompt != processed_queries[q.id]:
                                SubmissionError("Remote error: prompt mismatch for query id " + str(q.id))
                            q.response = processed_queries[q.id]
                        return
                    elif self.params.no_waiting:
                        print(f"Remote job already running but only {len(processed_queries)}/{len(queries)} queries completed. Not waiting.")
                        return
                    else:
                        print("Remote job already running but not all queries completed. Attaching to stdout...")
                        monitor_remote_running(hostname, user, password)
                except FileNotFoundError as _:
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
                        monitor_remote_running(hostname, user, password)

        else:
            self._submit_direct_local(queries)