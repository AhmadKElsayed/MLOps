import litserve as ls
from src.deployment.api import InferenceAPI

if __name__ == "__main__":

    api = InferenceAPI()
    server = ls.LitServer(api, accelerator="cpu")
    server.run(port=8000, generate_client_file=False)