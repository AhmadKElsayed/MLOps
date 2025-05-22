import os
import pickle

import litserve as ls
import pandas as pd
from pydantic import ValidationError

from src.deployment.requests import InferenceRequest


class InferenceAPI(ls.LitAPI):
    def setup(self, device="cpu"):
        with open(
            os.path.join("models\Titanic_Classifier_xgbclassifier\Titanic_Classifier_xgbclassifier_pipeline.pkl"),
            "rb",
        ) as pkl:
            self._model = pickle.load(pkl)


    def decode_request(self, request):
        try:
            columns = request["columns"]
            rows = request["data"]
            inference_requests = []

            for row in rows:
                row_dict = dict(zip(columns, row))
                try:
                    # Create an InferenceRequest instance and append it to the list
                    inference_request = InferenceRequest(**row_dict)
                    inference_requests.append(inference_request)
                except ValidationError as e:
                    print(f"Validation error for row {row}: {e}")
                    return {
                        "message": "Validation error",
                        "data": str(e),
                    }

            df = pd.DataFrame(rows, columns=columns)
            return df
        except Exception:
            return None

    def predict(self, x):
        print(x)
        if x is not None:
            return self._model.predict(x)
        else:
            return None

    def encode_response(self, output):
        print(output, 9 * "*")
        if output is None:
            message = "Error Occurred"
        else:
            message = "Response Produced Successfully"
        response = {
            "message": message,
            "data": output.tolist(),
        }
        return response