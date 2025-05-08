from mlflow.pyfunc import PythonModel, PythonModelContext

class ModelWrapper(PythonModel):
    def load_context(self, context: PythonModelContext):
        import pickle
        with open(context.artifacts["encoder"], "rb") as f:
            self._encoder = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self._model = pickle.load(f)

    def predict(self, context: PythonModelContext, data):
        preds = self._model.predict(data)
        return [self._encoder['decoder'][val] for val in preds]