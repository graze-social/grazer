import torch


class MLModel:
    @staticmethod
    def get_device():
        torch.backends.cuda.matmul.allow_tf32 = True
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    async def load_model(cls, model_name):
        return cls.get_model(model_name)
