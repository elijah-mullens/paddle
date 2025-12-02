import torch
import types


def save_tensors(tensor_map: dict[str, torch.Tensor], filename: str):
    class TensorModule(torch.nn.Module):
        def __init__(self, tensors):
            super().__init__()
            for name, tensor in tensors.items():
                self.register_buffer(name, tensor)

    module = TensorModule(tensor_map)
    scripted = torch.jit.script(module)  # Needed for LibTorch compatibility
    scripted.save(filename)


if __name__ == "__main__":
    tensors = {
        "foo": torch.randn(3, 4),
        "bar": torch.randn(5, 6),
    }
    save_tensors(tensors, "foo_bar.pt")
