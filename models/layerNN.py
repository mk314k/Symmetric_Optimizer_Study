from torch import nn

class LayerNN(nn.Module):
  def __init__(self, hid_layer = 1, bias =False, input_dim=784, hidden_unit=1024, num_class=10, *args, **kwargs) -> None:
      super().__init__(*args, **kwargs)
      self.model = nn.Sequential(
          nn.Linear(input_dim, hidden_unit, bias=bias),
          nn.ReLU(),
          *[
            nn.Sequential(
                nn.Linear(hidden_unit, hidden_unit, bias=bias),
                nn.ReLU()
            ) for _ in range(hid_layer - 1)
          ],
          nn.Linear(hidden_unit, num_class, bias=bias)
      )
  def forward(self, x):
    return self.model(x).softmax(dim=-1)
