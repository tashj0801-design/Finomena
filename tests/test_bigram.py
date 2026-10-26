import torch

from finomena.models.bigram import BigramLanguageModel


def test_bigram_forward_backward_and_update() -> None:
    model = BigramLanguageModel(vocab_size=8)
    inputs = torch.randint(0, 8, (2, 5))
    targets = torch.randint(0, 8, (2, 5))
    before = model.next_character.weight.detach().clone()
    logits, loss = model(inputs, targets)
    assert logits.shape == (2, 5, 8)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(model.next_character.weight.grad).all()
    torch.optim.AdamW(model.parameters(), lr=0.01).step()
    assert not torch.equal(before, model.next_character.weight)
