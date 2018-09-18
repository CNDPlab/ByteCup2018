import torch as t


def loss_function(inputs, targets):
    vocabulary_size = inputs.size()[-1]
    inputs = inputs.view(-1, vocabulary_size)
    targets = targets[:, 1:].contiguous()
    targets = targets.view(-1)
    loss = t.nn.functional.cross_entropy(inputs, targets, ignore_index=0,)
    return loss
