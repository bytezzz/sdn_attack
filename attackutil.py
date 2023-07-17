import torchattacks

import torchattacks
import torch
import torch.nn.functional as F

def evaluate_sdn_attack(model, test_loader, atk):
    atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_acc = 0
    n = 0
    model.eval()
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to('cuda'), y.to('cuda')

        if model.use_rpf:
          model.random_rp_matrix()

        X_adv = atk(X, y)  # advtorch

        if model.use_rpf:
          model.random_rp_matrix()

        with torch.no_grad():
            output = model(X_adv)
        test_acc += (output.max(1)[1] == y).sum().item()
        n += y.size(0)

    pgd_acc = test_acc / n
    return pgd_acc

