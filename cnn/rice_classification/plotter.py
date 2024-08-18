import matplotlib.pyplot as plt
from train import val_losses, train_losses

print(len(val_losses))
print(len(train_losses))

plt.plot(list(range(len(train_losses))),val_losses, label='val curve')
plt.plot(list(range(len(train_losses))),train_losses, label='loss curve')
plt.legend()
plt.show()


