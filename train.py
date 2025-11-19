
import argparse
import rbm_model
import data



train_dataloader, val_dataloader = data.load_data()

def train(epochs, cdk):
  err_metrics = {'training_err': [],
                  'validation_err': []}

  model = rbm_model.RBM(n_vis = 784, n_hid = 256, k = cdk, lr = 1e-3)
  for epoch in range(epochs):
    training_err = 0
    for x_train, y_train in train_dataloader:
      stats = model.forward(x_train)
      model.update(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5])
      err = model.reconstruction_error(stats[2], stats[3])
      training_err += err
      err_metrics['training_err'].append(err)

    val_err = 0
    for x_val, y_val in val_dataloader:
      stats = model.forward(x_val)
      err = model.reconstruction_error(stats[2], stats[3])
      val_err += err
      err_metrics['validation_err'].append(err)

    print(f'Epoch {epoch} - training_err: {10 * training_err/len(train_dataloader):.5f} - validation_err: {10 * val_err/len(val_dataloader):.5f}')

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--epochs", type = int, default = 50, help = "Number of Epochs")
  parser.add_argument("--cdk", type = int, default = 5, help = "Number of steps for contrastive divergence, k.")
  args = parser.parse_args()
  train(args.epochs, args.cdk)











