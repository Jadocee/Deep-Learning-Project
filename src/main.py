#--------------- Thomas ---------------#
# # Packaged with - dataset.py, resnet.py, train.py
# # Author - Thomas Bandy (c3374048)

# from Util.train import Train
# from Util import util

# learn_rates = [0.0001, 0.001, 0.01]
# epochs = [5, 10, 50]
# widths = [10, 50, 100, 500]
# activation_function = ['ReLU', 'Sigmoid', 'CeLU']



# # test = Train()

# # test.prepare_data()
# # util.print_checks(test.train_data, test.valid_data, test.test_data, test.train_loader, test.valid_loader, test.test_loader)
# # test.begin_training(10, 2, 0.001)

# #--------------- Test all hyper-params ---------------#
# for (w, e, lr) in widths, epochs, learn_rates:
#     test = Train()
#     test.prepare_data()
#     util.print_checks(test.train_data, test.valid_data, test.test_data, test.train_loader, test.valid_loader, test.test_loader)
#     test.begin_training(w, e, lr)

# #--------------- Metrics ---------------#
# util.loss_acc_diagram(test.train_losses, test.val_losses, test.train_accs, test.val_accs)