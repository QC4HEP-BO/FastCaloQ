import json, time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from HighLevelFeatures import HighLevelFeatures
import matplotlib.pyplot as plt
from pdb import set_trace
import os
torch.set_default_dtype(torch.float32)
device = torch.device("cpu")

def ttv_split(data1, data2, split=np.array([0.6, 0.2, 0.2])):
    """ splits data1 and data2 in train/test/val according to split,
        returns shuffled and merged arrays
    """
    assert len(data1) == len(data2)
    num_events = (len(data1) * split).astype(int)
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    train1, test1, val1 = np.split(data1, num_events.cumsum()[:-1])
    train2, test2, val2 = np.split(data2, num_events.cumsum()[:-1])
    train = np.concatenate([train1, train2], axis=0)
    test = np.concatenate([test1, test2], axis=0)
    val = np.concatenate([val1, val2], axis=0)
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)
    return train, test, val

def prepare_high_data_for_classifier(hlf_class, voxel, label, E_inc):
    hlf_class.CalculateFeatures(voxel)
    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)], axis=1)
    return ret

class DNN(torch.nn.Module):
    """ NN for vanilla classifier. Does not have sigmoid activation in last layer, should
        be used with torch.nn.BCEWithLogitsLoss()
    """
    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """ Forward pass through the DNN """
        x = self.layers(x)
        return x

def load_classifier(constructed_model, parser_args):
    """ loads a saved model """
    filename = f'{parser_args.mode}_{parser_args.ckpt}_{parser_args.dataset}.pt'
    checkpoint = torch.load(os.path.join(parser_args.output_dir, filename),
                            map_location=parser_args.device)
    constructed_model.load_state_dict(checkpoint['model_state_dict'])
    constructed_model.to(parser_args.device)
    constructed_model.eval()
    return constructed_model


def train_and_evaluate_cls(model, data_train, data_test, optim, args):
    """ train the model and evaluate along the way"""
    best_eval_acc = float('-inf')
    args.best_epoch = -1
    loss_history = {
        'epoch': [],
        'train_loss': [],
        'eval_acc': [],
        'eval_auc': [],
        'eval_JSD': [],
    }
    try:
        for i in range(args.cls_n_epochs):
            train_loss = train_cls(model, data_train, optim, i, args)
            loss_history['epoch'].append(i)
            loss_history['train_loss'].append(train_loss)
            with torch.no_grad():
                eval_acc, eval_auc, JSD = evaluate_cls(model, data_test, args)
                loss_history['eval_acc'].append(eval_acc.item())
                loss_history['eval_auc'].append(eval_auc.item())
                loss_history['eval_JSD'].append(JSD.item())
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                args.best_epoch = i+1
                filename = f'{args.mode}_{args.ckpt}_{args.dataset}.pt'
                os.makedirs(args.output_dir, exist_ok=True)
                torch.save({'model_state_dict':model.state_dict()},
                           os.path.join(args.output_dir, filename))
            if eval_acc == 1.:
                break
    except KeyboardInterrupt:
        # training can be cut short with ctrl+c, for example if overfitting between train/test set
        # is clearly visible
        pass
    with open(os.path.join(args.output_dir, f'loss_{args.ckpt}.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)
    plt.plot(loss_history['epoch'], loss_history['train_loss'], label='Train loss')
    plt.plot(loss_history['epoch'], loss_history['eval_auc'], label='Eval AUC')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (AUC)")
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, f'loss_{args.ckpt}.pdf'))


def train_cls(model, data_train, optim, epoch, arg):
    """ train one step """
    model.train()
    for i, data_batch in enumerate(data_train):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        #input_vector, target_vector = torch.split(data_batch, [data_batch.size()[1]-1, 1], dim=1)
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector.float())
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optim.zero_grad()
        loss.backward()
        optim.step()

        #if i % (len(data_train)//2) == 0:
        #    print('Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
        #        epoch+1, arg.cls_n_epochs, i, len(data_train), loss.item()))
        # PREDICTIONS
        pred = torch.round(torch.sigmoid(output_vector.detach()))
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    #print("Accuracy on training set is", i, accuracy_score(res_true.cpu(), res_pred.cpu()))
    return loss.item()

def train_evaluate_classifier(args, train_data, val_data, test_data):
    # set up DNN classifier
    input_dim = train_data.shape[1]-1
    DNN_kwargs = {'num_layer':args.cls_n_layer,
                  'num_hidden':args.cls_n_hidden,
                  'input_dim':input_dim,
                  'dropout_probability':args.cls_dropout_probability}
    classifier = DNN(**DNN_kwargs)
    classifier.to(args.device)
    #print(classifier)
    total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    print("{} has {} parameters".format(args.mode, int(total_parameters)))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)

    train_data = TensorDataset(torch.tensor(train_data).to(args.device))
    test_data = TensorDataset(torch.tensor(test_data).to(args.device))
    val_data = TensorDataset(torch.tensor(val_data).to(args.device))

    train_dataloader = DataLoader(train_data, batch_size=args.cls_batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=args.cls_batch_size, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=args.cls_batch_size, shuffle=False)

    train_and_evaluate_cls(classifier, train_dataloader, test_dataloader, optimizer, args)
    classifier = load_classifier(classifier, args)

    with torch.no_grad():
        eval_acc, eval_auc, eval_JSD = evaluate_cls(classifier, val_dataloader, args,
                                                    final_eval=True,
                                                    calibration_data=test_dataloader)
    print("Final result of classifier test (AUC / JSD): {:.4f} / {:.4f}".format(eval_auc, eval_JSD))
    return eval_acc, eval_auc, eval_JSD


def evaluate_cls(model, data_test, arg, final_eval=False, calibration_data=None):
    """ evaluate on test set """
    model.eval()
    for j, data_batch in enumerate(data_test):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector.float())
        pred = output_vector.reshape(-1)
        target = target_vector.float()
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
    result_pred = torch.sigmoid(result_pred).cpu().numpy()
    result_true = result_true.cpu().numpy()
    eval_acc = accuracy_score(result_true, np.round(result_pred))
    eval_auc = roc_auc_score(result_true, result_pred)
    JSD = - BCE + np.log(2.)
    #print("Test Accuracy", eval_acc, "Test AUC", eval_auc, "Test BCE loss {:.4f}, JSD of two dists {:.4f}".format(BCE, JSD/np.log(2.)))
    if final_eval:
        prob_true, prob_pred = calibration_curve(result_true, result_pred, n_bins=10)
        calibrator = calibrate_classifier(model, calibration_data, arg)
        rescaled_pred = calibrator.predict(result_pred)
        eval_acc = accuracy_score(result_true, np.round(rescaled_pred))
        eval_auc = roc_auc_score(result_true, rescaled_pred)
        prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
        # calibration was done after sigmoid, therefore only BCELoss() needed here:
        BCE = torch.nn.BCELoss()(torch.tensor(rescaled_pred), torch.tensor(result_true))
        JSD = - BCE.cpu().numpy() + np.log(2.)
        otp_str = "rescaled BCE loss of test set is {:.4f}, "+\
            "rescaled JSD of the two dists is {:.4f}"
    return eval_acc, eval_auc, JSD/np.log(2.)

def calibrate_classifier(model, calibration_data, arg):
    """ reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, ("Need calibration data for calibration!")
    for j, data_batch in enumerate(calibration_data):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector.float())
        pred = torch.sigmoid(output_vector).reshape(-1)
        target = target_vector.to(torch.float64)
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    iso_reg = IsotonicRegression(out_of_bounds='clip', y_min=1e-6, y_max=1.-1e-6).fit(result_pred,
                                                                                      result_true)
    return iso_reg
