import torch
from trainer.logger import AvgLossLogger
from torch import nn


# TODO loss вынести в отдельную функцию или нет
class AddaTrainer:
    def __init__(self, model, loss):
        self.model = model
        self.loss = loss
        # self.epoch = 0
        self.device = next(self.model.parameters()).device
        self.loss_logger = AvgLossLogger()

    def fit_src_classifier(self, src_data, n_epochs=1000, steps_per_epoch=100, val_freq=1,
                           opt='adam', opt_kwargs=None, validation_data=None, metrics=None, callbacks=None):
        self.model.src_cnn.train()
        self.model.class_classifier.train()

        parameters = [*self.model.src_cnn.parameters(), *self.model.class_classifier.parameters()]
        if opt == 'adam':
            optimizer = torch.optim.Adam(parameters, **opt_kwargs)
        else:
            raise ValueError('Optimazer not allowed {}. Only adam is available'.format(opt))

        criterion = nn.CrossEntropyLoss()

        if validation_data is not None:
            src_val_data, trg_val_data = validation_data

        for i in range(n_epochs):
            self.loss_logger.reset_history()
            for step, (src_batch) in enumerate(src_data):
                if step == steps_per_epoch:
                    break
                src_images, src_classes = src_batch
                pred = self.model.class_classifier(self.model.src_cnn(src_images))
                loss = criterion(pred, src_classes)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # TODO callback
                # TODO validation

            # validation
            src_metrics = None
            trg_metrics = None
            if i % val_freq == 0 and validation_data is not None:
                self.model.eval()

                # calculating metrics on validation
                if metrics is not None and validation_data is not None:
                        src_metrics = self.score(src_val_data, metrics, domain='src')
                        trg_metrics = self.score(trg_val_data, metrics, domain='src')

            if callbacks is not None:
                epoch_log = {
                    'loss': loss.data.cpu().item(),
                    'src_metrics': src_metrics,
                    'trg_metrics': trg_metrics
                }
                for callback in callbacks:
                    callback(self.model, epoch_log, i, n_epochs)


    def fit_adaptation(self, src_data, trg_data, n_epochs=1000, steps_per_epoch=100, val_freq=1,
                           opt='adam', opt_kwargs=None, validation_data=None, metrics=None, callbacks=None):
        self.model.trg_cnn.train()
        self.model.discriminator.train()

        discriminator_parameters = self.model.discriminator.parameters()
        trg_cnn_parameters = self.model.trg_cnn.parameters()
        if opt == 'adam':
            discriminator_optimizer = torch.optim.Adam(discriminator_parameters, **opt_kwargs)
            trg_cnn_optimizer = torch.optim.Adam(trg_cnn_parameters, **opt_kwargs)
        else:
            raise ValueError('Optimizer not allowed {}. Only adam is available'.format(opt))

        criterion = nn.CrossEntropyLoss()

        if validation_data is not None:
            src_val_data, trg_val_data = validation_data

        for i in range(n_epochs):
            self.model.trg_cnn.train()
            self.model.discriminator.train()
            self.loss_logger.reset_history()
            for step, (src_batch, trg_batch) in enumerate(zip(src_data, trg_data)):
                if step == steps_per_epoch:
                    break
                src_images, _ = src_batch
                trg_images, _ = trg_batch

                # discriminator step

                pred = self.model.predict_domain(src_images, trg_images)

                domain_labels = self._get_domain_labels(src_images, trg_images)

                loss = criterion(pred, domain_labels)

                discriminator_optimizer.zero_grad()
                loss.backward()
                discriminator_optimizer.step()

                #trg cnn step

                discriminator_optimizer.zero_grad()
                trg_cnn_optimizer.zero_grad()

                features_trg = self.model.trg_cnn(trg_images)
                pred_trg = self.model.discriminator(features_trg)
                fake_label_trg = self._get_domain_labels(trg_images, [])

                loss_cnn = criterion(pred_trg, fake_label_trg)

                trg_cnn_optimizer.zero_grad()
                loss_cnn.backward()
                trg_cnn_optimizer.step()

                # TODO callback
                # TODO validation

            dis_metrics = None
            trg_metrics = None
            if i % val_freq == 0 and validation_data is not None:
                # calculating metrics on validation
                if metrics is not None and validation_data is not None:
                        dis_metrics = self.dis_score(src_val_data, trg_val_data, metrics)
                        trg_metrics = self.score(trg_val_data, metrics, domain='trg')

            if callbacks is not None:
                epoch_log = {'d_loss': loss.data.cpu().item(),
                             'c_loss': loss_cnn.data.cpu().item(),
                             'dis_metrics': dis_metrics,
                             'trg_metrics': trg_metrics}

                for callback in callbacks:
                    callback(self.model, epoch_log, i, n_epochs)

    def dis_score(self, src_data, trg_data, metrics):
        self.model.eval()

        for metric in metrics:
            metric.reset()

        src_data.reload_iterator()
        trg_data.reload_iterator()
        for (src_images, _), (trg_images, _) in zip(src_data, trg_data):
            pred = self.model.predict_domain(src_images, trg_images)
            domain_labels = self._get_domain_labels(src_images, trg_images)

            for metric in metrics:
                metric(domain_labels, pred)
        src_data.reload_iterator()
        trg_data.reload_iterator()
        return {metric.name: metric.score for metric in metrics}

    def _get_domain_labels(self, src_images, trg_images):
        source_len = len(src_images)
        target_len = len(trg_images)
        is_target_on_src = torch.ones(source_len, dtype=torch.long, device=self.device)
        is_target_on_trg = torch.zeros(target_len, dtype=torch.long, device=self.device)
        domain_labels = torch.cat((is_target_on_src, is_target_on_trg), 0)
        return domain_labels

    def score(self, data, metrics, domain='trg'):
        self.model.eval()

        for metric in metrics:
            metric.reset()

        data.reload_iterator()
        for images, true_classes in data:
            pred_classes = self.model.predict(images, domain=domain)
            for metric in metrics:
                metric(true_classes, pred_classes)
        data.reload_iterator()
        return {metric.name: metric.score for metric in metrics}

    def predict(self, data, domain='trg'):
        predictions = []
        for batch in data:
            predictions.append(self.model.predict(batch, domain=domain))
        return torch.cat(predictions)