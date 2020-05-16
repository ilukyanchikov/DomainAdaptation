import torch
import wandb

from trainer.trainer_adda import AddaTrainer
from models.models_adda import ADDAModel
from dataloader import create_data_generators
from metrics import AccuracyScoreFromLogits
from utils.callbacks import simple_callback, print_callback, ModelSaver, HistorySaver, WandbCallback
import configs.adda_config as adda_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    train_gen_s, val_gen_s, test_gen_s = create_data_generators(adda_config.DATASET,
                                                                adda_config.SOURCE_DOMAIN,
                                                                batch_size=adda_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                image_size=adda_config.IMAGE_SIZE,
                                                                num_workers=adda_config.NUM_WORKERS,
                                                                device=device)

    train_gen_t, val_gen_t, test_gen_t = create_data_generators(adda_config.DATASET,
                                                                adda_config.TARGET_DOMAIN,
                                                                batch_size=adda_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                image_size=adda_config.IMAGE_SIZE,
                                                                num_workers=adda_config.NUM_WORKERS,
                                                                device=device)

    acc = AccuracyScoreFromLogits()

    if adda_config.IS_PRETRAINING_STAGE:
        model = ADDAModel().to(device)
        print(model)

        tr = AddaTrainer(model, None)

        tr.fit_src_classifier(train_gen_s,
               n_epochs=adda_config.N_EPOCHS_PRETRAINING,
               validation_data=[val_gen_s, val_gen_t],
               metrics=[acc],
               steps_per_epoch=adda_config.STEPS_PER_EPOCH,
               val_freq=adda_config.VAL_FREQ,
               opt='adam',
               opt_kwargs={},
               callbacks=[print_callback(watch=["loss", 'src_metrics']),
                          ModelSaver('ADDA_PRETRAINED', adda_config.SAVE_MODEL_FREQ),
                          WandbCallback(project="adda", entity=None)])

    model = ADDAModel().to(device)
    model.load_state_dict(torch.load(adda_config.CHECKPOINT))
    tr = AddaTrainer(model, None)
    model.copy_src_to_trg()

    score_before_adapt = tr.score(val_gen_t, [acc], domain='trg')
    print(score_before_adapt)

    tr.fit_adaptation(train_gen_s, train_gen_t,
                      n_epochs=adda_config.N_EPOCHS_ADAPT,
                      validation_data=[val_gen_s, val_gen_t],
                      metrics=[acc],
                      steps_per_epoch=adda_config.STEPS_PER_EPOCH,
                      val_freq=adda_config.VAL_FREQ,
                      opt='adam',
                      opt_kwargs={},
                      callbacks=[print_callback(watch=["loss", 'd_loss', 'c_loss', 'dis_metrics']),
                                 ModelSaver('ADDA', adda_config.SAVE_MODEL_FREQ),
                                 WandbCallback(project="adda", entity=None)])

    score = tr.score(val_gen_t, [acc], domain='trg')

    print(score_before_adapt)
    print(score)

    wandb.join()
