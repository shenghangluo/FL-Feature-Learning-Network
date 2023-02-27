from trainer import Trainer
import config


def run_experiment():
    params = config.create_params()
    trainer = Trainer(**params)
    trainer.train_model()


if __name__ == "__main__":
    run_experiment()






