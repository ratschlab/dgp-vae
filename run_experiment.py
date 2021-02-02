from absl import app
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning, RuntimeWarning))

import train_exp as train
import dsprites_dci as dci

def run_experiment(argv):
    # del argv
    # Train model
    model_dir = train.main(argv)
    # Evaluate metric
    dci.main(argv, model_dir=model_dir)

if __name__ == '__main__':
    app.run(run_experiment)