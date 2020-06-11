from absl import app
from absl import flags
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning, RuntimeWarning))

import train_exp as train
import dsprites_dci as eval

def run_experiment(argv):
    # del argv
    # Train model
    model_dir = train.main(argv)
    # Evaluate DCI score
    eval.main(argv, model_dir=model_dir)


if __name__ == '__main__':
    app.run(run_experiment)