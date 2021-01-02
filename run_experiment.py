from absl import app
from absl import flags
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, DeprecationWarning, RuntimeWarning))

import train_exp as train
import dsprites_dci as dci
import factorvae_metric as sens

FLAGS = flags.FLAGS

flags.DEFINE_enum('eval_type', 'dci', ['dci', 'sens'], 'Evaluation metric to use')

def run_experiment(argv):
    # del argv
    # Train model
    model_dir = train.main(argv)
    # Evaluate metric
    if FLAGS.eval_type == 'dci':
        dci.main(argv, model_dir=model_dir)
    elif FLAGS.eval_type == 'sens':
        sens.main(argv, model_dir=model_dir)


if __name__ == '__main__':
    app.run(run_experiment)