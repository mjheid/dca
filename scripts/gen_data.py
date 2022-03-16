from dca.api import dca

dca_zinb = dca('/home/kaies/csb/dca/data/test.csv', ae_type='zinb-conddisp', normalize_per_cell=False,
                    scale=False, log1p=False, batchnorm=False, epochs=1, learning_rate=0.001, transpose=True,
                    name='test0', batch_size=1)

from dca.loss import ZINB
import tensorflow as tf

target = tf.constant([[ 20.,   0.,   0.,  50., 200.,   0.,  30.,   0.,  90.,  10.]])
mean = tf.constant([[4.4575e-01, 3.2911e+03, 1.0000e+06, 2.2900e-03, 1.6775e+02, 1.0000e-05,
         1.0000e+06, 1.0000e-05, 5.5417e+05, 1.0000e-05]])
disp = tf.constant([[3.9533e+01, 1.5560e+01, 1.1123e+01, 6.9109e+00, 3.4450e+01, 9.3383e+00,
         1.0000e-04, 2.5055e+01, 1.9007e+01, 2.5116e+01]])
drop = tf.constant([[8.1262e-16, 5.1825e-06, 9.9999e-01, 8.1314e-14, 1.0000e+00, 1.0000e+00,
         7.6104e-01, 9.7419e-01, 2.2012e-07, 1.0000e+00]])

loss = ZINB(drop, theta=disp)
l = loss.loss(target, mean)
