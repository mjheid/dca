from dca.train import train_with_args

args = {

    'input': '/home/kaies/csb/dca/data/twogroupsimulation/twogroupsimulation_witDropout.csv',
    'outputdir': '/home/kaies/csb/dca/',
    'normtype': 'zhang',
    'transpose': True,
    'testsplit': True,
    'type': 'zinb-conddisp',
    'threads': 1,
    'batchsize': 32,
    'sizefactors': True,
    'norminput': True,
    'loginput': False,
    'dropoutrate': '0.0',
    'batchnorm': True,
    'l2': 0.0,
    'l1': 0.0,
    'l2enc': 0.0,
    'l1enc': 0.0,
    'ridge': 0.0,
    'gradclip': 5,
    'activation': 'relu',
    'optimizer': 'RMSprop',
    'init': 'glorot_uniform',
    'epochs': 300,
    'earlystop': 15,
    'reducelr': 10,
    'hiddensize': '64,32,64',
    'inputdropout': 0.0,
    'learningrate': 0.001,
    'saveweights': False,
    'hyper': False,
    'hypern': 0,
    'debug': False,
    'tensorboard': False,
    'checkcounts': True,
    'denoisesubset': False,
    }
train_with_args(args)

""""
0: enc.0.weight Dense
1: enc.0.bias Dense
2: enc.1. 0 Batchnorm
3: enc.1. 0 Batchnorm
4: enc.1. 1 Batchnorm
5
6: bot.0.weight Dense
7: bot.0.bias Dense
8: bot.1. 0 Batchnorm
9: bot.1. 0 Batchnorm
10: bot.1. 1 Batchnorm
11
12: dec.0.weight Dense
13: dec.0.bias Dense
14: dec.1. 0 Batchnorm
14: dec.1. 0 Batchnorm
15: dec.1. 1 Batchnorm
16
17: mean.0.weight Dense
18: mean.0.bias Dense
19: disp.0.weight Dense
20: disp.0.bias Dense
21: pi.0.weight Dense
22: pi.0.bias Dense
"""
#{'loss': [177.3775634765625], 'lr': [0.001]}
#          177.37754821777344

"""
Train without dropout and test on dropout:
2Sim:
true:   max: 13763
ours:   mean: 42.677        og: mean: 62.935
        var: 19376              var: 53234.266
        max: 10882.861          max: 21378.814
6Sim:
true:   max: 23810
ours:   mean: 38.476        og: mean: 64.08282
        var: 16498              var: 59188.93
        max: 22012.385          max: 19045.924
Train on dropout and test on dropout:
true:   max: 13763
ours:   mean: 36.356        og: mean: 37.044
        var: 9941.              var: 10412.424
        max: 12914.296          max: 12862.059
6Sim:
true:   max: 23810
ours:   mean: 35.473        og: mean: 38.86944
        var: 10079              var: 15261.043
        max: 23335.285          max: 21690.371

[[0.0, -1.0, 1e-5, 1e6, 1e-4, 1e7, 1e5, 1e-6]]
[[1.0000e+00, 3.6788e-01, 1.0000e+00, 1.0000e+06, 1.0001e+00, 1.0000e+06,
         1.0000e+06, 1.0000e+00]]

[[2.8966714e-04, 5.7304740e-01, 9.5032734e-01, 1.5138752e-04,
        2.5884995e-01, 3.8045990e-05, 1.7691525e-05, 1.8388042e-04,
        1.0509115e+04, 1.0186336e-03]]
[[4.4575e-01, 3.2911e+03, 1.0000e+06, 2.2900e-03, 1.6775e+02, 1.0000e-05,
         1.0000e+06, 1.0000e-05, 5.5417e+05, 1.0000e-05]]
"""